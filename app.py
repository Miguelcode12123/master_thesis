"""
app.py
======
Application logic — orchestrates the MCP client, model, logger,
and the agentic tool-call loop.

Multi-server: App receives a list of server configs and passes them to
MCPClient. The merged tool/resource pool is transparent to this layer.

Provider-agnostic: Reads MODEL_PROVIDER from the environment and instantiates
either Model (OpenAI/Azure) or AnthropicModel (Anthropic via MS Foundry).
The agentic loop works identically for both.
"""

import json
import asyncio
import os

from logger import MCPLogger
from mcp_client import MCPClient


def _build_model(endpoint=None, api_key=None, deployment=None):
    """
    Instantiate the correct model class based on MODEL_PROVIDER env var.
    No system prompt is passed at construction — it is injected after
    connect() via model.set_system_prompt() if the MCP server provides one.
    """
    provider = os.environ.get("MODEL_PROVIDER", "openai").lower()
    if provider == "anthropic":
        from anthropic_model import AnthropicModel
        return AnthropicModel(
            endpoint        = endpoint,
            api_key         = api_key,
            deployment_name = deployment,
            system_prompt   = None,
        )
    else:
        from model import Model
        return Model(
            endpoint      = endpoint,
            api_key       = api_key,
            deployment    = deployment,
            system_prompt = None,
        )


def _get_response_text(msg) -> str:
    """
    Extract the final text from a model response message.
    - OpenAI: msg.content is a plain string.
    - Anthropic: msg.text concatenates all text blocks.
    """
    if hasattr(msg, "text"):
        return msg.text or ""
    return msg.content or ""


def _get_content_preview(msg) -> str:
    """
    Safe content preview for logging.
    Returns a string regardless of whether content is a str or a list of blocks.
    """
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            getattr(b, "text", "") for b in content if hasattr(b, "type") and b.type == "text"
        )
    return ""


async def _inject_resources(query: str, client: MCPClient) -> tuple[str, str]:
    """
    Parse @mentions, fetch matching resources, return (cleaned_query, context_block).

    Two cases:
      @pages          → exact match against URI suffix "pages"
      @pages/page_001 → reconstruct scheme://pages/page_001 and read it

    Works across all connected servers.
    """
    mentions = [w[1:] for w in query.split() if w.startswith("@")]
    cleaned  = " ".join(w for w in query.split() if not w.startswith("@")).strip()

    if not mentions:
        return query, ""

    exact = {
        r["uri"].split("://", 1)[-1]: r["uri"]
        for r in client.resources
        if "{" not in r["uri"]
    }

    scheme_by_prefix: dict[str, str] = {}
    for r in client.resources:
        if "://" in r["uri"] and "{" not in r["uri"]:
            scheme   = r["uri"].split("://")[0]
            prefix   = r["uri"].split("://", 1)[-1].split("/")[0]
            scheme_by_prefix[prefix] = scheme

    fetched = []
    for mention in mentions:
        if mention in exact:
            uri = exact[mention]
        elif "/" in mention:
            parent = mention.split("/")[0]
            scheme = scheme_by_prefix.get(parent, "company")
            uri    = f"{scheme}://{mention}"
        else:
            continue

        try:
            content = await client.read_resource(uri)
            content_str = (
                json.dumps(content, indent=2)
                if isinstance(content, (dict, list))
                else str(content)
            )
            fetched.append((mention, content_str))
        except Exception:
            fetched.append((
                mention,
                f"RESOURCE_NOT_FOUND: '{mention}' does not exist. "
                "Do not call any tool to retry this lookup — tell the user the resource was not found."
            ))

    context_block = "".join(
        f'\n<resource id="{rid}">\n{content}\n</resource>\n'
        for rid, content in fetched
    )

    return cleaned, context_block


class App:

    def __init__(
        self,
        endpoint   : str,
        api_key    : str,
        deployment : str,
        servers    : list[dict],
    ):
        self.model  = _build_model(endpoint=endpoint, api_key=api_key, deployment=deployment)
        self.client = MCPClient(servers=servers)
        self.logger = MCPLogger(log_dir="./logs")

    async def start(self) -> dict:
        result = await self.client.connect()
        if self.client.system_prompt:
            self.model.set_system_prompt(self.client.system_prompt)
        return result

    async def stop(self):
        await self.client.disconnect()

    async def inject_prompt_as_task(self, prompt_messages: list):
        """
        Entry point for MCP prompt injections (e.g. /some_prompt command).

        Accepts a list of PromptMessage objects as returned by the MCP prompt spec.
        Each message has a `role` ('assistant' or 'user') and a `content` object
        with a `.text` attribute.

          - assistant role → injected as a system-level context message
          - user role      → added as the triggering user message

        This is fully general: any MCP prompt returning the standard role
        structure is handled here with zero changes to app.py.
        """
        first_user = next(
            (m for m in prompt_messages if getattr(m, "role", None) == "user"),
            None,
        )
        label = (getattr(getattr(first_user, "content", None), "text", "") or "")[:120]
        self.logger.start_session(user_query="[prompt] " + label, model=self.model.deployment)

        for msg in prompt_messages:
            role = getattr(msg, "role", None)
            text = getattr(getattr(msg, "content", None), "text", "") or ""
            if role == "assistant":
                self.model.inject_system_message(text)
            elif role == "user":
                self.model.add_user_message(text)

        await self._run_agentic_loop()

    async def handle_query(self, user_input: str):
        """Entry point for plain conversational queries."""
        self.logger.start_session(user_query=user_input, model=self.model.deployment)

        cleaned_query, context_block = await _inject_resources(user_input, self.client)
        full_message = context_block + cleaned_query if context_block else cleaned_query
        self.model.add_user_message(full_message)

        await self._run_agentic_loop()

    async def _run_agentic_loop(self):
        """
        Core tool-call loop. Runs until the model produces a final answer
        (no tool calls). Called by both handle_query() and inject_prompt_as_task()
        after the appropriate messages have been added to model history.

        Provider-agnostic: uses _get_response_text() and _get_content_preview()
        so it works with both OpenAI and Anthropic response shapes.
        """
        final_answer    = ""
        active_tools    = self.client.tools
        tools_available = [t["function"]["name"] for t in active_tools]
        empty_retries = 0

        try:
            while True:
                self.logger.increment_turn()

                resp = await self.model.get_response(tools=active_tools)
                msg  = resp.message

                has_tool_calls = bool(msg.tool_calls)
                response_type  = "tool_call" if has_tool_calls else "final_answer"

                self.logger.log_llm_call(
                    model=self.model.deployment,
                    messages_count=len(self.model.messages),
                    tools_available=tools_available,
                    response_type=response_type,
                    tool_calls_made=[tc.function.name for tc in msg.tool_calls] if has_tool_calls else [],
                    content_preview=_get_content_preview(msg),
                    prompt_tokens=resp.prompt_tokens,
                    completion_tokens=resp.completion_tokens,
                    latency_ms=resp.latency_ms,
                )

                if not has_tool_calls:
                    final_answer = _get_response_text(msg)

                    if not final_answer.strip():

                        empty_retries += 1
                        if empty_retries >= 3:
                            print("\n[ERROR] Model returned empty response 3 times. Aborting.\n")
                            break
                        
                        print("\n[WARN] Empty response received — waiting 15s before retry...\n")
                        await asyncio.sleep(15)
                        continue 

                    print(f"\nAssistant: {final_answer}\n")
                    self.model.add_assistant_message(final_answer)
                    break

                self.model.add_assistant_tool_calls(msg)

                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    result, latency_ms = await self.client.call_tool(tc.function.name, args)

                    self.logger.log_tool_call(
                        server_name=self.client._tool_router.get(tc.function.name, "unknown"),
                        tool_name=tc.function.name,
                        arguments=args,
                        result=result,
                        success=True,
                        latency_ms=latency_ms,
                    )

                    self.model.add_tool_result(tc.id, result)

            self.logger.end_session(final_answer=final_answer, success=True)

        except Exception as e:
            self.logger.end_session(final_answer="", success=False, error=str(e))
            raise