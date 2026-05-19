"""
anthropic_model.py
==================
Anthropic (via Microsoft Foundry) drop-in replacement for Model.

Differences from the OpenAI Model:
  - Uses AnthropicFoundry client (sync, wrapped in asyncio executor)
  - Tools use Anthropic schema: {name, description, input_schema}
  - Tool calls come back as content blocks with type="tool_use"
  - Tool results are sent as role="user" content blocks (not role="tool"),
    and consecutive results are coalesced into one user message
  - System messages are extracted and sent as the top-level `system` param
  - Usage keys are input_tokens / output_tokens (not prompt_ / completion_)

This module has no knowledge of any specific MCP server, workflow, or task domain.
System prompt injection is handled externally via set_system_prompt().
"""

import asyncio
import json
import os
import time
from functools import partial
from typing import Optional

from anthropic import AnthropicFoundry

from model import Model, ModelResponse


# ---------------------------------------------------------------------------
# Small shim objects so app.py can treat Anthropic responses the same way
# it treats OpenAI responses (i.e. access .tool_calls, .content, .id etc.)
# ---------------------------------------------------------------------------

class _ToolCall:
    """Wraps an Anthropic tool_use block to look like an OpenAI tool call."""
    def __init__(self, block):
        self.id   = block.id
        self.type = "function"
        self.function = _Function(block.name, block.input)


class _Function:
    def __init__(self, name: str, input_dict: dict):
        self.name      = name
        # app.py calls json.loads(tc.function.arguments) — keep compatible
        self.arguments = json.dumps(input_dict)


class _Message:
    """
    Wraps an Anthropic response so app.py can do:
        msg.content          → raw list of blocks (for add_assistant_tool_calls)
        msg.tool_calls       → list[_ToolCall] or None
        msg.text             → concatenated text from all text blocks
    """
    def __init__(self, response):
        self._response  = response
        self.tool_calls = None

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if tool_use_blocks:
            self.tool_calls = [_ToolCall(b) for b in tool_use_blocks]

        self.content = response.content

    @property
    def text(self) -> str:
        """Return concatenated text from all text blocks."""
        return "".join(
            b.text for b in self._response.content if b.type == "text"
        )


# ---------------------------------------------------------------------------
# AnthropicModel
# ---------------------------------------------------------------------------

class AnthropicModel(Model):
    """
    Drop-in replacement for Model that talks to Anthropic via MS Foundry.

    Inherits all history management from Model. Overrides only the
    provider-specific parts: client setup, tool schema conversion,
    message serialisation, and the API call itself.
    """

    def __init__(
        self,
        system_prompt  : Optional[str] = None,
        endpoint       : Optional[str] = None,
        api_key        : Optional[str] = None,
        deployment_name: Optional[str] = None,
    ):
        # Deliberately do NOT call super().__init__() — that creates an
        # AsyncOpenAI client we don't need. Replicate only what we use from Model.
        self.deployment = (
            deployment_name
            or os.environ["ANTHROPIC_FOUNDRY_DEPLOYMENT"]
        )
        _api_key  = api_key  or os.environ["ANTHROPIC_FOUNDRY_API_KEY"]
        _endpoint = endpoint or os.environ["ANTHROPIC_FOUNDRY_ENDPOINT"]

        self._anthropic = AnthropicFoundry(
            api_key  = _api_key,
            base_url = _endpoint,
        )

        self._summary: Optional[str] = None

        self.messages: list[dict] = (
            [{"role": "system", "content": system_prompt}]
            if system_prompt
            else []
        )

    # ------------------------------------------------------------------
    # set_system_prompt, inject_system_message, add_user_message,
    # add_assistant_message, _trim_history, _summarise_dropped,
    # _build_summary_msg — all inherited from Model unchanged.
    # ------------------------------------------------------------------

    def add_assistant_tool_calls(self, msg: _Message):
        """
        Store Anthropic tool-use content blocks in history.
        Anthropic expects the exact content block objects back in the next
        request, so we store them as-is under role='assistant'.
        """
        self.messages.append({
            "role"   : "assistant",
            "content": msg.content,
        })

    def add_tool_result(self, tool_call_id: str, content: str):
        """
        Anthropic tool results go as role='user' with type='tool_result'.
        Consecutive tool results are coalesced into one user message —
        Anthropic requires this for multi-tool turns.
        """
        result_block = {
            "type"       : "tool_result",
            "tool_use_id": tool_call_id,
            "content"    : content,
        }

        if (
            self.messages
            and self.messages[-1]["role"] == "user"
            and isinstance(self.messages[-1]["content"], list)
            and self.messages[-1]["content"]
            and self.messages[-1]["content"][0].get("type") == "tool_result"
        ):
            self.messages[-1]["content"].append(result_block)
        else:
            self.messages.append({
                "role"   : "user",
                "content": [result_block],
            })

    # ------------------------------------------------------------------
    # Tool schema conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _openai_tools_to_anthropic(tools: list[dict]) -> list[dict]:
        """
        Convert OpenAI-style tool defs to Anthropic format.

        OpenAI:
            {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}

        Anthropic:
            {"name": ..., "description": ..., "input_schema": {...}}
        """
        out = []
        for t in tools:
            fn = t.get("function", t)
            out.append({
                "name"        : fn["name"],
                "description" : fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return out

    # ------------------------------------------------------------------
    # System message extraction for Anthropic's top-level `system` param
    # ------------------------------------------------------------------

    def _build_anthropic_messages(self, trimmed: list[dict]) -> tuple[str, list[dict]]:
        """
        Split trimmed history into:
          - system_text : str  (concatenated system message contents)
          - messages    : list (only non-system messages)

        Anthropic does not allow system messages in the messages array.
        """
        system_parts = []
        conversation = []

        for m in trimmed:
            if m.get("role") == "system":
                content = m.get("content", "")
                if isinstance(content, str):
                    system_parts.append(content)
                elif isinstance(content, list):
                    system_parts.append(
                        " ".join(b.get("text", "") for b in content if isinstance(b, dict))
                    )
            else:
                conversation.append(m)

        return "\n\n".join(system_parts), conversation

    # ------------------------------------------------------------------
    # Main call
    # ------------------------------------------------------------------

    async def get_response(self, tools: Optional[list[dict]] = None) -> ModelResponse:
        """Send current messages to Anthropic and return a ModelResponse."""

        anthropic_tools = None
        if tools:
            clean = [{k: v for k, v in t.items() if k != "_server"} for t in tools]
            anthropic_tools = self._openai_tools_to_anthropic(clean)

        trimmed                = self._trim_history()
        system_text, conv_msgs = self._build_anthropic_messages(trimmed)

        kwargs: dict = {
            "model"     : self.deployment,
            "max_tokens": 8096,
            "messages"  : conv_msgs,
        }
        if system_text:
            kwargs["system"] = system_text
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        loop  = asyncio.get_event_loop()
        start = time.perf_counter()
        response = await loop.run_in_executor(
            None,
            partial(self._anthropic.messages.create, **kwargs),
        )
        latency_ms = (time.perf_counter() - start) * 1000

        return ModelResponse(
            message           = _Message(response),
            prompt_tokens     = response.usage.input_tokens,
            completion_tokens = response.usage.output_tokens,
            latency_ms        = latency_ms,
        )