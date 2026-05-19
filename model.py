"""
model.py
========
Generic OpenAI/Azure LLM wrapper.

Responsibilities:
  - Maintain conversation history
  - Manage a sliding-window trim with compact summarisation when history grows large
  - Send requests to the OpenAI-compatible API and return a ModelResponse

This module has no knowledge of any specific MCP server, workflow, or task domain.
System prompt injection is handled externally via set_system_prompt().
"""

import json
import os
import time
from typing import Optional
from openai import AsyncOpenAI

MAX_HISTORY_MESSAGES = 20


class ModelResponse:
    """Wraps the LLM response together with usage stats and latency."""
    def __init__(self, message, prompt_tokens: int, completion_tokens: int, latency_ms: float):
        self.message           = message
        self.prompt_tokens     = prompt_tokens
        self.completion_tokens = completion_tokens
        self.latency_ms        = latency_ms


class Model:
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        endpoint     : Optional[str] = None,
        api_key      : Optional[str] = None,
        deployment   : Optional[str] = None,
    ):
        self.endpoint   = (endpoint or os.environ["AZURE_OPENAI_ENDPOINT"]).rstrip("/") + "/"
        self.api_key    = api_key    or os.environ["AZURE_OPENAI_API_KEY"]
        self.deployment = deployment or os.environ["AZURE_OPENAI_DEPLOYMENT"]

        self.client = AsyncOpenAI(
            base_url=self.endpoint,
            api_key =self.api_key,
        )

        self._summary: Optional[str] = None

        self.messages: list[dict] = (
            [{"role": "system", "content": system_prompt}]
            if system_prompt
            else []
        )

    def set_system_prompt(self, prompt: str):
        """
        Inject or replace the system prompt after construction.
        Replaces any existing system message at index 0, or prepends one if none exists.
        """
        system_msg = {"role": "system", "content": prompt}
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0] = system_msg
        else:
            self.messages.insert(0, system_msg)

    def inject_system_message(self, content: str, index: int = 1):
        """
        Insert an arbitrary system message at the given position in history.
        Useful for injecting context or instructions at runtime without
        overwriting the primary system prompt.
        Defaults to index 1 (immediately after the system prompt).
        """
        self.messages.insert(index, {"role": "system", "content": content})

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def add_assistant_tool_calls(self, msg):
        self.messages.append({
            "role"      : "assistant",
            "content"   : msg.content,
            "tool_calls": [
                {
                    "id"      : tc.id,
                    "type"    : "function",
                    "function": {
                        "name"     : tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

    def add_tool_result(self, tool_call_id: str, content: str):
        self.messages.append({
            "role"        : "tool",
            "tool_call_id": tool_call_id,
            "content"     : content,
        })

    async def get_response(self, tools: Optional[list[dict]] = None):
        """Send current messages to the LLM and return its response."""

        clean_tools = None
        if tools:
            clean_tools = [
                {k: v for k, v in t.items() if k != "_server"}
                for t in tools
            ]

        trimmed = self._trim_history()

        kwargs: dict = {
            "model"   : self.deployment,
            "messages": trimmed,
            # "max_tokens" : 8192,
        }
        if clean_tools:
            kwargs["tools"]       = clean_tools
            kwargs["tool_choice"] = "auto"

        start      = time.perf_counter()
        response   = await self.client.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        # DeepSeek API Issue
        prompt_tokens = 0
        completion_tokens = 0
        if response.usage:
            prompt_tokens = response.usage.prompt_tokens or 0
            completion_tokens = response.usage.completion_tokens or 0

        return ModelResponse(
            message           = response.choices[0].message,
            prompt_tokens     = prompt_tokens,
            completion_tokens = completion_tokens,
            latency_ms        = latency_ms,
        )

    def _trim_history(self) -> list[dict]:
        """
        Sliding window history management.

        System messages are always preserved in full and never dropped.
        When the non-system window exceeds MAX_HISTORY_MESSAGES, the oldest
        messages are dropped and a compact summary is injected in their place.
        Orphaned leading tool results (whose tool call was dropped) are also
        dropped to maintain a valid message sequence.
        """
        system_msgs = [m for m in self.messages if m["role"] == "system"]
        non_system  = [m for m in self.messages if m["role"] != "system"]

        if len(non_system) <= MAX_HISTORY_MESSAGES:
            if self._summary:
                return list(system_msgs) + [self._build_summary_msg()] + non_system
            return list(system_msgs) + non_system

        keep_count = MAX_HISTORY_MESSAGES
        drop       = non_system[:-keep_count]
        keep       = non_system[-keep_count:]

        # Drop any leading tool results whose tool call was already dropped.
        first_keep_idx = 0
        while first_keep_idx < len(keep) and keep[first_keep_idx]["role"] == "tool":
            first_keep_idx += 1

        if first_keep_idx > 0:
            drop = drop + keep[:first_keep_idx]
            keep = keep[first_keep_idx:]

        self._summary = self._summarise_dropped(drop, existing=self._summary)
        self.messages = list(system_msgs) + keep

        return list(system_msgs) + [self._build_summary_msg()] + keep

    def _build_summary_msg(self) -> dict:
        return {
            "role"   : "system",
            "content": (
                "[CONVERSATION SUMMARY — earlier messages compressed to save context]\n"
                f"{self._summary}"
            ),
        }

    @staticmethod
    def _summarise_dropped(messages: list[dict], existing: Optional[str]) -> str:
        """
        Build a compact summary of dropped messages.
        Records tool names, slim arguments, and success/error status.
        Conversation turns (non-tool) are noted by role only.
        """
        lines: list[str] = ["[CONVERSATION HISTORY — earlier turns summarised below]"]

        if existing:
            lines.append(existing)
            lines.append("---")

        for msg in messages:
            role = msg.get("role", "")

            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls") or []:
                    fn   = tc.get("function", {})
                    name = fn.get("name", "?")
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    slim_args = {
                        k: (v[:80] + "…" if isinstance(v, str) and len(v) > 80 else v)
                        for k, v in args.items()
                    }
                    lines.append(f"  CALLED {name}({json.dumps(slim_args)})")

            elif role == "assistant":
                content = msg.get("content") or ""
                preview = content[:120] + "…" if len(content) > 120 else content
                if preview:
                    lines.append(f"  ASSISTANT: {preview}")

            elif role == "tool":
                content = msg.get("content") or ""
                try:
                    data = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    data = {}
                if data.get("error"):
                    lines.append(
                        f"    → ERROR {data.get('error')} "
                        f"(status={data.get('status_code', '?')})"
                    )
                else:
                    lines.append("    → OK")

            elif role == "user":
                content = msg.get("content") or ""
                preview = content[:120] + "…" if len(content) > 120 else content
                if preview:
                    lines.append(f"  USER: {preview}")

        return "\n".join(lines)