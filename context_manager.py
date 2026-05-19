"""
context_manager.py
==================
Context window optimisation for the MCP agentic loop.

PROBLEM:
    Every tool result gets appended verbatim to the message history.
    As the loop progresses, the context fills with raw JSON that the
    model has already "acted on" — wasting tokens and increasing latency.

STRATEGIES IMPLEMENTED:

    1. Tool result compression
       Raw JSON tool results are stripped of redundant keys and
       stored as compact strings rather than pretty-printed JSON.

    2. Search-then-fetch pruning
       When the model calls get_page_full_content(page_id) after a
       search_intranet that returned that page_id, the search result
       message is replaced with a one-line stub:
           "[search result replaced — full content fetched for page_id]"
       The full content fetch result is kept as-is since that's the
       payload the model will actually reason over.

    3. Token threshold summarisation
       When total estimated tokens in history exceeds a configurable
       threshold, the oldest tool exchange pairs (assistant tool_call +
       tool result) are collapsed into a single summarisation message,
       preserving the information in compressed form.

THESIS NOTE:
    This module is itself a research contribution. You can measure:
    - Token reduction per session (before vs after)
    - Whether pruning affects answer quality
    - Which strategy has the highest impact
"""

import json
import re
from typing import Optional


# ──────────────────────────────────────────────────────────────
# ROUGH TOKEN ESTIMATOR
# 1 token ≈ 4 chars for English text — good enough for decisions
# ──────────────────────────────────────────────────────────────

def estimate_tokens(messages: list[dict]) -> int:
    total_chars = sum(
        len(json.dumps(m)) for m in messages
    )
    return total_chars // 4


# ──────────────────────────────────────────────────────────────
# TOOL RESULT COMPRESSOR
# Removes redundant metadata from known tool output shapes
# ──────────────────────────────────────────────────────────────

def compress_tool_result(tool_name: str, result_text: str) -> str:
    """
    Compress a tool result string before storing it in the context.

    For structured JSON results, we strip keys that are purely
    operational (query echo, available IDs lists, suggestion fields)
    and keep only what the model needs to reason over.

    Returns the compressed string, or the original if parsing fails.
    """
    try:
        data = json.loads(result_text)
    except (json.JSONDecodeError, TypeError):
        return result_text  # not JSON, return as-is

    if tool_name == "search_intranet":
        # Remove: query echo, category_filter (model already knows these)
        # Keep: results (with content_excerpt), total
        data.pop("query", None)
        data.pop("category_filter", None)
        # Trim each result — remove relevance_tags (verbose, low value)
        for r in data.get("results", []):
            r.pop("relevance_tags", None)

    elif tool_name == "get_page_full_content":
        # Remove: available_page_ids (only needed on error)
        # Keep: found, page (with all content)
        if data.get("found"):
            data.pop("available_page_ids", None)
            # Also trim tags from the page itself
            if "page" in data:
                data["page"].pop("tags", None)

    elif tool_name == "get_employee_profile":
        # Remove: available_names (only needed on error)
        if data.get("found"):
            data.pop("available_names", None)
            data.pop("suggestion", None)

    elif tool_name == "get_recent_announcements":
        # Remove: requested_limit, days_back echoes
        data.pop("requested_limit", None)
        data.pop("days_back", None)

    elif tool_name == "get_employee_info":
        if data.get("found"):
            data.pop("suggestion", None)

    return json.dumps(data, separators=(",", ":"))  # compact, no whitespace


# ──────────────────────────────────────────────────────────────
# MAIN CONTEXT MANAGER CLASS
# ──────────────────────────────────────────────────────────────

class ContextManager:
    """
    Wraps model.messages and applies pruning strategies.
    """

    def __init__(self, model, token_threshold: int = 4000):
        self.model = model
        self.token_threshold = token_threshold

        # Tracks search results per tool_call_id so we can prune them
        # Format: tool_call_id → {"tool_name": str, "page_ids_returned": list[str]}
        self._tool_call_meta: dict[str, dict] = {}

    # ── Public interface ────────────────────────────────────────

    def record_tool_call(self, tool_call_id: str, tool_name: str, arguments: dict):
        """
        Call this when the model requests a tool call, before execution.
        Records metadata needed for later pruning decisions.
        """
        meta: dict = {"tool_name": tool_name, "arguments": arguments}

        # For fetch tools, record which page_id is being fetched
        # so we can find and prune the search result that produced it
        if tool_name == "get_page_full_content":
            meta["fetching_page_id"] = arguments.get("page_id", "")

        self._tool_call_meta[tool_call_id] = meta

    def add_tool_result(self, tool_call_id: str, tool_name: str, result: str):
        """
        Compress and store a tool result in model.messages.
        Use this instead of model.add_tool_result() directly.
        """
        compressed = compress_tool_result(tool_name, result)

        # Also track page_ids returned by search results for pruning
        if tool_name == "search_intranet":
            try:
                data = json.loads(result)
                page_ids = [r["id"] for r in data.get("results", [])]
                if tool_call_id in self._tool_call_meta:
                    self._tool_call_meta[tool_call_id]["page_ids_returned"] = page_ids
            except (json.JSONDecodeError, KeyError):
                pass

        self.model.add_tool_result(tool_call_id, compressed)

    def prune_after_turn(self):
        """
        Call after all tool results for a turn have been added.

        Applies search-then-fetch pruning:
        If a get_page_full_content was just called with a page_id that
        appeared in a previous search_intranet result, replace that
        search result message with a compact stub.
        """
        # Find any fetch calls made this turn
        fetch_calls = {
            tc_id: meta
            for tc_id, meta in self._tool_call_meta.items()
            if meta.get("tool_name") == "get_page_full_content"
            and "fetching_page_id" in meta
        }

        if not fetch_calls:
            return

        fetched_page_ids = {
            meta["fetching_page_id"]
            for meta in fetch_calls.values()
        }

        # Find search result messages that returned any of these page IDs
        for i, msg in enumerate(self.model.messages):
            if msg.get("role") != "tool":
                continue

            content = msg.get("content", "")
            try:
                data = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                continue

            # Is this a search result message?
            if "results" not in data:
                continue

            returned_ids = {r.get("id") for r in data.get("results", [])}

            # If any fetched page came from this search result, prune it
            if returned_ids & fetched_page_ids:
                matched = returned_ids & fetched_page_ids
                self.model.messages[i] = {
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": json.dumps({
                        "pruned": True,
                        "reason": "full content fetched for these page IDs",
                        "page_ids": list(matched),
                    }),
                }

    def maybe_summarise(self):
        """
        If estimated token count exceeds the threshold, collapse the
        oldest tool exchange pairs into a compact summary message.

        A "tool exchange pair" is:
            - The assistant message containing tool_calls
            - The following tool result message(s)

        These are replaced with a single system-style message summarising
        what was looked up and what was found.
        """
        if estimate_tokens(self.model.messages) < self.token_threshold:
            return

        # Find the oldest collapsible exchange (after the system message)
        # We never collapse: system message, the most recent user message,
        # or the last two turns (model still needs recent context)
        messages = self.model.messages
        collapse_up_to = max(1, len(messages) - 6)  # keep last 6 messages intact

        i = 1  # skip system message
        while i < collapse_up_to:
            msg = messages[i]

            # Find an assistant message with tool_calls
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                i += 1
                continue

            # Collect the tool result messages that follow
            tool_results = []
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                tool_results.append(messages[j])
                j += 1

            if not tool_results:
                i += 1
                continue

            # Build a compact summary of this exchange
            tool_names_called = [tc["function"]["name"] for tc in msg.get("tool_calls", [])]
            summaries = []
            for result_msg in tool_results:
                content = result_msg.get("content", "")
                try:
                    data = json.loads(content)
                    # Extract just the most useful bit per tool type
                    if "results" in data:
                        titles = [r.get("title", "") for r in data["results"]]
                        summaries.append(f"search returned: {', '.join(titles)}")
                    elif "page" in data:
                        title = data["page"].get("title", "unknown")
                        summaries.append(f"fetched full content of: {title}")
                    elif "employee" in data:
                        name = data["employee"].get("name", "unknown")
                        summaries.append(f"found employee: {name}")
                    elif "announcements" in data:
                        count = len(data["announcements"])
                        summaries.append(f"retrieved {count} announcements")
                    elif data.get("pruned"):
                        summaries.append(f"[already pruned: {data.get('page_ids')}]")
                    else:
                        summaries.append(content[:80])
                except (json.JSONDecodeError, TypeError):
                    summaries.append(content[:80])

            summary_message = {
                "role": "user",   # OpenAI accepts system-style info in user role
                "content": (
                    f"[Context summary — turn already completed]\n"
                    f"Tools called: {', '.join(tool_names_called)}\n"
                    f"Results: {'; '.join(summaries)}"
                ),
            }

            # Replace the exchange with the summary
            self.model.messages = (
                messages[:i] +
                [summary_message] +
                messages[j:]
            )

            # Only collapse one exchange per call to avoid over-pruning
            break