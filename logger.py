"""
logger.py
=========
Observability layer for the MCP thesis project.

Structure
---------
  Conversation  — one per application run (start to quit)
    └── Interaction  — one per user message / prompt command
          └── LLMCallLog   — one per model API call
          └── ToolCallLog  — one per MCP tool execution

The JSON and Markdown files are rewritten after every interaction
so the log is always up to date, even if the process crashes.

Logs go to:
  ./logs/conv_YYYYMMDD_HHMMSS.json   — full fidelity, no truncation
  ./logs/conv_YYYYMMDD_HHMMSS.md     — human-readable, thesis-ready

Schema version: 3
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ── Cost estimates (USD per 1M tokens) ───────────────────────
# Update if you switch models.
# _COST_PER_1M = {
#     "input" : 3.00,    # Claude Sonnet 4 input
#     "output": 15.00,   # Claude Sonnet 4 output
# }

_COST_PER_1M = {
    "input": 0.20,    # grok-4-1-fast-reasoning input
    "output": 0.50,   # grok-4-1-fast-reasoning output
}

_SCHEMA_VERSION = 3


# ──────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────────

@dataclass
class LLMCallLog:
    """One model API call. All text fields stored in full — no truncation."""
    timestamp          : str
    model              : str
    turn               : int
    messages_count     : int
    tools_available    : list[str]
    response_type      : str        # "tool_call" | "final_answer"
    tool_calls_made    : list[str]
    content            : str        # full assistant response
    prompt_tokens      : int
    completion_tokens  : int
    total_tokens       : int
    latency_ms         : float
    estimated_cost_usd : float
    error              : Optional[str] = None


@dataclass
class ToolCallLog:
    """One MCP tool execution. Full result stored — no truncation."""
    timestamp          : str
    turn               : int
    server_name        : str
    tool_name          : str
    arguments          : dict
    result             : str        # full result text
    result_size_chars  : int
    success            : bool
    latency_ms         : float
    error              : Optional[str] = None


@dataclass
class InteractionLog:
    """
    One user message → agentic loop → final answer.
    Nested inside a ConversationLog.
    """
    interaction_index       : int       # 1-based position in conversation
    timestamp_start         : str
    timestamp_end           : str
    user_query              : str       # full query
    final_answer            : str       # full answer
    total_turns             : int
    total_tool_calls        : int
    total_latency_ms        : float
    llm_latency_ms          : float
    tool_latency_ms         : float
    total_tokens            : int
    total_prompt_tokens     : int
    total_completion_tokens : int
    estimated_cost_usd      : float
    success                 : bool
    llm_calls               : list[dict] = field(default_factory=list)
    tool_calls              : list[dict] = field(default_factory=list)
    error                   : Optional[str] = None


@dataclass
class ConversationLog:
    """
    One complete application run — everything from startup to quit.
    Contains N interactions, one per user message.
    """
    schema_version          : int
    conversation_id         : str
    timestamp_start         : str
    timestamp_end           : str
    model_used              : str
    servers_used            : list[str]
    total_interactions      : int
    total_turns             : int
    total_tool_calls        : int
    total_latency_ms        : float
    total_tokens            : int
    total_prompt_tokens     : int
    total_completion_tokens : int
    estimated_cost_usd      : float
    interactions            : list[dict] = field(default_factory=list)
    error                   : Optional[str] = None


# ──────────────────────────────────────────────────────────────
# LOGGER
# ──────────────────────────────────────────────────────────────

class MCPLogger:

    def __init__(self, log_dir: str = "./logs", verbose: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.verbose  = verbose
        self.console  = Console()

        # Conversation state
        self._conv_id          : Optional[str]           = None
        self._conv_start_time  : float                   = 0.0
        self._conv_start_ts    : str                     = ""
        self._conv_model       : str                     = ""
        self._interactions     : list[InteractionLog]    = []

        # Current interaction state
        self._ix_index         : int                     = 0
        self._ix_start_time    : float                   = 0.0
        self._ix_start_ts      : str                     = ""
        self._ix_query         : str                     = ""
        self._ix_llm_calls     : list[LLMCallLog]        = []
        self._ix_tool_calls    : list[ToolCallLog]       = []
        self._turn_counter     : int                     = 0

    # ── Conversation lifecycle ──────────────────────────────────

    def start_conversation(self, model: str) -> str:
        """
        Call once when the application starts.
        Returns the conversation_id.
        """
        self._conv_id         = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._conv_start_time = time.perf_counter()
        self._conv_start_ts   = datetime.now().isoformat()
        self._conv_model      = model
        self._interactions    = []
        self._ix_index        = 0

        if self.verbose:
            self.console.print(Panel(
                f"[bold cyan]{self._conv_id}[/bold cyan]\n"
                f"[dim]Model:[/dim] {model}",
                title="🟢 Conversation Started",
                border_style="cyan",
            ))

        return self._conv_id

    def end_conversation(self, error: Optional[str] = None) -> ConversationLog:
        """
        Call when the user quits. Writes the final version of the log files.
        """
        conv = self._build_conversation(error=error)
        self._flush(conv)

        if self.verbose:
            self._print_conversation_summary(conv)

        return conv

    # ── Interaction lifecycle ───────────────────────────────────

    def start_interaction(self, user_query: str, model: str) -> None:
        """
        Call at the start of each user message / prompt command.
        `model` is accepted for compatibility — the conversation model is used.
        """
        self._ix_index      += 1
        self._ix_start_time  = time.perf_counter()
        self._ix_start_ts    = datetime.now().isoformat()
        self._ix_query       = user_query
        self._ix_llm_calls   = []
        self._ix_tool_calls  = []
        self._turn_counter   = 0

        if self.verbose:
            self.console.print(
                f"\n[bold cyan]── Interaction {self._ix_index}[/bold cyan]  "
                f"[dim]{user_query[:120]}[/dim]"
            )

    def end_interaction(
        self,
        final_answer : str,
        success      : bool,
        error        : Optional[str] = None,
    ) -> InteractionLog:
        """
        Call when the agentic loop finishes for one user message.
        Appends the interaction to the conversation and flushes the log files.
        """
        total_ms    = (time.perf_counter() - self._ix_start_time) * 1000
        llm_ms      = sum(c.latency_ms for c in self._ix_llm_calls)
        tool_ms     = sum(t.latency_ms for t in self._ix_tool_calls)
        t_prompt    = sum(c.prompt_tokens     for c in self._ix_llm_calls)
        t_complete  = sum(c.completion_tokens for c in self._ix_llm_calls)
        cost        = sum(c.estimated_cost_usd for c in self._ix_llm_calls)

        interaction = InteractionLog(
            interaction_index       = self._ix_index,
            timestamp_start         = self._ix_start_ts,
            timestamp_end           = datetime.now().isoformat(),
            user_query              = self._ix_query,
            final_answer            = final_answer,
            total_turns             = self._turn_counter,
            total_tool_calls        = len(self._ix_tool_calls),
            total_latency_ms        = total_ms,
            llm_latency_ms          = llm_ms,
            tool_latency_ms         = tool_ms,
            total_tokens            = t_prompt + t_complete,
            total_prompt_tokens     = t_prompt,
            total_completion_tokens = t_complete,
            estimated_cost_usd      = cost,
            success                 = success,
            llm_calls               = [asdict(c) for c in self._ix_llm_calls],
            tool_calls              = [asdict(c) for c in self._ix_tool_calls],
            error                   = error,
        )

        self._interactions.append(interaction)

        # Flush after every interaction — crash-safe
        conv = self._build_conversation()
        self._flush(conv)

        if self.verbose:
            self._print_interaction_summary(interaction)

        return interaction

    # ── Backward-compat aliases (used by app.py) ───────────────

    def start_session(self, user_query: str, model: str) -> str:
        """Alias for start_interaction — keeps app.py unchanged."""
        self.start_interaction(user_query=user_query, model=model)
        return self._conv_id or ""

    def end_session(
        self,
        final_answer : str,
        success      : bool,
        error        : Optional[str] = None,
    ) -> InteractionLog:
        """Alias for end_interaction — keeps app.py unchanged."""
        return self.end_interaction(final_answer=final_answer, success=success, error=error)

    # ── Turn counter ────────────────────────────────────────────

    def increment_turn(self):
        self._turn_counter += 1

    # ── Per-call logging ────────────────────────────────────────

    def log_llm_call(
        self,
        model             : str,
        messages_count    : int,
        tools_available   : list[str],
        response_type     : str,
        tool_calls_made   : list[str],
        content_preview   : str,        # param name kept for app.py compat
        prompt_tokens     : int,
        completion_tokens : int,
        latency_ms        : float,
        error             : Optional[str] = None,
    ) -> LLMCallLog:
        cost = (
            (prompt_tokens     / 1_000_000) * _COST_PER_1M["input"] +
            (completion_tokens / 1_000_000) * _COST_PER_1M["output"]
        )
        entry = LLMCallLog(
            timestamp          = datetime.now().isoformat(),
            model              = model,
            turn               = self._turn_counter,
            messages_count     = messages_count,
            tools_available    = tools_available,
            response_type      = response_type,
            tool_calls_made    = tool_calls_made,
            content            = content_preview,   # full — no cap
            prompt_tokens      = prompt_tokens,
            completion_tokens  = completion_tokens,
            total_tokens       = prompt_tokens + completion_tokens,
            latency_ms         = latency_ms,
            estimated_cost_usd = cost,
            error              = error,
        )
        self._ix_llm_calls.append(entry)

        if self.verbose:
            is_final = response_type == "final_answer"
            color = "green" if is_final else "yellow"
            label = "✅ FINAL ANSWER" if is_final else "🔧 TOOL CALL"
            if error:
                color, label = "red", "❌ ERROR"

            self.console.print(f"\n[bold {color}]── Turn {self._turn_counter} · {label}[/bold {color}]")
            t = Table(show_header=False, box=None, padding=(0, 1))
            t.add_row("[dim]model[/dim]",    f"[cyan]{model}[/cyan]")
            t.add_row("[dim]messages[/dim]", str(messages_count))
            t.add_row("[dim]tokens[/dim]",   f"[yellow]{prompt_tokens}↑ {completion_tokens}↓[/yellow]")
            t.add_row("[dim]latency[/dim]",  f"[magenta]{latency_ms:.0f}ms[/magenta]")
            t.add_row("[dim]cost[/dim]",     f"[dim]${cost:.4f}[/dim]")
            if tool_calls_made:
                t.add_row("[dim]tools called[/dim]", f"[bold green]{', '.join(tool_calls_made)}[/bold green]")
            if content_preview:
                t.add_row("[dim]response[/dim]", f"[dim white]{content_preview[:150]}[/dim white]")
            if error:
                t.add_row("[dim]error[/dim]", f"[red]{error}[/red]")
            self.console.print(t)

        return entry

    def log_tool_call(
        self,
        server_name : str,
        tool_name   : str,
        arguments   : dict,
        result      : str,
        success     : bool,
        latency_ms  : float,
        error       : Optional[str] = None,
    ) -> ToolCallLog:
        entry = ToolCallLog(
            timestamp         = datetime.now().isoformat(),
            turn              = self._turn_counter,
            server_name       = server_name,
            tool_name         = tool_name,
            arguments         = arguments,
            result            = result,         # full — no cap
            result_size_chars = len(result),
            success           = success,
            latency_ms        = latency_ms,
            error             = error,
        )
        self._ix_tool_calls.append(entry)

        if self.verbose:
            color = "green" if success else "red"
            self.console.print(
                f"\n  [bold {color}]{'✅' if success else '❌'}  {tool_name}[/bold {color}]"
            )
            t = Table(show_header=False, box=None, padding=(0, 1))
            t.add_row("   [dim]args[/dim]",    f"[white]{json.dumps(arguments)[:120]}[/white]")
            t.add_row("   [dim]latency[/dim]", f"[magenta]{latency_ms:.0f}ms[/magenta]")
            t.add_row("   [dim]size[/dim]",    f"[yellow]{entry.result_size_chars} chars[/yellow]")
            if error:
                t.add_row("   [dim]error[/dim]", f"[red]{error}[/red]")
            else:
                t.add_row("   [dim]preview[/dim]", f"[dim white]{result[:180]}[/dim white]")
            self.console.print(t)

        return entry

    # ── Internal: build + flush ─────────────────────────────────

    def _build_conversation(self, error: Optional[str] = None) -> ConversationLog:
        all_interactions = self._interactions
        servers = list({
            tc["server_name"]
            for ix in all_interactions
            for tc in ix.tool_calls
        })
        return ConversationLog(
            schema_version          = _SCHEMA_VERSION,
            conversation_id         = self._conv_id or "unknown",
            timestamp_start         = self._conv_start_ts,
            timestamp_end           = datetime.now().isoformat(),
            model_used              = self._conv_model,
            servers_used            = servers,
            total_interactions      = len(all_interactions),
            total_turns             = sum(ix.total_turns      for ix in all_interactions),
            total_tool_calls        = sum(ix.total_tool_calls for ix in all_interactions),
            total_latency_ms        = sum(ix.total_latency_ms for ix in all_interactions),
            total_tokens            = sum(ix.total_tokens     for ix in all_interactions),
            total_prompt_tokens     = sum(ix.total_prompt_tokens     for ix in all_interactions),
            total_completion_tokens = sum(ix.total_completion_tokens for ix in all_interactions),
            estimated_cost_usd      = sum(ix.estimated_cost_usd      for ix in all_interactions),
            interactions            = [asdict(ix) for ix in all_interactions],
            error                   = error,
        )

    def _flush(self, conv: ConversationLog) -> None:
        """Write JSON and Markdown files. Called after every interaction."""
        base = self.log_dir / conv.conversation_id

        with open(f"{base}.json", "w", encoding="utf-8") as f:
            json.dump(asdict(conv), f, indent=2, ensure_ascii=False)

        with open(f"{base}.md", "w", encoding="utf-8") as f:
            f.write(self._render_markdown(conv))

    # ── Markdown export ─────────────────────────────────────────

    def _render_markdown(self, conv: ConversationLog) -> str:
        lines: list[str] = []

        lines.append(f"# Conversation: {conv.conversation_id}\n")
        lines.append(f"**Model:** {conv.model_used}  ")
        lines.append(f"**Started:** {conv.timestamp_start}  ")
        lines.append(f"**Ended:** {conv.timestamp_end}\n")

        lines.append("## Conversation Summary\n")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Interactions | {conv.total_interactions} |")
        lines.append(f"| Total turns | {conv.total_turns} |")
        lines.append(f"| Total tool calls | {conv.total_tool_calls} |")
        lines.append(f"| Servers | {', '.join(conv.servers_used) or 'none'} |")
        lines.append(f"| Total tokens | {conv.total_tokens:,} |")
        lines.append(f"| Input tokens | {conv.total_prompt_tokens:,} |")
        lines.append(f"| Output tokens | {conv.total_completion_tokens:,} |")
        lines.append(f"| Estimated cost | ${conv.estimated_cost_usd:.4f} |")
        lines.append(f"| Total latency | {conv.total_latency_ms/1000:.1f}s |\n")

        for ix in conv.interactions:
            lines.append(f"---\n")
            lines.append(
                f"## Interaction {ix['interaction_index']}: "
                f"{'✅' if ix['success'] else '❌'}\n"
            )
            lines.append(f"**User:** {ix['user_query']}\n")
            lines.append("| Metric | Value |")
            lines.append("|---|---|")
            lines.append(f"| Turns | {ix['total_turns']} |")
            lines.append(f"| Tool calls | {ix['total_tool_calls']} |")
            lines.append(f"| Tokens | {ix['total_tokens']:,} ({ix['total_prompt_tokens']:,}↑ {ix['total_completion_tokens']:,}↓) |")
            lines.append(f"| Cost | ${ix['estimated_cost_usd']:.4f} |")
            lines.append(f"| Latency | {ix['total_latency_ms']/1000:.1f}s (llm {ix['llm_latency_ms']/1000:.1f}s / tools {ix['tool_latency_ms']/1000:.1f}s) |\n")

            # Turn-by-turn log with tool calls interleaved
            tools_by_turn: dict[int, list[dict]] = {}
            for tc in ix["tool_calls"]:
                tools_by_turn.setdefault(tc["turn"], []).append(tc)

            for call in ix["llm_calls"]:
                turn = call["turn"]
                lines.append(f"### Turn {turn} — {call['response_type'].replace('_', ' ').title()}\n")
                lines.append("| | |")
                lines.append("|---|---|")
                lines.append(f"| Tokens | {call['prompt_tokens']:,}↑ {call['completion_tokens']:,}↓ |")
                lines.append(f"| Cost | ${call['estimated_cost_usd']:.4f} |")
                lines.append(f"| Latency | {call['latency_ms']:.0f}ms |\n")

                if call.get("tool_calls_made"):
                    lines.append(f"**Tools called:** {', '.join(call['tool_calls_made'])}\n")

                if call.get("content"):
                    lines.append("**Assistant:**\n")
                    lines.append(call["content"])
                    lines.append("")

                for tc in tools_by_turn.get(turn, []):
                    status = "✅" if tc["success"] else "❌"
                    lines.append(f"#### {status} `{tc['tool_name']}`\n")
                    lines.append("**Arguments:**")
                    lines.append(f"```json\n{json.dumps(tc['arguments'], indent=2, ensure_ascii=False)}\n```")
                    lines.append(f"**Result** ({tc['result_size_chars']} chars):")
                    lines.append(f"```json\n{tc['result']}\n```\n")

            lines.append(f"**Final Answer:**\n\n{ix['final_answer']}\n")

            if ix.get("error"):
                lines.append(f"**Error:** `{ix['error']}`\n")

        return "\n".join(lines)

    # ── Console summaries ───────────────────────────────────────

    def _print_interaction_summary(self, ix: InteractionLog):
        color = "green" if ix.success else "red"
        body = (
            f"[bold {color}]{'✅ SUCCESS' if ix.success else '❌ FAILED'}[/bold {color}]\n\n"
            f"[dim]turns[/dim]       {ix.total_turns}\n"
            f"[dim]tool calls[/dim]  {ix.total_tool_calls}\n"
            f"[dim]tokens[/dim]      [yellow]{ix.total_tokens:,}[/yellow]  "
            f"[dim]({ix.total_prompt_tokens:,}↑ {ix.total_completion_tokens:,}↓)[/dim]\n"
            f"[dim]cost[/dim]        [yellow]${ix.estimated_cost_usd:.4f}[/yellow]\n"
            f"[dim]latency[/dim]     [magenta]{ix.total_latency_ms:.0f}ms[/magenta]  "
            f"[dim](llm {ix.llm_latency_ms:.0f}ms / tools {ix.tool_latency_ms:.0f}ms)[/dim]\n\n"
            f"{ix.final_answer[:400]}"
        )
        self.console.print(Panel(
            body,
            title=f"📊 Interaction {ix.interaction_index} Complete",
            border_style=color,
        ))
        self.console.print(
            f"[dim]→ {self.log_dir}/{self._conv_id}.json[/dim]\n"
            f"[dim]→ {self.log_dir}/{self._conv_id}.md[/dim]\n"
        )

    def _print_conversation_summary(self, conv: ConversationLog):
        body = (
            f"[bold cyan]Conversation ended[/bold cyan]\n\n"
            f"[dim]interactions[/dim]  {conv.total_interactions}\n"
            f"[dim]total turns[/dim]   {conv.total_turns}\n"
            f"[dim]tool calls[/dim]    {conv.total_tool_calls}\n"
            f"[dim]tokens[/dim]        [yellow]{conv.total_tokens:,}[/yellow]  "
            f"[dim]({conv.total_prompt_tokens:,}↑ {conv.total_completion_tokens:,}↓)[/dim]\n"
            f"[dim]total cost[/dim]    [yellow]${conv.estimated_cost_usd:.4f}[/yellow]\n"
            f"[dim]total latency[/dim] [magenta]{conv.total_latency_ms/1000:.1f}s[/magenta]"
        )
        self.console.print(Panel(body, title="🏁 Conversation Complete", border_style="cyan"))
        self.console.print(
            f"[dim]→ {self.log_dir}/{conv.conversation_id}.json[/dim]\n"
            f"[dim]→ {self.log_dir}/{conv.conversation_id}.md[/dim]\n"
        )

    # ── Analysis helpers ────────────────────────────────────────

    def load_all_conversations(self) -> list[dict]:
        """Load all saved conversation JSON files."""
        convs = []
        for f in sorted(self.log_dir.glob("conv_*.json")):
            with open(f, encoding="utf-8") as fp:
                convs.append(json.load(fp))
        return convs

    def summarise_conversations(self) -> dict:
        """
        Aggregate stats across all conversation logs.
        Useful for thesis evaluation tables.
        """
        convs = self.load_all_conversations()
        if not convs:
            return {}

        n = len(convs)

        def avg(key):
            vals = [c.get(key, 0) for c in convs]
            return sum(vals) / len(vals) if vals else 0

        per_model: dict[str, dict] = {}
        for c in convs:
            m = c.get("model_used", "unknown")
            if m not in per_model:
                per_model[m] = {"conversations": 0, "interactions": 0, "tokens": 0, "cost": 0.0}
            per_model[m]["conversations"] += 1
            per_model[m]["interactions"]  += c.get("total_interactions", 0)
            per_model[m]["tokens"]        += c.get("total_tokens", 0)
            per_model[m]["cost"]          += c.get("estimated_cost_usd", 0.0)

        return {
            "conversation_count"     : n,
            "avg_interactions"       : avg("total_interactions"),
            "avg_turns"              : avg("total_turns"),
            "avg_tool_calls"         : avg("total_tool_calls"),
            "avg_tokens"             : avg("total_tokens"),
            "avg_cost_usd"           : avg("estimated_cost_usd"),
            "avg_latency_ms"         : avg("total_latency_ms"),
            "total_cost_usd"         : sum(c.get("estimated_cost_usd", 0) for c in convs),
            "per_model"              : per_model,
        }

    def export_thesis_table(self) -> str:
        """
        Markdown table of all conversations — one row per run.
        Suitable for a thesis appendix.
        """
        convs = self.load_all_conversations()
        if not convs:
            return "_No conversations found._"

        header = (
            "| Conversation | Model | Interactions | Turns | Tools | Tokens | Cost | Latency |\n"
            "|---|---|---|---|---|---|---|---|"
        )
        rows = []
        for c in convs:
            rows.append(
                f"| {c['conversation_id']} "
                f"| {c.get('model_used','?')} "
                f"| {c.get('total_interactions','?')} "
                f"| {c.get('total_turns','?')} "
                f"| {c.get('total_tool_calls','?')} "
                f"| {c.get('total_tokens',0):,} "
                f"| ${c.get('estimated_cost_usd',0):.4f} "
                f"| {c.get('total_latency_ms',0)/1000:.1f}s |"
            )

        return header + "\n" + "\n".join(rows)