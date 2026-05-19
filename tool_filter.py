"""
tool_filter.py
==============
Lightweight keyword-based tool filtering.
Reduces tool schema tokens sent to the model on each turn
by only including tools relevant to the current query.
This is intentionally simple — a keyword heuristic, not a semantic classifier.
For a nano-class model, reducing schema tokens also reduces the chance of
the model getting confused by irrelevant tool signatures.
"""

# Maps each tool to keywords that suggest it's relevant.
# If a query matches none of these, the tool is excluded from that turn.
#
# IMPORTANT: avoid generic stop-words like "what", "how", "do", "is" —
# they match almost every query and will incorrectly include unrelated tools.
# Signals should be domain-specific, not grammatical.

_TOOL_SIGNALS: dict[str, list[str]] = {
    "get_employee_info": [
        # identity / directory questions
        "who", "whose", "employee", "person", "people", "staff",
        # contact / org fields
        "email", "contact", "location", "office", "city",
        # role / team fields
        "role", "team", "department", "manager", "reports",
        # job title words that appear in the directory
        "engineer", "designer", "director", "scientist", "manager",
        "developer", "analyst", "recruiter", "psychologist", "sales",
        # casual person-question phrasing (verb forms)
        "does", "works", "doing", "working",
        # lookup intent
        "look up", "find out about", "tell me about",
    ],
    "get_recent_announcements": [
        "news", "announcement", "announced", "recent", "latest",
        "happened", "leadership", "posted", "published", "memo",
    ],
    "search_intranet": [
        # policy / procedure topics (domain-specific, not generic question words)
        "policy", "policies", "procedure", "guide", "guideline", "rules",
        "onboarding", "travel", "expense", "expenses", "budget",
        "remote", "hybrid", "equipment", "okr", "goal", "objective",
        "agreement", "process", "handbook", "documentation",
        # explicit search intent
        "find", "search", "intranet", "page", "document", "article",
    ],
    "get_page_full_content": [
        # almost always triggered by model after search, not directly by user
        "policy", "guide", "full", "content", "details", "more",
    ],
    "update_employee_info": [
        "update", "change", "set", "move", "relocate", "edit", "modify",
        "location", "email", "role", "team", "transfer",
    ],
    "create_announcement": [
        "announce", "announcement", "post", "publish", "create",
        "notify", "tell everyone", "company-wide", "send out",
    ],
}

# Tools that are always included regardless of query content.
_ALWAYS_INCLUDE: set[str] = set()


def filter_tools(tools: list[dict], query: str) -> list[dict]:
    """
    Return only the tools relevant to this query.
    Falls back to all tools if nothing matches (safe default).

    tools: OpenAI-format tool list from MCPClient
    query: the raw user input (before @mention stripping)
    """
    query_lower = query.lower()
    relevant = []

    for tool in tools:
        name = tool["function"]["name"]
        if name in _ALWAYS_INCLUDE:
            relevant.append(tool)
            continue

        signals = _TOOL_SIGNALS.get(name, [])
        if any(signal in query_lower for signal in signals):
            relevant.append(tool)

    # Safe fallback: if nothing matched, send everything
    return relevant if relevant else tools