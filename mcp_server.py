import sys
import json
from mcp.server.fastmcp import FastMCP 
from typing import Literal, Optional
from mcp.server.fastmcp.prompts import base
from pydantic import Field

FAKE_PAGES = [
    {
        "id": "page_001",
        "title": "Engineering Onboarding Guide",
        "content": "Welcome to the Engineering team! This guide covers your first two weeks. Week 1: Set up your development environment using the DevSetup script in the Platform repo. Install Docker, configure VPN, and request access to Azure DevOps. Your buddy is assigned by your manager. Week 2: Shadow your team, attend the weekly architecture review every Tuesday at 10:00 CET, and complete the security training in the Learning Portal.",
        "category": "onboarding",
        "last_updated": "2025-11-15",
        "author": "Frank Müller",
        "tags": ["onboarding", "engineering", "setup"],
    },
    {
        "id": "page_002",
        "title": "International Travel Expense Policy",
        "content": "Employees traveling internationally for business may claim: flights (economy class for trips under 6 hours, business class for longer), hotels up to €200/night in Western Europe and €150/night elsewhere, meals up to €75/day. Submit all claims within 30 days of return via the HR portal. Receipts required for any individual expense over €25. Pre-approval required for trips exceeding €2,000 total.",
        "category": "policy",
        "last_updated": "2025-09-01",
        "author": "HR Team",
        "tags": ["travel", "expense", "policy", "international"],
    },
    {
        "id": "page_003",
        "title": "Q4 2025 Company OKRs",
        "content": "Objective 1: Launch MCP-powered AI assistant (KR: 3 enterprise workflows automated, KR: <2s average response time). Objective 2: Expand to 3 new enterprise customers (KR: 2 POCs signed by November, KR: €500k pipeline generated). Objective 3: Improve platform reliability (KR: 99.9% uptime, KR: MTTR under 15 minutes).",
        "category": "strategy",
        "last_updated": "2025-10-01",
        "author": "Leadership Team",
        "tags": ["okr", "strategy", "q4", "goals"],
    },
    {
        "id": "page_004",
        "title": "Platform Team — Working Agreements",
        "content": "Our team agreements: Code reviews within 24 hours. PRs should be small (<400 lines). All PRs need at least 2 approvals. Deploy to staging every Wednesday, production every other Friday. Incidents: use the #platform-incidents Slack channel, post an RCA within 48 hours. Meetings: standup daily at 9:30 CET (15 min max). Architecture decisions documented in ADR format in the /docs folder.",
        "category": "team",
        "last_updated": "2025-10-20",
        "author": "Eva Nowak",
        "tags": ["platform", "team", "agreements", "code review", "pull request"],
    },
    {
        "id": "page_005",
        "title": "Remote Work Policy 2025",
        "content": "Create IT supports a hybrid-first work model. Employees may work remotely up to 3 days per week. Home office equipment budget: €500 one-time, €100/year recurring. Team anchor days (mandatory in-office): Tuesday and Thursday. International remote work allowed for up to 30 days/year with manager approval. Coworking space budget: €150/month if you lack a suitable home office.",
        "category": "policy",
        "last_updated": "2025-01-10",
        "author": "HR Team",
        "tags": ["remote", "hybrid", "policy", "work"],
    },
]

FAKE_EMPLOYEES = [
    {"id": "emp_001", "name": "Alice Johansson", "role": "Senior Engineer", "team": "Platform", "email": "alice@createit.com", "location": "Stockholm"},
    {"id": "emp_002", "name": "Bob Martens", "role": "Product Manager", "team": "Product", "email": "bob@createit.com", "location": "Amsterdam"},
    {"id": "emp_003", "name": "Clara Ferreira", "role": "UX Designer", "team": "Design", "email": "clara@createit.com", "location": "Lisbon"},
    {"id": "emp_004", "name": "David Kim", "role": "Data Scientist", "team": "Analytics", "email": "david@createit.com", "location": "Seoul"},
    {"id": "emp_005", "name": "Eva Nowak", "role": "DevOps Engineer", "team": "Platform", "email": "eva@createit.com", "location": "Warsaw"},
    {"id": "emp_006", "name": "Frank Müller", "role": "Engineering Manager", "team": "Platform", "email": "frank@createit.com", "location": "Berlin"},
    {"id": "emp_007", "name": "Grace Chen", "role": "Backend Engineer", "team": "Platform", "email": "grace@createit.com", "location": "Singapore"},
    {"id": "emp_008", "name": "Hans Weber", "role": "Sales Director", "team": "Sales", "email": "hans@createit.com", "location": "Frankfurt"},
    {"id": "emp_009", "name": "Maria Albino", "role": "Psychologist", "team": "Human Resources", "email": "maria@createit.com", "location": "Castelo Branco"}
]

FAKE_ANNOUNCEMENTS = [
    {"id": "ann_001", "title": "New AI Tools Available", "summary": "We've rolled out GitHub Copilot to all engineering staff. Licenses are now in your GitHub settings.", "date": "2025-12-01", "author": "Frank Müller"},
    {"id": "ann_002", "title": "Office Closed December 24–26", "summary": "The Amsterdam and Stockholm offices will be closed for the Christmas holidays.", "date": "2025-12-10", "author": "Office Manager"},
    {"id": "ann_003", "title": "Q3 Results: Strong Growth", "summary": "We hit 127% of our Q3 revenue target. Full results presented at the All-Hands on December 15.", "date": "2025-11-30", "author": "CEO"},
    {"id": "ann_004", "title": "Platform Team Hiring", "summary": "We are opening 2 Senior Engineer positions on the Platform team. Referrals welcome — €2,000 referral bonus.", "date": "2025-11-25", "author": "HR Team"},
]

mcp = FastMCP("mock-mcp")

# ══════════════════════════════════════════════════════════════
# TOOLS
# ══════════════════════════════════════════════════════════════

@mcp.tool(
    name="get_employee_info",
    description="""Retrieve employee information from the company directory. Use when the user asks about a specific person or job role.
                Identifier may be name, email, employee ID, or title.
                """
)
def get_employee_info(
    identifier: str = Field(
        description= "Name, first name, email, emp ID, or role. e.g. 'Alice', 'emp_001', 'Data Scientist'."
        )
    ) -> dict:

    identifier_lower = identifier.lower().strip()
    for emp in FAKE_EMPLOYEES:
        if (
            emp["id"].lower()    == identifier_lower or
            emp["name"].lower()  == identifier_lower or
            emp["email"].lower() == identifier_lower or
            emp["role"].lower() == identifier_lower or
            emp["name"].lower().startswith(identifier_lower)  # first name match
        ):
            return {"found": True, "employee": emp}
        

    identifier_words = set(identifier_lower.split())
    close_matches = [
        f"{emp['name']} ({emp['role']})"
        for emp in FAKE_EMPLOYEES
        if identifier_words & (
            set(emp["name"].lower().split()) |
            set(emp["role"].lower().split())  # ← also match against role words
       )
    ]
        
    error_response = {
        "found": False,
        "error": f"No employee found matching '{identifier}'.",
        "hint": ( "Retry with corrected spelling, or use search_intranet with role/team context."
        ),
    }

    # Surface candidates only when there is a plausible partial match
    if close_matches:
        error_response["possible_matches"] = close_matches

    return error_response


@mcp.tool(
    name="get_recent_announcements",
    description="Retrieve recent company-wide announcements. Use only for news, events, or leadership communications explicitly posted as announcements."
)
def get_recent_announcements(
    limit: int = Field(
        default=5,
        description="Number of announcements to return (1-20)",
        ge=1,
        le=20
    ),
    days_back: int = Field(
        default=30,
        description="Look-back window in days",
        ge=1
    )
    ) -> dict:
    
    limit = min(limit, 20)
    results = FAKE_ANNOUNCEMENTS[:limit]

    return {
       "announcements": results,
       "total": len(results)
   }

@mcp.tool(
    name="search_intranet",
    description= "Search the company intranet and return the most relevant page for a query. Use for policies, procedures, documentation, or internal knowledge."
)
def search_intranet(
    query: str = Field(description="Keywords for the topic. e.g. 'remote work', 'travel expenses', 'onboarding'."
               ),
    category: Optional[
        Literal["policy", "team", "onboarding", "strategy"]] = Field(
        default=None,
        description="Only set if unambiguous. Leave empty if unsure — omitting gives better results. e.g. 'policy', 'team', 'onboarding', 'strategy' "
    ),
) -> dict:
    query_lower = query.lower()
    query_words = query_lower.split()

    scored = []

    for page in FAKE_PAGES:
        if category and page["category"] != category:
            continue

        # Score by relevance: title matches weighted highest, then tags, then content
        score = 0
        for word in query_words:
            if word in page["title"].lower():
                score += 3
            if word in " ".join(page["tags"]):
                score += 2
            if word in page["content"].lower():
                score += 1

        if score > 0:
            scored.append((score, {
                "id": page["id"],
                "title": page["title"],
                "content": page["content"],
                "category": page["category"],
                "last_updated": page["last_updated"],
                "author": page["author"],
            }))

    scored.sort(key=lambda x: x[0], reverse=True)

    return {
        "results": [item for _, item in scored[:1]],
        "total_matches": len(scored),
    }

@mcp.tool(
    name="get_page_full_content",
    description="Retrieve the full text of an intranet page by page ID. Only call if the excerpt was cut off mid-sentence or is clearly missing information needed to answer."
)
def get_page_full_content(
    page_id: str = Field(description="Page ID from search_intranet, e.g. 'page_001'.")
) -> dict:
    for page in FAKE_PAGES:
        if page["id"] == page_id:
            return {
                "found": True,
                "id": page["id"],
                "title": page["title"],
                "content": page["content"],
                "category": page["category"],
                "last_updated": page["last_updated"],
                "author": page["author"],
                "tags": page["tags"],
            }
    return {
        "found": False,
        "error": f"No page found with ID '{page_id}'"
    }

@mcp.tool(
    name="update_employee_info",
    description="Update an employee directory field (location, role, team, or email).Use when the user requests a change to an employee's information."
)
def update_employee_info(
    identifier: str = Field(
        description="Name, first name, emp ID, or email."
    ),
    field: str = Field(
        description="Field to update: 'location', 'role', 'team', or 'email'."
    ),
    new_value: str = Field(
        description="New value to set."
    ),
) -> dict:
    ALLOWED_FIELDS = {"location", "role", "team", "email"}

    if field not in ALLOWED_FIELDS:
        return {
            "updated": False,
            "error": f"Field '{field}' cannot be updated.",
            "allowed_fields": sorted(ALLOWED_FIELDS),
        }

    identifier_lower = identifier.lower().strip()
    for emp in FAKE_EMPLOYEES:
        if (
            emp["id"].lower()    == identifier_lower or
            emp["name"].lower()  == identifier_lower or
            emp["email"].lower() == identifier_lower or
            emp["role"].lower()  == identifier_lower or
            emp["name"].lower().startswith(identifier_lower)
        ):
            old_value = emp[field]
            emp[field] = new_value
            return {
                "updated": True,
                "employee_id": emp["id"],
                "employee_name": emp["name"],
                "field": field,
                "old_value": old_value,
                "new_value": new_value,
            }

    # Employee not found — surface close matches so the model can self-correct
    identifier_words = set(identifier_lower.split())
    close_matches = [
        f"{emp['name']} ({emp['role']})"
        for emp in FAKE_EMPLOYEES
        if identifier_words & (
            set(emp["name"].lower().split()) |
            set(emp["role"].lower().split())
        )
    ]

    error_response = {
        "updated": False,
        "error": f"No employee found matching '{identifier}'. No changes were made.",
    }
    if close_matches:
        error_response["possible_matches"] = close_matches
        error_response["hint"] = "Did you mean one of these? Retry with the exact name."
    else:
        error_response["hint"] = "Check the spelling or try a different identifier (name, email, or employee ID)."

    return error_response


@mcp.tool(
    name="create_announcement",
    description="Create a company-wide announcement. Use when the user wants to publish news, updates, or events for all employees."
)
def create_announcement(
    title: str = Field(
        description="Exact title as provided by the user."
    ),
    summary: str = Field(
        description="Exact summary as provided by the user."
    ),
    author: str = Field(
        description="Publisher name, e.g. 'HR Team'. Ask if unclear."
    ),
) -> dict:
    from datetime import date

    if not title.strip():
        return {"created": False, "error": "Title cannot be empty."}
    if not summary.strip():
        return {"created": False, "error": "Summary cannot be empty."}
    if not author.strip():
        return {"created": False, "error": "Author cannot be empty."}

    new_id = f"ann_{len(FAKE_ANNOUNCEMENTS) + 1:03d}"

    new_announcement = {
        "id": new_id,
        "title": title.strip(),
        "summary": summary.strip(),
        "date": date.today().isoformat(),
        "author": author.strip(),
    }

    FAKE_ANNOUNCEMENTS.insert(0, new_announcement)  # newest first

    return {
        "created": True,
        "announcement": new_announcement,
        "message": "Announcement published",
    }

# ══════════════════════════════════════════════════════════════
# RESOURCES
# ══════════════════════════════════════════════════════════════

@mcp.resource(
    "company://pages", 
    mime_type="application/json"
    )
def resource_all_pages() -> list:
    """Intranet page index — id, title, category, author, last_updated. No full content."""

    return [
        {"id": p["id"], "title": p["title"], "category": p["category"],
         "author": p["author"], "last_updated": p["last_updated"]}
        for p in FAKE_PAGES
    ]

@mcp.resource(
    "company://pages/{page_id}", 
    mime_type="application/json"
    )
def resource_page_by_id(page_id: str) -> dict:
    """Full content of a single intranet page by ID (e.g. page_001)."""

    for page in FAKE_PAGES:
        if page["id"].lower() == page_id.lower():
            return page
    raise ValueError(f"No page with ID '{page_id}'")

@mcp.resource(
    "company://employees", 
    mime_type="application/json"
    )
def resource_all_employees() -> list:
    """Full employee directory."""
    return FAKE_EMPLOYEES


@mcp.resource(
    "company://employees/{employee_id}", 
    mime_type="application/json"
    )
def resource_employee_by_id(employee_id: str) -> dict:
    """Single employee record by ID (e.g. emp_001)."""
    for emp in FAKE_EMPLOYEES:
        if emp["id"].lower() == employee_id.lower():
            return emp
    raise ValueError(f"No employee with ID '{employee_id}'")

@mcp.resource(
    "company://announcements", 
    mime_type="application/json"
    )
def resource_all_announcements() -> list:
    """All announcements, newest first. Reflects any created this session."""
    return FAKE_ANNOUNCEMENTS


# ══════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════

@mcp.prompt(
    name="onboard_employee",
    description="Generate a personalised onboarding checklist for a new employee"
)
def prompt_onboard_employee(
    employee_id: str = Field(description="ID of the new employee, e.g. emp_001")
) -> list:
    return [base.UserMessage(f"""
    A new employee has just joined the company. Their employee ID is: {employee_id}

    Do the following in order:
    1. Use get_employee_info to retrieve their name, role, team, and location.
    2. Use search_intranet to find the onboarding guide relevant to their team or role.
    3. Use get_recent_announcements to find any recent company news they should know about on day one.

    Then produce a personalised onboarding summary with:
    - A welcome message using their name and role
    - A checklist of their first-week steps (from the onboarding guide)
    - 2-3 recent announcements they should be aware of
    - Key contacts on their team (from the employee directory if available)

    Be warm and practical. This will be sent directly to the new employee.
    """)]

@mcp.prompt(
    name="policy_summary",
    description="Look up a company policy and produce a plain-language summary with key rules"
)
def prompt_policy_summary(
    topic: str = Field(description="Policy topic to look up, e.g. 'remote work', 'travel expenses', 'code review'")
) -> list:
    return [base.UserMessage(f"""
    The user wants a clear summary of the company policy on: {topic}

    Do the following:
    1. Use search_intranet with the most relevant keywords for "{topic}".
    2. If the excerpt is not detailed enough, use get_page_full_content with the returned page ID.

    Then write a plain-language summary with:
    - **What the policy allows** (the key permissions or entitlements)
    - **What the policy requires** (approval steps, limits, deadlines)
    - **What to watch out for** (common gotchas or restrictions)

    Use bullet points. Avoid quoting the policy verbatim — explain it as a helpful colleague would.
    If the policy is not found, say so clearly and suggest the user contact HR.
    """)]


if __name__ == "__main__":
    print("Starting Mock MCP Server (stdio transport)...", file=sys.stderr, flush=True)
    mcp.run(transport="stdio")
