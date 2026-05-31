from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Sequence
from typing import Annotated, Any, Literal

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import PromptMessage, TextContent
from pydantic import BaseModel, Field, StringConstraints


# ═════════════════════════════════════════════════════════════════════════════
#                                   CONFIG
# ═════════════════════════════════════════════════════════════════════════════

BASE_URL   = os.getenv("DIGGSPACE_API_BASE_URL", "https://localhost:44313").rstrip("/")
TOKEN      = os.getenv("DIGGSPACE_BEARER_TOKEN", "")
VERIFY_SSL = os.getenv("DIGGSPACE_VERIFY_SSL", "true").lower() != "false"


# ═════════════════════════════════════════════════════════════════════════════
#                       SYSTEM PROMPT  (orchestration only)
# ═════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
HARD RULES — read these first, they override everything else
• Never fabricate IDs, names, image paths, routes, or field values. Use only values
  returned by tool calls or supplied by the user.
• Never call a write tool (create_*, update_*, seed_channel_content) without
  first presenting a confirmation block and receiving explicit user approval.
• Never call `get_*` before an `update_*` — every update tool fetches current
  state internally and merges. Calling get first wastes tokens.
• Never use `list_*` to look up by name — use `resolve_scope` / `resolve_channel`.
• Never guess a channel or scope ID. IDs come from tool responses only.
• Content bodies must be valid HTML. Never use plain text — wrap in <p>...</p>.
• On resolver 0 matches: tell the user, ask them to check spelling. Do not retry.
• On resolver 2+ matches: list candidates and ask which. Never pick silently.

ROLE
You are an AI assistant for the Diggspace CMS. You help admins manage scopes
(workspaces), channels (sections), content (articles or events), and global settings.
Hierarchy: Global Settings > Scopes > Channels > Content (Articles / Events).

WRITE CONFIRMATION PROTOCOL
For every write tool, before calling it:
  1. CLARIFY ambiguous or possibly misspelled names before proceeding.
  2. PRESENT a confirmation block:
       • Entity type, name, id (if known)
       • Every field you will send. Mark omitted fields as "unchanged" for updates.
       • Any follow-up calls planned (e.g. "then seed 3 content items").
     End with: "Shall I proceed, or would you like to change anything?"
  3. WAIT for explicit confirmation. A follow-up question is NOT confirmation.
  4. REPORT after each write — what changed and the returned id. On error,
     surface the API `hint` field verbatim and suggest a next action.

TOOL ROUTING
- Lookup by name       → `resolve_scope` or `resolve_channel`
- Show inventory       → `list_scopes` / `list_channels` / `list_content`
- Read full config     → `get_scope` / `get_channel` / `get_content` / `get_global_settings`
- Patch an entity      → `update_scope` / `update_channel` / `update_content` / `update_global_settings`
  All update tools share the same contract: pass only the fields you want changed —
  omitted fields are preserved. Each tool fetches current state internally and merges
  before sending. List-type fields (e.g. tabs, footer_block_links, sticky_channel_ids,
  frequent_questions) replace the entire list when passed — they are not appended to.
- Build scope + channels together → `create_scope_with_channels` (preferred over chaining)
- Seed multiple content items → `seed_channel_content` (preferred over looping `create_content`)
- Create content              → `create_content`. Pass `event_start` + `event_end` for Events;
  omit them for Articles. The API determines content type automatically.
  Add `location` for Events when known.
- End of build run                → `verify_scope`

SESSION START
Call `check_connection` exactly once as the very first tool call. Do not call it again
unless a later tool returns a 4xx/5xx error. A zero-result from a list or resolve tool
is NOT an error — it means the item does not exist.

ERROR HANDLING
All tool responses are JSON envelopes. On failure, surface the `hint` field — never retry blindly.
  401 → token expired; ask the user to refresh DIGGSPACE_BEARER_TOKEN.
  403 → missing role; tell the user what role is required.
  404 → ID not found; suggest `resolve_scope` / `resolve_channel`.
  409 → name/route collision; suggest a different name.
  500 → server bug; show the traceId and ask the user to report it.

GENERAL
Be concise. Use bullet lists for multi-item results."""


mcp = FastMCP("diggspace", instructions=SYSTEM_PROMPT)


# ═════════════════════════════════════════════════════════════════════════════
#                            SHARED TYPE ALIASES
# ═════════════════════════════════════════════════════════════════════════════

HexColor    = Annotated[str, StringConstraints(pattern=r"^#[0-9a-fA-F]{6}$")]
LanguageTag = Annotated[str, StringConstraints(pattern=r"^[a-z]{2}(-[A-Z]{2})?$")]

SidebarWidget = Literal[
    "birthdays",
    "workAnniversaries",
    "upcomingEvents",
    "recentDocuments",
]

SidebarNavigationKey = Literal[
    "apps",
    "collaborators",
    "documents",
    "workgroups",
    "ideation",
    "learning",
    "approvalRequests",
]

ChannelTab  = Literal["Articles", "Pages"]
ChannelRole = Literal["None", "Reader", "Contributor", "Author", "Editor", "Administrator"]
ChannelType = Literal["public", "private", "corporate"]


# ═════════════════════════════════════════════════════════════════════════════
#                           NESTED INPUT MODELS
# ═════════════════════════════════════════════════════════════════════════════

class SocialLink(BaseModel):
    type: Literal[
        "facebookLink", "twitterLink", "linkedInLink",
        "instagramLink", "youtubeLink",
    ] = Field(description="Social platform identifier.")
    link: str = Field(description="Full HTTPS URL to the profile.")


class NavLink(BaseModel):
    caption  : str  = Field(description="Visible text.")
    link     : str  = Field(description="URL or in-app path.")
    newWindow: bool = Field(default=False, description="Open in a new tab.")


class NavigationLinks(BaseModel):
    topBarLinks: list[NavLink]    = Field(default_factory=list)
    footerLinks: list[NavLink]    = Field(default_factory=list)
    socialLinks: list[SocialLink] = Field(default_factory=list)


class FooterBlockGroup(BaseModel):
    title: str           = Field(description="Heading of this footer column.")
    icon : str | None    = Field(default=None, description="Optional icon key.")
    links: list[NavLink] = Field(description="Links under this heading.")


class FAQItem(BaseModel):
    question: str = Field(description="FAQ question.")
    answer  : str = Field(description="FAQ answer.")


class LoginQuickLink(BaseModel):
    text: str = Field(description="Visible text for this quick link.")
    url : str = Field(description="URL for this quick link.")


# ─── Composite-tool input models ─────────────────────────────────────────────

class ContentSpec(BaseModel):
    """Full specification for a single content item — either an article or an event.
    Used by `seed_channel_content`. All fields are written verbatim into the content-create payload.
    Provide `event_start` and `event_end` to create an Event; omit them for an Article.
    The API determines content type automatically from the presence of `event_start`.
    """
    title                        : str                          = Field(description="Title, plain text, no HTML.")
    body                         : str                          = Field(description="Body as HTML. Wrap plain text in <p>...</p>. Use <h2>/<h3> for headings.")
    status                       : Literal["Draft", "Published"] = Field(default="Published", description="'Draft' = saved but invisible. 'Published' = live.")
    tags                         : list[str]                    = Field(default_factory=list, description="Free-text tags, e.g. ['announcement', 'q4-2024']. Empty list is fine.")
    is_main_highlight            : bool                         = Field(default=False, description="Feature on the scope homepage carousel.")
    is_sticky                    : bool                         = Field(default=False, description="Pin to the top of the channel feed.")
    hide_likes                   : bool                         = Field(default=False, description="Disable the like button on this item.")
    hide_comments                : bool                         = Field(default=False, description="Disable the comment section.")
    hide_image_in_article_detail : bool                         = Field(default=False, description="Hide hero image on the detail page (still shown in feed cards).")
    disable_auto_related_articles: bool                         = Field(default=False, description="Turn off the auto Related-content block.")
    require_read_confirmation    : bool                         = Field(default=False, description="Force users to click 'I have read this'. Use for policies, compliance, legal, security.")
    notify                       : bool                         = Field(default=False, description="Notify scope users on publish. Only meaningful when status='Published'. Use for major announcements and policies.")
    image_url                    : str | None                   = Field(default=None, description="CMS media path for the hero image, e.g. 'cms/media/.../hero.jpg'. Must already exist on the server. Omit if no image.")
    event_start                  : str | None                   = Field(default=None, description="ISO 8601 datetime for event start, e.g. '2026-06-01T09:00:00Z'. Providing this turns the item into an Event.")
    event_end                    : str | None                   = Field(default=None, description="ISO 8601 datetime for event end, e.g. '2026-06-01T17:00:00Z'. Required when event_start is provided.")
    location                     : str | None                   = Field(default=None, description="Plain-text event location, e.g. 'Conference Room A, HQ'. Only meaningful for Events.")
 

class ChannelConfig(BaseModel):
    """Full specification for one channel inside `create_scope_with_channels`.
    Mirrors the parameters of `create_channel` exactly — no content seeding here
    (use `seed_channel_content` after the scope is built)."""
    name                 : str             = Field(description="Channel display name, e.g. 'The Daily'. Server derives the URL slug.")
    description          : str             = Field(description="HTML description, e.g. '<p>Company-wide announcements</p>'. Never empty.")
    channel_type         : ChannelType     = Field(default="public", description="'public' = visible to all; 'private' = opt-in; 'corporate' = restricted (leadership/legal/compliance).")
    is_sticky            : bool            = Field(default=False, description="Pin this channel to the top of the scope's channel list.")
    color                : HexColor        = Field(default="#ffbb33", description="Channel chrome color, 6-digit hex (e.g. '#FF9900').")
    image_url            : str | None      = Field(default=None, description="Optional CMS media path for the channel image. Must already exist.")
    hide_on_homepage_feed: bool            = Field(default=False, description="Exclude this channel's content from the scope homepage feed.")
    hide_highlights      : bool            = Field(default=False, description="Disqualify this channel's content from highlight slots.")
    default_role         : ChannelRole     = Field(default="None", description="Role automatically granted to every user in the parent scope.")
    tabs                 : list[ChannelTab] = Field(default_factory=lambda: ["Articles", "Pages"], description="Tabs shown on the channel page. Order matters.")
    initial_tab          : ChannelTab      = Field(default="Articles", description="Active tab on load. Must also appear in `tabs`.")
    frequent_questions   : list[FAQItem]   = Field(default_factory=list, description="Optional FAQ list for this channel. Use [] for pure broadcast channels.")


# ═════════════════════════════════════════════════════════════════════════════
#                           HTTP CLIENT + HELPERS
# ═════════════════════════════════════════════════════════════════════════════

_http: httpx.AsyncClient | None = None


def _client() -> httpx.AsyncClient:
    global _http
    if _http is None:
        _http = httpx.AsyncClient(
            base_url=BASE_URL,
            verify  =VERIFY_SSL,
            timeout =30.0,
            headers ={
                "Authorization": f"Bearer {TOKEN}",
                "Accept"       : "application/json",
                "Content-Type" : "application/json",
            },
        )
    return _http


def _drop_none(d: dict) -> dict:
    """Strip keys whose value is None from a dict."""
    return {k: v for k, v in d.items() if v is not None}


def _dump(obj: Any) -> Any:
    """Convert Pydantic models (or lists of them) to plain dicts for JSON."""
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, list):
        return [_dump(x) for x in obj]
    return obj


def _build_theme(
    primary_color        : str | None,
    primary_color_contrast: str | None,
    secondary_color      : str | None,
    warn_color           : str | None,
    warn_color_contrast  : str | None,
) -> dict | None:
    """Build a theme dict from color arguments, returning None if all are absent."""
    return _drop_none({
        "primaryColor"        : primary_color,
        "primaryColorContrast": primary_color_contrast,
        "secondaryColor"      : secondary_color,
        "warnColor"           : warn_color,
        "warnColorContrast"   : warn_color_contrast,
    }) or None


def _build_image(image_url: str | None) -> dict | None:
    """Wrap an image URL in the API's expected object shape, or return None."""
    return {"url": image_url} if image_url else None


def _build_scope_components(sidebar_components: Sequence[str] | None) -> dict:
    """Build the scope components structure with optional sidebar widgets."""
    return {
        "mainComponents"   : [{"id": "", "type": "homepageFeed"}],
        "sideBarComponents": [{"id": "", "type": t} for t in (sidebar_components or [])],
    }


def _build_channel_components() -> dict:
    """Build the standard channel components structure."""
    return {"mainComponents": [{"id": "", "type": "mainChannelFeed"}]}


def _extract_content(body_data: Any) -> list:
    """Normalise the content response, which may be a plain list or a paginated envelope."""
    if isinstance(body_data, list):
        return body_data
    if isinstance(body_data, dict):
        return (
            body_data.get("items")
            or body_data.get("data")
            or body_data.get("results")
            or []
        )
    return []


async def _request(
    method: str,
    path  : str,
    *,
    params: dict | None = None,
    json_b: dict | None = None,
) -> str:
    """Execute one HTTP call and return a JSON-string envelope. Never raises."""
    verb = method.upper()
    try:
        resp = await _client().request(verb, path, params=params, json=json_b)
    except httpx.RequestError as e:
        return json.dumps({
            "error" : "network_error",
            "detail": f"Could not reach {BASE_URL}{path}: {e.__class__.__name__}: {e}",
        })

    try:
        body = resp.json() if resp.content else None
    except ValueError:
        body = resp.text

    if resp.is_error:
        return json.dumps({
            "error"      : "http_error",
            "status_code": resp.status_code,
            "url"        : f"{BASE_URL}{path}",
            "method"     : verb,
            "body"       : body,
        })

    return json.dumps({"status_code": resp.status_code, "data": body})


# ═════════════════════════════════════════════════════════════════════════════
#                           RESOLVER HELPERS
# ═════════════════════════════════════════════════════════════════════════════

_QUERY_NOISE_WORDS = frozenset({
    # scope-level
    "scope", "workspace", "workspaces", "scopes", "hub",
    # channel-level
    "channel", "channels", "section", "sections", "feed", "tab",
    # content-level
    "article", "articles", "event", "events", "post", "posts", "content",
    # generic
    "the", "a", "an", "my", "our",
})


def _tokenize_query(text: str) -> set[str]:
    """Split a string into lowercase words, dropping noise words."""
    return {w for w in text.lower().split() if w not in _QUERY_NOISE_WORDS}


def _fuzzy_matches(needle: str, needle_tokens: set[str], name: str) -> bool:
    """Return True if needle is an exact substring of name, or all needle tokens
    prefix-match at least one name token."""
    if needle in name:
        return True
    name_tokens = _tokenize_query(name)
    return bool(needle_tokens) and all(
        any(nt.startswith(qt) or qt.startswith(nt) for nt in name_tokens)
        for qt in needle_tokens
    )


# ═════════════════════════════════════════════════════════════════════════════
#                               CONNECTION
# ═════════════════════════════════════════════════════════════════════════════

@mcp.tool(
    name="check_connection",
    description="Verify the server is reachable and the bearer token is valid.",
)
async def check_connection() -> str:
    raw      = await _request("GET", "/cms/api/settings")
    envelope = json.loads(raw)
    if envelope.get("error"):
        return json.dumps({
            "ok"         : False,
            "error"      : envelope.get("error"),
            "detail"     : envelope.get("detail"),
            "status_code": envelope.get("status_code"),
        })
    return json.dumps({"ok": True, "status_code": envelope.get("status_code", 200)})


# ═════════════════════════════════════════════════════════════════════════════
#                                  SCOPES
# ═════════════════════════════════════════════════════════════════════════════

@mcp.tool(
    name="list_scopes",
    description="List every scope (workspace) in the hub — name, route, and id.",
)
async def list_scopes() -> str:
    raw      = await _request("GET", "/cms/api/scopes/")
    envelope = json.loads(raw)
    if envelope.get("error"):
        return raw
    scopes = envelope.get("data") or []
    lines  = [
        f"- {s.get('name', '?')} — route: {s.get('route', '?')} — id: {s.get('id', '?')}"
        for s in scopes
    ]
    return "\n".join(lines) if lines else "No scopes found."


@mcp.tool(
    name="get_scope",
    description=(
        "Fetch the full configuration of one scope by ID — branding, theme, "
        "footer, welcome message, navigation."
    ),
)
async def get_scope(
    scope_id: Annotated[str, Field(
        description=(
            "Opaque scope ID returned by `list_scopes` / `create_scope` / "
            "`resolve_scope`, e.g. '4jpp2bpb93dj8v7w7wpw44r2xe'. "
            "Never the route slug like 'global' or 'amazon-hub'."
        ),
    )],
) -> str:
    raw      = await _request("GET", f"/cms/api/scopes/{scope_id}")
    envelope = json.loads(raw)
    if envelope.get("error"):
        return raw
    d = envelope.get("data") or {}
    slim = {k: d.get(k) for k in (
        "id", "name", "route", "description", "welcomeMessage",
        "channelCreationByAdminsOnly", "numberOfHighlights",
        "highlightsCarousel", "highlightsStyle",
        "theme", "footerBlockLinks", "navigationLinks", "image",
    )}
    return json.dumps({"status_code": envelope.get("status_code", 200), "data": slim})


@mcp.tool(
    name="resolve_scope",
    description="Look up a scope's ID from a display name or route.",
)
async def resolve_scope(
    name_or_route: Annotated[str, Field(description="Display name or URL route — just the meaningful part.")],
) -> str:
    raw      = await _request("GET", "/cms/api/scopes/")
    envelope = json.loads(raw)
    if envelope.get("error"):
        return raw
    scopes        = envelope.get("data") or []
    needle        = name_or_route.strip().lower()
    needle_tokens = _tokenize_query(needle)

    def _matches(s: dict) -> bool:
        if needle == (s.get("route") or "").lower():
            return True
        return _fuzzy_matches(needle, needle_tokens, (s.get("name") or "").lower())

    matches = [s for s in scopes if _matches(s)]
    slim    = [{"name": s.get("name"), "route": s.get("route"), "id": s.get("id")} for s in matches]
    return json.dumps({"query": name_or_route, "match_count": len(slim), "matches": slim})


@mcp.tool(
    name="create_scope",
    description="Provision a new scope (workspace). The server generates the ID and URL route from the name.",
)
async def create_scope(
    name: Annotated[str, Field(
        description=(
            "Display name of the new scope, e.g. 'Acme Corp'. The server derives the "
            "URL route from this. Do not pre-format with slashes or lowercasing."
        ),
    )],
    description                    : Annotated[str | None,             Field(description="Short free-text description of the scope's purpose. Single sentence.")] = None,
    welcome_message                : Annotated[str | None,             Field(description="Homepage welcome message. 1-2 sentences. Emojis allowed.")] = None,
    channel_creation_by_admins_only: Annotated[bool,                   Field(description="If true, only scope admins can create channels. Recommended True for managed hubs.")] = False,
    number_of_highlights           : Annotated[int,                    Field(description="How many highlight items to render (0-10).", ge=0, le=10)] = 4,
    highlights_carousel            : Annotated[bool,                   Field(description="Render highlights as a rotating carousel.")] = True,
    highlights_style               : Annotated[Literal["BannerCarousel", "default"] | None, Field(description="Visual style for highlights.")] = "BannerCarousel",
    primary_color                  : Annotated[HexColor | None,        Field(description="Theme primary color, 6-digit hex e.g. '#0064f0'.")] = None,
    primary_color_contrast         : Annotated[HexColor | None,        Field(description="Contrast color paired with primary, e.g. '#ffffff' for dark primaries.")] = None,
    secondary_color                : Annotated[HexColor | None,        Field(description="Theme secondary color, 6-digit hex.")] = None,
    warn_color                     : Annotated[HexColor | None,        Field(description="Theme warning/accent color, 6-digit hex.")] = None,
    warn_color_contrast            : Annotated[HexColor | None,        Field(description="Contrast paired with warn_color.")] = None,
    image_url                      : Annotated[str | None,             Field(description="Relative CMS media path for the scope hero image. Must already exist on the server. Omit if no image.")] = None,
    administrators                 : Annotated[list[str] | None,       Field(description="User IDs who administer this scope. Empty list / omitted is fine.")] = None,
    footer_block_links             : Annotated[list[FooterBlockGroup] | None, Field(description="Grouped footer link columns. Each group has a title and a list of links. Omit for a plain scope.")] = None,
    navigation_links               : Annotated[NavigationLinks | None, Field(description="Top-bar links, footer-row links, and social icons. footerLinks is a flat list (NOT grouped — use footer_block_links for groups). socialLinks types: facebookLink, twitterLink, linkedInLink, instagramLink, youtubeLink.")] = None,
    sidebar_components             : Annotated[list[SidebarWidget] | None, Field(description="Optional sidebar widgets on the scope homepage. 'birthdays' = upcoming team birthdays, 'workAnniversaries' = work milestone celebrations, 'upcomingEvents' = calendar events feed, 'recentDocuments' = recently edited SharePoint/O365 documents. Omit or pass null for no sidebar.")] = None,
) -> str:
    body = _drop_none({
        "name"                       : name,
        "description"                : description,
        "welcomeMessage"             : welcome_message,
        "channelCreationByAdminsOnly": channel_creation_by_admins_only,
        "numberOfHighlights"         : number_of_highlights,
        "highlightsCarousel"         : highlights_carousel,
        "highlightsStyle"            : highlights_style,
        "administrators"             : administrators or [],
        "stickyChannelsIds"          : [],
        "highlights"                 : [],
        "defaultChannels"            : [],
        "favoriteApps"               : [],
        "footerBlockLinks"           : _dump(footer_block_links) or [],
        "navigationLinks"            : _dump(navigation_links),
        "image"                      : _build_image(image_url),
        "components"                 : _build_scope_components(sidebar_components),
        "theme"                      : _build_theme(primary_color, primary_color_contrast, secondary_color, warn_color, warn_color_contrast),
    })
    raw      = await _request("POST", "/cms/api/scopes", json_b=body)
    envelope = json.loads(raw)
    if envelope.get("error"):
        return raw
    d    = envelope.get("data") or {}
    slim = {"id": d.get("id"), "name": d.get("name"), "route": d.get("route")}
    return json.dumps({"status_code": envelope.get("status_code", 200), "created": slim})


@mcp.tool(
    name="update_scope",
    description=(
        "Update an existing scope — rename, retheme, swap footer, change navigation, "
        "or adjust highlights. Theme colors merge field-by-field."
    ),
)
async def update_scope(
    scope_id                       : Annotated[str,                    Field(description="ID of the scope to update.")],
    name                           : Annotated[str | None,             Field(description="New display name. Also changes the URL route.")] = None,
    description                    : Annotated[str | None,             Field(description="New description text.")] = None,
    welcome_message                : Annotated[str | None,             Field(description="New homepage welcome message.")] = None,
    channel_creation_by_admins_only: Annotated[bool | None,            Field(description="Flip the channel-creation permission policy.")] = None,
    number_of_highlights           : Annotated[int | None,             Field(description="How many highlight items to render (0-10).", ge=0, le=10)] = None,
    highlights_carousel            : Annotated[bool | None,            Field(description="Render highlights as a rotating carousel.")] = None,
    highlights_style               : Annotated[Literal["BannerCarousel", "default"] | None, Field(description="Visual style for highlights.")] = None,
    primary_color                  : Annotated[HexColor | None,        Field(description="New primary color (6-digit hex).")] = None,
    primary_color_contrast         : Annotated[HexColor | None,        Field(description="New primary contrast color.")] = None,
    secondary_color                : Annotated[HexColor | None,        Field(description="New secondary color.")] = None,
    warn_color                     : Annotated[HexColor | None,        Field(description="New warning color.")] = None,
    warn_color_contrast            : Annotated[HexColor | None,        Field(description="New warn contrast color.")] = None,
    administrators                 : Annotated[list[str] | None,       Field(description="Replace the administrators list entirely.")] = None,
    sticky_channel_ids             : Annotated[list[str] | None,       Field(description="Replace the sticky channel list entirely.")] = None,
    default_channels               : Annotated[list[str] | None,       Field(description="Replace the default-favorite channel list entirely.")] = None,
    footer_block_links             : Annotated[list[FooterBlockGroup] | None, Field(description="Replace ALL footer link groups with this list.")] = None,
    navigation_links               : Annotated[NavigationLinks | None, Field(description="Replace top-bar / footer / social nav in full.")] = None,
    sidebar_components             : Annotated[list[SidebarWidget] | None, Field(description="Replace the sidebar widget list entirely. Pass [] to clear all widgets. Omit to keep current.")] = None,
    image_url                      : Annotated[str | None,             Field(description="CMS media path for the scope hero image. Pass '' to clear it. Omit to keep current.")] = None,
    additional_fields              : Annotated[dict | None,            Field(description="Escape hatch for fields not exposed above. Merged shallowly.")] = None,
) -> str:
    raw      = await _request("GET", f"/cms/api/scopes/{scope_id}")
    envelope = json.loads(raw)
    if envelope.get("error"):
        return raw
    current = envelope.get("data") or {}

    updates = _drop_none({
        "name"                       : name,
        "description"                : description,
        "welcomeMessage"             : welcome_message,
        "channelCreationByAdminsOnly": channel_creation_by_admins_only,
        "numberOfHighlights"         : number_of_highlights,
        "highlightsCarousel"         : highlights_carousel,
        "highlightsStyle"            : highlights_style,
        "administrators"             : administrators,
        "stickyChannelsIds"          : sticky_channel_ids,
        "defaultChannels"            : default_channels,
    })

    theme_updates = _build_theme(primary_color, primary_color_contrast, secondary_color, warn_color, warn_color_contrast)
    if theme_updates:
        updates["theme"] = {**(current.get("theme") or {}), **theme_updates}

    if footer_block_links is not None:
        updates["footerBlockLinks"] = _dump(footer_block_links)
    if navigation_links is not None:
        updates["navigationLinks"] = _dump(navigation_links)
    if sidebar_components is not None:
        # Preserve the existing main components (or fall back to homepageFeed),
        # and replace the sidebar list with what the user passed.
        existing_main = (current.get("components") or {}).get("mainComponents") or [{"id": "", "type": "homepageFeed"}]
        updates["components"] = {
            "mainComponents"   : existing_main,
            "sideBarComponents": [{"id": "", "type": t} for t in sidebar_components],
        }
    if image_url is not None:
        # Empty string clears the image; any other value replaces it.
        updates["image"] = None if image_url == "" else {"url": image_url}
    if additional_fields:
        updates.update(additional_fields)

    PUT_FIELDS = {
        "name", "description", "welcomeMessage", "channelCreationByAdminsOnly",
        "numberOfHighlights", "highlightsCarousel", "highlightsStyle",
        "administrators", "stickyChannelsIds", "highlights", "defaultChannels",
        "favoriteApps", "footerBlockLinks", "navigationLinks", "image",
        "components", "theme",
    }
    LIST_FIELDS = (
        "administrators", "stickyChannelsIds", "highlights",
        "defaultChannels", "favoriteApps", "footerBlockLinks",
    )

    base = {k: v for k, v in current.items() if k in PUT_FIELDS}
    for f in LIST_FIELDS:
        if base.get(f) is None:
            base[f] = []

    merged = {**base, **updates}
    for f in LIST_FIELDS:
        if merged.get(f) is None:
            merged[f] = []

    return await _request("PUT", f"/cms/api/scopes/{scope_id}", json_b=merged)


@mcp.tool(
    name="verify_scope",
    description=(
        "Audit a finished scope: channel count, sticky channels, brand colors, "
        "highlights config, and per-channel content counts."
    ),
)
async def verify_scope(
    scope_id: Annotated[str, Field(description="ID of the scope to audit (from `create_scope` / `resolve_scope`).")],
) -> str:
    # 1. Fetch channel list
    raw_ch = await _request("GET", "/cms/api/channels", params={"scopeId": scope_id, "skip": 0, "take": 100})
    env_ch = json.loads(raw_ch)
    if env_ch.get("error"):
        return raw_ch
    channels = (env_ch.get("data") or {}).get("items") or []

    if not channels:
        return json.dumps({
            "scope_id"     : scope_id,
            "channel_count": 0,
            "warning"      : "No channels found — scope may be empty or id is wrong.",
        })

    # 2. Fetch scope config + all channel details + content item counts concurrently
    async def _fetch_channel_data(ch: dict) -> dict:
        ch_id   = ch.get("id", "")
        ch_name = ch.get("name", "?")
        raw_detail, raw_content = await asyncio.gather(
            _request("GET", f"/cms/api/channels/{ch_id}"),
            _request("GET", "/cms/api/content-management", params={"channelId": ch_id, "skip": 0, "take": 200}),
        )
        ch_detail     = json.loads(raw_detail).get("data") or {}
        env_content      = json.loads(raw_content)
        content_count = 0
        if not env_content.get("error"):
            content_count = len(_extract_content(env_content.get("data")))
        return {
            "channel"      : ch_name,
            "id"           : ch_id,
            "isSticky"     : ch.get("isSticky", False),
            "color"        : ch_detail.get("color"),
            "tabs"         : ch_detail.get("tabs"),
            "faq_count"    : len(ch_detail.get("frequentQuestions") or []),
            "content_count": content_count,
        }

    channel_reports, raw_scope = await asyncio.gather(
        asyncio.gather(*[_fetch_channel_data(ch) for ch in channels]),
        _request("GET", f"/cms/api/scopes/{scope_id}"),
    )
    channel_reports = list(channel_reports)
    total_content  = sum(r["content_count"] for r in channel_reports)

    # 3. Evaluate scope config
    env_scope  = json.loads(raw_scope)
    scope_data = env_scope.get("data") or {}
    theme      = scope_data.get("theme") or {}

    sticky_channels = [r for r in channel_reports if r["isSticky"]]
    empty_channels  = [r["channel"] for r in channel_reports if r["content_count"] == 0]

    warnings = []
    if not sticky_channels:
        warnings.append("No sticky channels — consider pinning your primary channel.")
    if scope_data.get("numberOfHighlights") is None:
        warnings.append("Highlights not configured on scope.")
    if not theme.get("primaryColor"):
        warnings.append("No brand colors set on scope.")
    if empty_channels:
        warnings.append(f"Channels with no content: {', '.join(empty_channels)}")

    return json.dumps({
        "scope_id"            : scope_id,
        "scope_name"          : scope_data.get("name"),
        "route"               : scope_data.get("route"),
        "brand_colors"        : {
            "primary"  : theme.get("primaryColor"),
            "secondary": theme.get("secondaryColor"),
            "warn"     : theme.get("warnColor"),
        },
        "highlights"          : {
            "count"   : scope_data.get("numberOfHighlights"),
            "carousel": scope_data.get("highlightsCarousel"),
            "style"   : scope_data.get("highlightsStyle"),
        },
        "channel_count"       : len(channel_reports),
        "sticky_channel_count": len(sticky_channels),
        "total_content"      : total_content,
        "channels"            : channel_reports,
        "warnings"            : warnings,
        "status"              : "OK" if not warnings else "WARNINGS",
    }, indent=2)


# ═════════════════════════════════════════════════════════════════════════════
#                                 CHANNELS
# ═════════════════════════════════════════════════════════════════════════════

@mcp.tool(
    name="list_channels",
    description="List the channels (sections) inside one scope — name, type, id. Requires the parent scope ID.",
)
async def list_channels(
    scope_id: Annotated[str, Field(description="ID of the parent scope (from `resolve_scope`).")],
    skip    : Annotated[int, Field(description="Pagination offset. Defaults to 0.", ge=0)] = 0,
    take    : Annotated[int, Field(description="Page size. Defaults to 10, max 100.", ge=1, le=100)] = 10,
) -> str:
    raw      = await _request("GET", "/cms/api/channels", params={"scopeId": scope_id, "skip": skip, "take": take})
    envelope = json.loads(raw)
    if envelope.get("error"):
        return raw
    items = (envelope.get("data") or {}).get("items") or []
    lines = [
        f"- {c.get('name', '?')} — type: {c.get('type', '?')} — id: {c.get('id', '?')}"
        for c in items
    ]
    return "\n".join(lines) if lines else "No channels found in this scope."


@mcp.tool(
    name="get_channel",
    description="Fetch the full configuration of one channel — name, type, color, tabs, FAQ, component layout.",
)
async def get_channel(
    channel_id: Annotated[str, Field(description="Channel ID from `list_channels` / `create_channel` / `resolve_channel`.")],
) -> str:
    raw      = await _request("GET", f"/cms/api/channels/{channel_id}")
    envelope = json.loads(raw)
    if envelope.get("error"):
        return raw
    d    = envelope.get("data") or {}
    slim = {k: d.get(k) for k in (
        "id", "scopeId", "name", "description", "internalName",
        "isSticky", "color", "tabs", "initialTab",
        "hideOnHomepageFeed", "hideHighlights",
        "frequentQuestions", "childChannels",
        "staff",
    )}
    return json.dumps({"status_code": envelope.get("status_code", 200), "data": slim})


@mcp.tool(
    name="resolve_channel",
    description=(
        "Look up a channel's ID from a name within one scope. "
        "Scope ID is required — channel names are not unique across scopes."
    ),
)
async def resolve_channel(
    scope_id     : Annotated[str, Field(description="ID of the scope to search within (from `resolve_scope`). Required.")],
    name_or_route: Annotated[str, Field(description="Channel name — just the meaningful part.")],
) -> str:
    raw      = await _request("GET", "/cms/api/channels", params={"scopeId": scope_id, "skip": 0, "take": 100})
    envelope = json.loads(raw)
    if envelope.get("error"):
        return raw
    items         = (envelope.get("data") or {}).get("items") or []
    needle        = name_or_route.strip().lower()
    needle_tokens = _tokenize_query(needle)

    def _matches(c: dict) -> bool:
        if needle == (c.get("internalName") or "").lower():
            return True
        return _fuzzy_matches(needle, needle_tokens, (c.get("name") or "").lower())

    matches = [c for c in items if _matches(c)]
    slim    = [{"name": c.get("name"), "id": c.get("id"), "type": c.get("type")} for c in matches]
    return json.dumps({
        "query"      : name_or_route,
        "scope_id"   : scope_id,
        "match_count": len(slim),
        "matches"    : slim,
    })


@mcp.tool(
    name="create_channel",
    description="Add a new channel (section) to an existing scope. Requires the parent scope ID.",
)
async def create_channel(
    scope_id             : Annotated[str,                  Field(description="ID of the parent scope. Required.")],
    name                 : Annotated[str,                  Field(description="Channel display name e.g. 'News', 'HR'. Server derives the route.")],
    description          : Annotated[str | None,           Field(description="HTML description e.g. '<p>Company-wide announcements</p>'.")] = None,
    channel_type         : Annotated[ChannelType,          Field(description="'public' = everyone; 'private' = opt-in; 'corporate' = restricted.")] = "public",
    is_sticky            : Annotated[bool,                 Field(description="Pin to the top of the scope's channel list.")] = False,
    color                : Annotated[HexColor,             Field(description="Channel chrome color, 6-digit hex.")] = "#ffbb33",
    image_url            : Annotated[str | None,           Field(description="Optional CMS media path. Must already exist.")] = None,
    hide_on_homepage_feed: Annotated[bool,                 Field(description="Exclude from the main homepage feed.")] = False,
    hide_highlights      : Annotated[bool,                 Field(description="Disqualify from highlight slots.")] = False,
    default_role         : Annotated[ChannelRole,          Field(description="Role automatically granted to every user in the scope.")] = "None",
    tabs                 : Annotated[list[ChannelTab] | None, Field(description="Tabs shown on the channel page. Order matters.")] = None,
    initial_tab          : Annotated[ChannelTab,           Field(description="Active tab on load. Must also appear in tabs.")] = "Articles",
    child_channels       : Annotated[list[str] | None,     Field(description="Existing channel IDs to nest beneath this one.")] = None,
    frequent_questions   : Annotated[list[FAQItem] | None, Field(description="Optional FAQ list.")] = None,
) -> str:
    body = _drop_none({
        "scopeId"           : scope_id,
        "name"              : name,
        "description"       : description or "",
        "type"              : channel_type,
        "isSticky"          : is_sticky,
        "color"             : color,
        "imageUrl"          : image_url,
        "hideOnHomepageFeed": hide_on_homepage_feed,
        "hideHighlights"    : hide_highlights,
        "defaultRole"       : default_role,
        "childChannels"     : child_channels or [],
        "frequentQuestions" : _dump(frequent_questions) or [],
        "tabs"              : tabs or ["Articles", "Pages"],
        "initialTab"        : initial_tab,
        "components"        : _build_channel_components(),
    })
    raw      = await _request("POST", "/cms/api/channels/", json_b=body)
    envelope = json.loads(raw)
    if envelope.get("error"):
        return raw
    d    = envelope.get("data") or {}
    slim = {"id": d.get("id"), "name": d.get("name"), "internalName": d.get("internalName")}
    return json.dumps({"status_code": envelope.get("status_code", 200), "created": slim})


@mcp.tool(
    name="update_channel",
    description=(
        "Update an existing channel — rename, recolor, change type, toggle flags, "
        "or replace tabs, FAQ, and components."
    ),
)
async def update_channel(
    channel_id           : Annotated[str,                          Field(description="ID of the channel to update.")],
    name                 : Annotated[str | None,                   Field(description="New name. Changes the internal route too.")] = None,
    description          : Annotated[str | None,                   Field(description="New HTML description.")] = None,
    channel_type         : Annotated[ChannelType | None,           Field(description="Change visibility tier.")] = None,
    is_sticky            : Annotated[bool | None,                  Field(description="Pin/unpin at top of scope.")] = None,
    color                : Annotated[HexColor | None,              Field(description="New chrome color (6-digit hex).")] = None,
    image_url            : Annotated[str | None,                   Field(description="New CMS media path for channel image. Pass '' to clear. Omit to preserve current.")] = None,
    hide_on_homepage_feed: Annotated[bool | None,                  Field(description="Toggle homepage feed visibility.")] = None,
    hide_highlights      : Annotated[bool | None,                  Field(description="Toggle highlight eligibility.")] = None,
    default_role         : Annotated[ChannelRole | None,           Field(description="Change the auto-granted role.")] = None,
    tabs                 : Annotated[list[ChannelTab] | None,      Field(description="Replace the tabs list entirely.")] = None,
    initial_tab          : Annotated[ChannelTab | None,            Field(description="New landing tab. Must appear in tabs.")] = None,
    child_channels       : Annotated[list[str] | None,             Field(description="Replace child channel list entirely.")] = None,
    frequent_questions   : Annotated[list[FAQItem] | None,         Field(description="Replace the FAQ list entirely.")] = None,
    additional_fields    : Annotated[dict | None,                  Field(description="Escape hatch for fields not exposed above.")] = None,
) -> str:
    raw      = await _request("GET", f"/cms/api/channels/{channel_id}")
    envelope = json.loads(raw)
    if envelope.get("error"):
        return raw
    current = envelope.get("data") or {}

    updates = _drop_none({
        "name"              : name,
        "description"       : description,
        "type"              : channel_type,
        "isSticky"          : is_sticky,
        "color"             : color,
        "hideOnHomepageFeed": hide_on_homepage_feed,
        "hideHighlights"    : hide_highlights,
        "defaultRole"       : default_role,
        "tabs"              : tabs,
        "initialTab"        : initial_tab,
        "childChannels"     : child_channels,
    })
    if frequent_questions is not None:
        updates["frequentQuestions"] = _dump(frequent_questions)
    if image_url is not None:
        # Empty string clears the image; any other value replaces it.
        updates["imageUrl"] = None if image_url == "" else image_url
    if additional_fields:
        updates.update(additional_fields)

    PUT_FIELDS = {
        "scopeId", "name", "description", "type", "isSticky", "color",
        "imageUrl", "hideOnHomepageFeed", "hideHighlights", "defaultRole",
        "childChannels", "frequentQuestions", "tabs", "initialTab",
    }
    LIST_FIELDS = ("childChannels", "frequentQuestions")

    base = {k: v for k, v in current.items() if k in PUT_FIELDS}

    # The GET response shape uses isCorporate/isPrivate booleans instead of a
    # single `type` field — derive `type` from those flags.
    if current.get("isCorporate"):
        base["type"] = "corporate"
    elif current.get("isPrivate"):
        base["type"] = "private"
    else:
        base["type"] = "public"

    # `defaultRole` lives under `staff` on the GET response.
    base["defaultRole"] = (current.get("staff") or {}).get("defaultRole", "None")

    # `image` on GET is an object; PUT expects a flat `imageUrl` string.
    base["imageUrl"] = (current.get("image") or {}).get("url") or None

    base["components"] = _build_channel_components()

    for f in LIST_FIELDS:
        if base.get(f) is None:
            base[f] = []

    merged = {**base, **updates, "id": channel_id}
    for f in LIST_FIELDS:
        if merged.get(f) is None:
            merged[f] = []

    return await _request("PUT", f"/cms/api/channels/{channel_id}", json_b=merged)


# ═════════════════════════════════════════════════════════════════════════════
#                                 CONTENT
# ═════════════════════════════════════════════════════════════════════════════

@mcp.tool(
    name="create_content",
    description=(
        "Publish or draft a single article or event in a channel. Requires the channel ID."
    ),
)
async def create_content(
    channel_id                   : Annotated[str,                       Field(description="Destination channel ID (from `resolve_channel` / `list_channels`). Confirm with the user if ambiguous.")],
    title                        : Annotated[str,                       Field(description="Title, plain text, no HTML.")],
    body                         : Annotated[str,                       Field(description="Body as HTML. Wrap plain text in <p>...</p>. Use <h2>/<h3> for headings, <ul>/<ol> for lists.")],
    status                       : Annotated[Literal["Draft", "Published"], Field(description="'Draft' = saved invisibly. 'Published' = live. Default 'Draft' when intent is unclear.")] = "Draft",
    tags                         : Annotated[list[str] | None,          Field(description="Free-text tags, e.g. ['announcement', 'policy']. Empty list / omitted is fine.")] = None,
    is_main_highlight            : Annotated[bool,                      Field(description="Feature on the scope homepage carousel.")] = False,
    is_sticky                    : Annotated[bool,                      Field(description="Pin to the top of the channel feed.")] = False,
    hide_likes                   : Annotated[bool,                      Field(description="Disable the like button on this item.")] = False,
    hide_comments                : Annotated[bool,                      Field(description="Disable the comment section.")] = False,
    hide_image_in_article_detail : Annotated[bool,                      Field(description="Hide hero image on the detail page (still shown in feed cards).")] = False,
    disable_auto_related_articles: Annotated[bool,                      Field(description="Turn off the auto Related-content block.")] = False,
    require_read_confirmation    : Annotated[bool,                      Field(description="Force users to click 'I have read this'. Use for policies, compliance, legal, security.")] = False,
    notify                       : Annotated[bool,                      Field(description="Notify scope users on publish. Only meaningful when status='Published'.")] = False,
    image_url                    : Annotated[str | None,                Field(description="CMS media path for the hero image, e.g. 'cms/media/.../hero.jpg'. Must already exist on the server. Omit if no image.")] = None,
    event_start                  : Annotated[str | None,                Field(description="ISO 8601 datetime for event start, e.g. '2026-06-01T09:00:00Z'. Providing this makes the item an Event; omitting it makes it an Article.")] = None,
    event_end                    : Annotated[str | None,                Field(description="ISO 8601 datetime for event end, e.g. '2026-06-01T17:00:00Z'. Required when event_start is provided.")] = None,
    location                     : Annotated[str | None,                Field(description="Plain-text event location, e.g. 'Conference Room A, HQ'. Only meaningful for Events.")] = None,
) -> str:
    payload = _drop_none({
        "title"                     : title,
        "body"                      : body,
        "channelId"                 : channel_id,
        "status"                    : status,
        "tags"                      : tags or [],
        "isMainHighlight"           : is_main_highlight,
        "isSticky"                  : is_sticky,
        "hideLikes"                 : hide_likes,
        "hideComments"              : hide_comments,
        "hideImageInArticleDetail"  : hide_image_in_article_detail,
        "disableAutoRelatedArticles": disable_auto_related_articles,
        "requireReadConfirmation"   : require_read_confirmation,
        "notify"                    : notify,
        "image"                     : _build_image(image_url),
        "eventStart"                : event_start,
        "eventEnd"                  : event_end,
        "location"                  : location,
    })
    return await _request("POST", "/cms/api/content-management", json_b=payload)


@mcp.tool(
    name="get_content",
    description="Fetch one content item (article or event) by ID — title, body HTML, tags, image, channel, all flags, and event fields (eventStart, eventEnd, location) when present.",
)
async def get_content(
    content_id: Annotated[str, Field(description="Opaque content ID (from `list_content` / `create_content`). Works for both articles and events.")],
) -> str:
    raw      = await _request("GET", f"/cms/api/content-management/{content_id}")
    envelope = json.loads(raw)
    if envelope.get("error"):
        return raw
    d    = envelope.get("data") or {}
    slim = {k: d.get(k) for k in (
        "id", "title", "body", "tags", "publishDate", "url",
        "image", "documents", "channel",
        "status", "isMainHighlight", "isSticky", "notify",
        "hideLikes", "hideComments", "hideImageInArticleDetail",
        "disableAutoRelatedArticles", "requireReadConfirmation",
        "eventStart", "eventEnd", "location",
    )}
    return json.dumps({"status_code": envelope.get("status_code", 200), "data": slim})


@mcp.tool(
    name="update_content",
    description=(
        "Update an existing article or event — change title, body, tags, image, flags, status, or move it to a different channel"
    ),
)
async def update_content(
    content_id                   : Annotated[str,                       Field(description="ID of the content item to update (from `list_content` / `create_content`). Works for both articles and events.")],
    title                        : Annotated[str | None,                Field(description="New title, plain text. Omit to keep current.")] = None,
    body                         : Annotated[str | None,                Field(description="New body as HTML. Wrap plain text in <p>...</p>. Omit to keep current.")] = None,
    channel_id                   : Annotated[str | None,                Field(description="Move the content item to this channel ID. Omit to keep it in its current channel.")] = None,
    status                       : Annotated[Literal["Draft", "Published"] | None, Field(description="Change publication status. Omit to keep current. ⚠️ Defaults to 'Published' if omitted — pass 'Draft' explicitly for drafts.")] = None,
    tags                         : Annotated[list[str] | None,          Field(description="REPLACE the tag list entirely. Pass [] to clear all tags. Omit to keep current.")] = None,
    is_main_highlight            : Annotated[bool | None,               Field(description="Toggle homepage-carousel feature.")] = None,
    is_sticky                    : Annotated[bool | None,               Field(description="Pin/unpin in channel feed.")] = None,
    hide_likes                   : Annotated[bool | None,               Field(description="Toggle like button.")] = None,
    hide_comments                : Annotated[bool | None,               Field(description="Toggle comment section.")] = None,
    hide_image_in_article_detail : Annotated[bool | None,               Field(description="Toggle hero image on detail page.")] = None,
    disable_auto_related_articles: Annotated[bool | None,               Field(description="Toggle auto Related-content block.")] = None,
    require_read_confirmation    : Annotated[bool | None,               Field(description="Toggle the 'I have read this' requirement.")] = None,
    notify                       : Annotated[bool | None,               Field(description="Notify scope users on this update. Default False on edits — only set True for major republishes.")] = None,
    image_url                    : Annotated[str | None,                Field(description="New CMS media path for hero image. Pass '' to clear it. Omit to keep current.")] = None,
    event_start                  : Annotated[str | None,                Field(description="ISO 8601 datetime for event start. Omit to keep current.")] = None,
    event_end                    : Annotated[str | None,                Field(description="ISO 8601 datetime for event end. Omit to keep current.")] = None,
    location                     : Annotated[str | None,                Field(description="Plain-text event location. Pass '' to clear. Omit to keep current.")] = None,
) -> str:
    raw      = await _request("GET", f"/cms/api/content-management/{content_id}")
    envelope = json.loads(raw)
    if envelope.get("error"):
        return raw
    current = envelope.get("data") or {}
 
    base: dict = {
        "title"                     : current.get("title"),
        "body"                      : current.get("body"),
        "channelId"                 : (current.get("channel") or {}).get("id"),
        "tags"                      : current.get("tags") or [],
        "documents"                 : current.get("documents") or [],
        "relatedArticles"           : [],
        "isMainHighlight"           : current.get("isMainHighlight", False),
        "isSticky"                  : current.get("isSticky", False),
        "hideLikes"                 : current.get("hideLikes", False),
        "hideComments"              : current.get("hideComments", False),
        "hideImageInArticleDetail"  : current.get("hideImageInArticleDetail", False),
        "disableAutoRelatedArticles": current.get("disableAutoRelatedArticles", False),
        "requireReadConfirmation"   : current.get("requireReadConfirmation", False),
        "notify"                    : False,
        "status"                    : "Published",
        "image"                     : current.get("image"),
        "eventStart"                : current.get("eventStart"),
        "eventEnd"                  : current.get("eventEnd"),
        "location"                  : current.get("location"),
    }
 
    overrides = _drop_none({
        "title"                     : title,
        "body"                      : body,
        "channelId"                 : channel_id,
        "status"                    : status,
        "tags"                      : tags,
        "isMainHighlight"           : is_main_highlight,
        "isSticky"                  : is_sticky,
        "hideLikes"                 : hide_likes,
        "hideComments"              : hide_comments,
        "hideImageInArticleDetail"  : hide_image_in_article_detail,
        "disableAutoRelatedArticles": disable_auto_related_articles,
        "requireReadConfirmation"   : require_read_confirmation,
        "notify"                    : notify,
        "eventStart"                : event_start,
        "eventEnd"                  : event_end,
    })
 
    if image_url is not None:
        # Empty string clears the image; any other value replaces it.
        overrides["image"] = None if image_url == "" else {"url": image_url}
 
    if location is not None:
        # Empty string clears the location; any other value replaces it.
        overrides["location"] = None if location == "" else location
 
    merged = {**base, **overrides}
    return await _request("PUT", f"/cms/api/content-management/{content_id}", json_b=merged)


@mcp.tool(
    name="list_content",
    description=(
        "List content items (articles and events) in one channel — returns channel_id, content_count, and a slim list of {id, title} per item."
    ),
)
async def list_content(
    channel_id: Annotated[str, Field(
        description=(
            "Channel ID to query (from `resolve_channel` / `list_channels` / "
            "`create_channel`). Never guess."
        ),
    )],
) -> str:
    raw      = await _request("GET", "/cms/api/content-management", params={"channelId": channel_id, "skip": 0, "take": 200})
    envelope = json.loads(raw)
    if envelope.get("error"):
        return raw

    items    = _extract_content(envelope.get("data"))
    content_items = [
        {"id": a.get("id"), "title": a.get("title")}
        for a in items
        if isinstance(a, dict)
    ]
    return json.dumps({
        "status_code"  : envelope.get("status_code", 200),
        "channel_id"   : channel_id,
        "content_count": len(content_items),
        "truncated"    : len(content_items) >= 200,
        "items"     : content_items,
    })


# ═════════════════════════════════════════════════════════════════════════════
#          COMPOSITE TOOLS  (multi-step operations — preferred for hub builds)
# ═════════════════════════════════════════════════════════════════════════════

@mcp.tool(
    name="create_scope_with_channels",
    description="Create a new scope AND all of its channels in a single call.",
)
async def create_scope_with_channels(
    name                           : Annotated[str,                    Field(description="Display name of the new scope, e.g. 'Acme Corp'. The server derives the URL route.")],
    channels                       : Annotated[list[ChannelConfig],    Field(description="Ordered list of channels to create inside the new scope. Each entry is a full ChannelConfig.")],
    description                    : Annotated[str | None,             Field(description="Short free-text description of the scope's purpose.")] = None,
    welcome_message                : Annotated[str | None,             Field(description="Homepage welcome message. 1-2 sentences.")] = None,
    channel_creation_by_admins_only: Annotated[bool,                   Field(description="If true, only scope admins can create channels. Recommended True for managed hubs.")] = False,
    number_of_highlights           : Annotated[int,                    Field(description="How many highlight items to render (0-10).", ge=0, le=10)] = 4,
    highlights_carousel            : Annotated[bool,                   Field(description="Render highlights as a rotating carousel.")] = True,
    highlights_style               : Annotated[Literal["BannerCarousel", "default"] | None, Field(description="Visual style for highlights.")] = "BannerCarousel",
    primary_color                  : Annotated[HexColor | None,        Field(description="Theme primary color, 6-digit hex.")] = None,
    primary_color_contrast         : Annotated[HexColor | None,        Field(description="Contrast paired with primary_color.")] = None,
    secondary_color                : Annotated[HexColor | None,        Field(description="Theme secondary color.")] = None,
    warn_color                     : Annotated[HexColor | None,        Field(description="Theme warning/accent color.")] = None,
    warn_color_contrast            : Annotated[HexColor | None,        Field(description="Contrast paired with warn_color.")] = None,
    image_url                      : Annotated[str | None,             Field(description="CMS media path for the scope hero image. Must already exist on the server.")] = None,
    footer_block_links             : Annotated[list[FooterBlockGroup] | None, Field(description="Grouped footer link columns.")] = None,
    navigation_links               : Annotated[NavigationLinks | None, Field(description="Top-bar links, footer-row links, and social icons.")] = None,
    sidebar_components             : Annotated[list[SidebarWidget] | None, Field(description="Optional sidebar widgets on the scope homepage. 'birthdays' = upcoming team birthdays, 'workAnniversaries' = work milestone celebrations, 'upcomingEvents' = calendar events feed, 'recentDocuments' = recently edited SharePoint/O365 documents. Omit or pass null for no sidebar.")] = None,
) -> str:
    # 1. Build and POST the scope
    scope_body = _drop_none({
        "name"                       : name,
        "description"                : description,
        "welcomeMessage"             : welcome_message,
        "channelCreationByAdminsOnly": channel_creation_by_admins_only,
        "numberOfHighlights"         : number_of_highlights,
        "highlightsCarousel"         : highlights_carousel,
        "highlightsStyle"            : highlights_style,
        "administrators"             : [],
        "stickyChannelsIds"          : [],
        "highlights"                 : [],
        "defaultChannels"            : [],
        "favoriteApps"               : [],
        "footerBlockLinks"           : _dump(footer_block_links) or [],
        "navigationLinks"            : _dump(navigation_links),
        "image"                      : _build_image(image_url),
        "components"                 : _build_scope_components(sidebar_components),
        "theme"                      : _build_theme(primary_color, primary_color_contrast, secondary_color, warn_color, warn_color_contrast),
    })

    raw_scope = await _request("POST", "/cms/api/scopes", json_b=scope_body)
    env_scope = json.loads(raw_scope)
    if env_scope.get("error"):
        return raw_scope

    scope_data  = env_scope.get("data") or {}
    scope_id    = scope_data.get("id")
    scope_name  = scope_data.get("name")
    scope_route = scope_data.get("route")

    if not scope_id:
        return json.dumps({"error": "scope_creation_failed", "detail": env_scope})

    # 2. Create each channel sequentially
    channel_map    = []
    channel_errors = []

    for ch in channels:
        ch_body = _drop_none({
            "scopeId"           : scope_id,
            "name"              : ch.name,
            "description"       : ch.description or "",
            "type"              : ch.channel_type,
            "isSticky"          : ch.is_sticky,
            "color"             : ch.color,
            "imageUrl"          : ch.image_url,
            "hideOnHomepageFeed": ch.hide_on_homepage_feed,
            "hideHighlights"    : ch.hide_highlights,
            "defaultRole"       : ch.default_role,
            "childChannels"     : [],
            "frequentQuestions" : _dump(ch.frequent_questions) or [],
            "tabs"              : ch.tabs or ["Articles", "Pages"],
            "initialTab"        : ch.initial_tab,
            "components"        : _build_channel_components(),
        })

        raw_ch = await _request("POST", "/cms/api/channels/", json_b=ch_body)
        env_ch = json.loads(raw_ch)

        if env_ch.get("error"):
            channel_errors.append({"channel": ch.name, "error": env_ch})
            channel_map.append({
                "name"     : ch.name,
                "id"       : None,
                "is_sticky": ch.is_sticky,
                "error"    : env_ch.get("error"),
            })
        else:
            ch_data = env_ch.get("data") or env_ch.get("created") or {}
            channel_map.append({
                "name"         : ch.name,
                "id"           : ch_data.get("id"),
                "internal_name": ch_data.get("internalName"),
                "is_sticky"    : ch.is_sticky,
            })

    result: dict = {
        "scope_id"       : scope_id,
        "scope_name"     : scope_name,
        "scope_route"    : scope_route,
        "channel_map"    : channel_map,
        "channels_ok"    : sum(1 for c in channel_map if c.get("id")),
        "channels_failed": len(channel_errors),
    }
    if channel_errors:
        result["channel_errors"] = channel_errors

    return json.dumps(result)


@mcp.tool(
    name="seed_channel_content",
    description=(
        "Create multiple articles or events in one channel in a single call. "
        "Idempotent — skips items whose title already exists in the channel, "
        "so safe to retry after partial failures."
    ),
)
async def seed_channel_content(
    channel_id: Annotated[str,               Field(description="Server-confirmed channel ID (from `create_scope_with_channels` / `list_channels`). Never guess.")],
    items  : Annotated[list[ContentSpec],    Field(description="Content items to create in this channel. Each ContentSpec is a complete item (article or event) with body HTML and all flags. Add event_start + event_end to make a spec an Event; omit them for an Article.")],
) -> str:
    raw_existing = await _request("GET", "/cms/api/content-management", params={"channelId": channel_id, "skip": 0, "take": 200})
    env_existing = json.loads(raw_existing)
    existing_titles: set[str] = set()
    existing_count  = 0
    if not env_existing.get("error"):
        existing_raw    = _extract_content(env_existing.get("data"))
        existing_count  = len(existing_raw)
        existing_titles = {
            (item.get("title") or "").strip().lower()
            for item in existing_raw
            if isinstance(item, dict)
        }

    to_create = [item for item in items if item.title.strip().lower() not in existing_titles]
 
    created_ids: list[str] = []
    failed     : list[dict] = []
 
    for spec in to_create:
        payload = _drop_none({
            "title"                     : spec.title,
            "body"                      : spec.body,
            "channelId"                 : channel_id,
            "status"                    : spec.status,
            "tags"                      : spec.tags or [],
            "isMainHighlight"           : spec.is_main_highlight,
            "isSticky"                  : spec.is_sticky,
            "hideLikes"                 : spec.hide_likes,
            "hideComments"              : spec.hide_comments,
            "hideImageInArticleDetail"  : spec.hide_image_in_article_detail,
            "disableAutoRelatedArticles": spec.disable_auto_related_articles,
            "requireReadConfirmation"   : spec.require_read_confirmation,
            "notify"                    : spec.notify,
            "image"                     : _build_image(spec.image_url),
            "eventStart"                : spec.event_start,
            "eventEnd"                  : spec.event_end,
            "location"                  : spec.location,
        })
        raw_item = await _request("POST", "/cms/api/content-management", json_b=payload)
        env_item = json.loads(raw_item)
        if env_item.get("error"):
            failed.append({"title": spec.title, "error": env_item.get("error")})
        else:
            created_ids.append((env_item.get("data") or {}).get("id", "unknown"))
 
    final_total = existing_count + len(created_ids)
    result: dict = {
        "channel_id"       : channel_id,
        "pre_existing"     : existing_count,
        "created_this_call": len(created_ids),
        "total_content"   : final_total,
        "target"           : len(items),
        "status"           : "complete" if final_total >= len(items) else "partial",
    }
    if failed:
        result["failed"] = failed
    return json.dumps(result)


# ═════════════════════════════════════════════════════════════════════════════
#                             GLOBAL SETTINGS
# ═════════════════════════════════════════════════════════════════════════════

@mcp.tool(
    name="get_global_settings",
    description=(
        "Read hub-wide configuration: company name, default language, supported languages, "
        "color theme, sidebar navigation, login quick links, and integration keys."
    ),
)
async def get_global_settings() -> str:
    raw      = await _request("GET", "/cms/api/settings")
    envelope = json.loads(raw)
    if envelope.get("error"):
        return raw
    d    = envelope.get("data") or {}
    slim = {
        "general"     : {k: v for k, v in (d.get("general") or {}).items()
                         if k in ("companyName", "websiteName", "welcomeMessage",
                                  "defaultCulture", "defaultLanguage", "supportedLanguages",
                                  "sidebarNavigation", "loginQuickLinks")},
        "colorTheme"  : d.get("colorTheme"),
        "engagement"  : d.get("engagement"),
        "integrations": {k: v for k, v in (d.get("integrations") or {}).items()
                         if k in ("googleAnalyticsIds", "googleMapsApiKey",
                                  "pbiAppId", "pbiTenantId")},
        "microsoft"   : {k: v for k, v in (d.get("microsoft") or {}).items()
                         if k in ("microsoft365EmailUrl", "microsoft365CalendarUrl")},
    }
    return json.dumps({"status_code": envelope.get("status_code", 200), "data": slim})


@mcp.tool(
    name="update_global_settings",
    description=(
        "Update hub-wide settings — company name, languages, navigation, color theme, "
        "engagement toggles, or third-party integration keys."
    ),
)
async def update_global_settings(
    company_name             : Annotated[str | None,                       Field(description="Company name in UI headers.")] = None,
    website_name             : Annotated[str | None,                       Field(description="Product/website name in page titles.")] = None,
    welcome_message          : Annotated[str | None,                       Field(description="Global welcome message (distinct from per-scope).")] = None,
    default_culture          : Annotated[str | None,                       Field(description="Default culture code e.g. 'en'.")] = None,
    default_language         : Annotated[LanguageTag | None,               Field(description="Default UI language tag e.g. 'en-US'.")] = None,
    supported_languages      : Annotated[list[LanguageTag] | None,         Field(description="Language tags the hub offers e.g. ['en-US', 'pt-PT'].")] = None,
    sidebar_navigation       : Annotated[list[SidebarNavigationKey] | None, Field(description="Ordered sidebar section keys. Omit entries to hide them.")] = None,
    login_quick_links        : Annotated[list[LoginQuickLink] | None,      Field(description="Links shown on the login page.")] = None,
    primary_color            : Annotated[HexColor | None,                  Field(description="Theme primary color, 6-digit hex.")] = None,
    primary_color_contrast   : Annotated[HexColor | None,                  Field(description="Contrast paired with primary.")] = None,
    secondary_color          : Annotated[HexColor | None,                  Field(description="Theme secondary color.")] = None,
    warn_color               : Annotated[HexColor | None,                  Field(description="Theme warning color.")] = None,
    warn_color_contrast      : Annotated[HexColor | None,                  Field(description="Contrast paired with warn_color.")] = None,
    hide_all_comments        : Annotated[bool | None,                      Field(description="Kill-switch: hide comment UI hub-wide.")] = None,
    hide_all_likes           : Annotated[bool | None,                      Field(description="Kill-switch: hide like UI hub-wide.")] = None,
    google_analytics_ids     : Annotated[str | None,                       Field(description="Google Analytics measurement ID(s).")] = None,
    google_maps_api_key      : Annotated[str | None,                       Field(description="Google Maps API key.")] = None,
    pbi_app_id               : Annotated[str | None,                       Field(description="Power BI app registration ID.")] = None,
    pbi_app_secret           : Annotated[str | None,                       Field(description="Power BI app secret. Sensitive.")] = None,
    pbi_tenant_id            : Annotated[str | None,                       Field(description="Power BI tenant ID.")] = None,
    microsoft365_email_url   : Annotated[str | None,                       Field(description="Outlook Web URL for the email widget.")] = None,
    microsoft365_calendar_url: Annotated[str | None,                       Field(description="Outlook Calendar URL for the calendar widget.")] = None,
) -> str:
    general_updates = _drop_none({
        "companyName"       : company_name,
        "websiteName"       : website_name,
        "welcomeMessage"    : welcome_message,
        "defaultCulture"    : default_culture,
        "defaultLanguage"   : default_language,
        "supportedLanguages": supported_languages,
        "sidebarNavigation" : sidebar_navigation,
        "loginQuickLinks"   : _dump(login_quick_links),
    })
    color_theme_updates = _build_theme(primary_color, primary_color_contrast, secondary_color, warn_color, warn_color_contrast) or {}
    engagement_updates  = _drop_none({
        "hideAllComments": hide_all_comments,
        "hideAllLikes"   : hide_all_likes,
    })
    integrations_updates = _drop_none({
        "googleAnalyticsIds": google_analytics_ids,
        "googleMapsApiKey"  : google_maps_api_key,
        "pbiAppId"          : pbi_app_id,
        "pbiAppSecret"      : pbi_app_secret,
        "pbiTenantId"       : pbi_tenant_id,
    })
    microsoft_updates = _drop_none({
        "microsoft365EmailUrl"   : microsoft365_email_url,
        "microsoft365CalendarUrl": microsoft365_calendar_url,
    })

    if not any([general_updates, color_theme_updates, engagement_updates,
                integrations_updates, microsoft_updates]):
        return json.dumps({
            "error" : "no_fields_provided",
            "detail": "Pass at least one setting field to update.",
        })

    raw_get = await _request("GET", "/cms/api/settings")
    get_env = json.loads(raw_get)
    if get_env.get("error"):
        return raw_get
    current = get_env.get("data") or {}

    body: dict = {}
    if general_updates:
        body["general"]      = {**(current.get("general") or {}),      **general_updates}
    if color_theme_updates:
        body["colorTheme"]   = {**(current.get("colorTheme") or {}),   **color_theme_updates}
    if engagement_updates:
        body["engagement"]   = {**(current.get("engagement") or {}),   **engagement_updates}
    if integrations_updates:
        body["integrations"] = {**(current.get("integrations") or {}), **integrations_updates}
    if microsoft_updates:
        body["microsoft"]    = {**(current.get("microsoft") or {}),    **microsoft_updates}

    return await _request("PATCH", "/cms/api/settings", json_b=body)


# ═════════════════════════════════════════════════════════════════════════════
#                                  PROMPTS
# ═════════════════════════════════════════════════════════════════════════════

@mcp.prompt(
    name="build_hub_from_profile",
    description=(
        "Build a FULLY POPULATED Diggspace hub from a company profile. "
        "Designs architecture, builds scope + channels with FAQs, seeds multiple "
        "realistic content items per primary channel, and polishes hub-wide settings. "
        "This is the production-quality onboarding flow — use it for real demos."
    ),
)
def build_hub_from_profile(company_profile: str) -> list:
    INSTRUCTIONS = """\
You are a senior digital workplace consultant onboarding a client onto the Diggspace platform.
You have been given a company profile. Your job is to design and build a complete, populated
intranet hub — not a skeleton. A real internal workspace has density: multiple content items per
channel, FAQs where they make sense, highlights, pinned posts, policies requiring read
confirmation, leadership updates, and hub-wide polish. You will use EVERY available feature
of the Diggspace platform — scope branding, channel types, sticky channels, content flags,
FAQs, global settings, highlights. No field left at its default if a better value exists.

═══════════════════════════════════════════════════════════════════════
PHASE 0 — DESIGN PROPOSAL
(Execute only if no SCOPE_ID exists yet; if a scope was already created,
skip directly to the next incomplete phase.)
═══════════════════════════════════════════════════════════════════════
Read the profile TWICE. Think about how this company ACTUALLY operates day-to-day.
What would leadership post? What policies must every employee acknowledge? What events,
milestones, and product launches are underway? What cultural rituals matter?
Produce a complete design across 4 sections:

1. SCOPE CONFIG
   - Scope name (professional, matches brand voice)
   - Description (short free-text, scope's purpose)
   - Welcome message (1-2 sentences, specific to this company)
   - image_url: CMS media path for scope hero image if available; else null
   - 5 brand hex colors (primary + contrast, secondary, warn + contrast) —
     use exact values from the profile if given; derive from brand identity if not
   - number_of_highlights: 4, highlights_carousel: true, highlights_style: BannerCarousel
   - Footer: 3 columns with real links from the profile
     (e.g. About/Careers/Investors, Offices/Regions, Resources/Support)
   - Social links (derive from profile — e.g. LinkedIn, Twitter/X, YouTube, etc.)
   - navigation_links: topBarLinks (2-3), footerLinks (3-5), socialLinks
   - sidebar_components: pick 1-3 based on the client's profile —
       HR/people-focused        → birthdays + workAnniversaries
       Event-driven org         → upcomingEvents
       Knowledge/document-heavy → recentDocuments

2. CHANNEL ARCHITECTURE (4-5 channels)
   Design channels that reflect how THIS company communicates. No generic names.
   Number each channel (Ch1, Ch2, ...) — this numbering is used in the execution phases.
   Mark 3 channels as PRIMARY (most content, most visible) and the rest SECONDARY.
   For each channel provide ALL of the following — none are optional:
   - Name (specific to this company's voice — not 'News' or 'HR')
   - Purpose (one sentence)
   - Type: public / private / corporate
       public    = visible to all employees
       private   = opt-in, hidden by default
       corporate = restricted (use for leadership, legal, compliance)
   - Color (hex — brand-aligned or function-coded:
       leadership/corporate = brand primary or dark neutral
       legal/compliance     = warn color (red/amber)
       HR/culture           = warm tone
       IT/ops               = cool neutral)
   - is_sticky: True for ALL primary channels (3) — these must be pinned
   - image_url: CMS media path for the channel image. Must already be uploaded.
     Always try to assign an image if the profile provides one.
   - FAQ: 2-3 real Q&A pairs for HR, Legal, Compliance, IT, and policy channels.
     Skip FAQ only on pure broadcast channels (Newsroom, Leadership letters).
     FAQs must answer real questions employees would actually ask.
   - hide_on_homepage_feed: True only for utility/admin channels; False for all others
   - hide_highlights: False for primary channels; True for secondary/utility channels
   - Description: 2-3 sentences of real HTML <p> tags, not placeholder text
   - Tabs: ['Articles'] for broadcast channels; ['Articles', 'Pages'] for reference/policy channels
   - initial_tab: 'Articles'
   - default_role: 'Reader' for public channels; 'None' for private/corporate

3. CONTENT CALENDAR
   PRIMARY channels get 3 content items each.
   SECONDARY channels get 2 content items each.
   Include 1-2 events across the hub where an event type makes sense.
   List content items grouped BY CHANNEL. For every item specify ALL flags:
   - Title (specific and newsy — never 'Welcome to X' or 'Update from the team')
   - Type: announcement / policy / recap / spotlight / guide / milestone / event
   - tags: 3-5 specific tags (e.g. ['q4-2024', 'earnings', 'leadership'])
   - is_main_highlight: pick exactly 3-4 items TOTAL across the whole hub.
     These appear on the homepage hero carousel. Choose high-signal content:
     financial results, product launches, major policy announcements.
     IMPORTANT: is_main_highlight is set on the ContentSpec — it is the ONLY
     correct mechanism for homepage highlights.
   - is_sticky: pin 1-2 items per channel (the most important ones)
   - require_read_confirmation: True for ALL policy, compliance, legal, and security
     items. False for news, spotlights, recaps, and leadership letters.
   - status: 'Published' for everything (this is a live hub, not a draft)
   - notify: True for major announcements and policy items;
             False for evergreen content, guides, and spotlights
   - image_url: CMS media path for the item image. Must already be uploaded.
     Always try to assign an image if the profile provides one.
   - event_start / event_end: ISO 8601 datetimes for Event items; omit for Articles.
   - One-sentence outline of the body content
   CONTENT IDEAS — derive from the profile, never invent:
   • Recent financial results  → quarterly recap with specific numbers
   • Product launches          → launch recap with dates and key details
   • ESG / sustainability      → progress update with targets and milestones
   • Culture / ERGs / diversity→ employee resource group spotlight
   • Compliance / regulation   → policy acknowledgement (require_read_confirmation=True)
   • Office openings / regions → regional milestone article
   • Leadership messages       → CEO letter or exec memo
   • R&D / innovation          → technology or product innovation update
   • IT / security             → policy or best-practice guide (require_read_confirmation=True)
   • Benefits / HR programs    → guide or announcement
   • Town halls / team days / product launch events → Event (event_start + event_end required)

4. HUB-LEVEL POLISH
   - company_name, website_name (from profile)
   - default_language (infer from HQ: US→en-US, Portugal→pt-PT, Germany→de-DE, etc.)
   - supported_languages (infer from operating regions in the profile)
   - login_quick_links: 3-5 real URLs from the profile
     (Careers, Investor Relations, Newsroom, Support, Privacy, etc.)
   - sidebar_navigation: ordered list that reflects this company's priorities.
     Options: apps, collaborators, documents, workgroups, ideation, learning, approvalRequests.
     Lead with what this company values most —
       services firm    → collaborators
       legal            → documents / approvalRequests
       product company  → apps / ideation
       people-first     → learning / collaborators

Present the COMPLETE design in structured sections. Then ask:
'This is a complete design. Shall I build it end-to-end, or would you like to
adjust anything first — channel names, content plan, colors, or hub settings?'
STOP. Do not call any tool until the user approves.

═══════════════════════════════════════════════════════════════════════
PHASE 1 — BUILD SCOPE + CHANNELS IN ONE CALL (after design is approved)
═══════════════════════════════════════════════════════════════════════
a. Call create_scope_with_channels with EVERY design field populated.
   This single tool creates the scope AND all channels atomically.
   Populate EVERY scope-level field — no field left at its default
   when a better value exists from the design:
     name                            — professional scope name matching brand voice
     description                     — short free-text description of the scope's purpose
     welcome_message                 — 1-2 sentences specific to this company, never generic
     channel_creation_by_admins_only — True (lock channel creation to admins)
     number_of_highlights            — 4
     highlights_carousel             — True
     highlights_style                — 'BannerCarousel'
     primary_color                   — brand primary, 6-digit hex e.g. '#0064f0'
     primary_color_contrast          — contrast paired with primary e.g. '#ffffff'
     secondary_color                 — brand secondary, 6-digit hex
     warn_color                      — brand warning/accent, 6-digit hex
     warn_color_contrast             — contrast paired with warn color
     image_url                       — CMS media path for scope hero image; null if not available
     footer_block_links              — 3 columns of real links from the profile
                                       (About/Careers/Investors, Offices/Regions, Resources/Support)
     navigation_links                — topBarLinks (2-3), footerLinks (3-5), socialLinks
                                       (LinkedIn/Twitter/YouTube etc.)
     sidebar_components              — 1-3 widgets from the design phase

   The `channels` parameter is a list of ChannelConfig objects.
   Populate EVERY field for each channel — no field left at its default
   when a better value exists from the design:
     name                   — display name (specific to company voice, not generic)
     description            — full HTML, 2-3 sentences in <p> tags, never empty
     channel_type           — 'public' / 'private' / 'corporate'
     is_sticky              — True for ALL primary channels
     color                  — brand-aligned 6-digit hex from the design
     image_url              — CMS media path if available from the profile or media library; null if not
     frequent_questions     — REQUIRED at creation time. list of FAQItem for HR/Legal/Compliance/IT channels; [] for pure broadcast 
     hide_on_homepage_feed  — True only for utility/admin channels; False for all others
     hide_highlights        — False for primary channels; True for secondary/utility
     default_role           — 'Reader' for public channels; 'None' for private/corporate
     tabs                   — ['Articles'] for broadcast; ['Articles', 'Pages'] for policy/reference
     initial_tab            — 'Articles'

   ChannelConfig does NOT include content — those are seeded in Phase 2 via seed_channel_content

b. The tool returns scope_id AND a complete channel_map with every confirmed
   channel ID — no manual tracking needed.
c. Report the full channel_map to the user:
     Scope: [name] / id: [SCOPE_ID] / route: /scope/[route]
     Ch1: [name] → [id]  [PRIMARY]
     Ch2: [name] → [id]  [PRIMARY]
     ... (mark each PRIMARY or SECONDARY)
   Then proceed immediately to Phase 2 — no further approval needed.

═══════════════════════════════════════════════════════════════════════
PHASE 2 — POPULATE CONTENT (the phase that makes this feel real)
═══════════════════════════════════════════════════════════════════════
For EACH channel, call seed_channel_content(channel_id, items) once.
Pass ALL content items for that channel as a list of ContentSpec objects in a
single call — do NOT call it once per item.
seed_channel_content is idempotent: it checks existing content count first
and only creates what's missing. It is safe to re-run on partial failures.
Each ContentSpec must explicitly set ALL of the following fields:
  title                        — specific and newsy (from the content calendar)
  body                         — rich HTML (see body requirements below)
  status                       — 'Published'
  tags                         — 3-5 specific tags from the design
  is_main_highlight            — True for the 3-4 hub-highlight items.
                                  This is the ONLY correct mechanism for homepage
                                  highlights. Set this on the ContentSpec, not elsewhere.
  is_sticky                    — True for the 1-2 pinned items per channel
  require_read_confirmation    — True for policy, compliance, legal, IT/security.
                                  False for news, spotlights, recaps.
  notify                       — True for major announcements and policies; False for evergreen
  hide_likes                   — False (keep engagement features on)
  hide_comments                — False (keep engagement features on)
  hide_image_in_article_detail — False
  disable_auto_related_articles— False
  image_url                    — CMS media path if available from the profile or media library; null if not
  event_start / event_end      — ISO 8601 datetimes for Event items; omit for plain Articles
  location                     — plain-text venue for Event items; omit otherwise
 
BODY REQUIREMENTS — every content body must meet this bar:
  - HTML with proper structure:
      <h2> for major sections, <h3> for subsections
      <ul>/<ol> for lists, <strong> for key terms
      <blockquote> for executive quotes or policy excerpts
      <em> for emphasis. Multiple paragraphs — 6 to 8 minimum.
  - Reference SPECIFIC facts from the profile:
      revenue figures, product names, office locations, employee counts,
      leadership names, dates, percentages, milestones.
  - Voice must match the company:
      Enterprise/corporate   → measured, authoritative, third-person
      Professional services  → formal, precise, expert
      Tech/startup           → energetic, first-person plural, forward-looking
      Retail/consumer        → warm, direct, employee-first
  - Length: 400-700 words of substantive content. Not filler.
  - End with a closing line or CTA appropriate to the content type.
 
Quality bar example — NOT this:
  <p>Welcome to our company. We are a great place to work.</p>
But this:
  <h2>Q4 FY2024 Results: Revenue Up 6% Year-on-Year</h2>
  <p>The company reported <strong>quarterly revenue of $94.9 billion</strong>,
  up 6% year-over-year, driven by record Services performance.</p>
  <h3>Segment highlights</h3>
  <ul>
    <li>Services reached an all-time high of $24.97 billion</li>
    <li>International revenue accounted for 58% of total revenue</li>
  </ul>
  <blockquote>"We closed the year strong across every product line." — CEO</blockquote>
  <p>Read the full earnings release at investor.[company].com</p>
 
After each seed_channel_content call report:
'Ch[N] [name]: [created_this_call] items created — total: [total_content]/[target] ✓'
Continue channel by channel — do not pause for confirmation between channels.

═══════════════════════════════════════════════════════════════════════
PHASE 3 — VERIFY COMPLETION
═══════════════════════════════════════════════════════════════════════
Call verify_scope(scope_id=SCOPE_ID) and report:
  - Channel count matches the design
  - Brand colors are set (primary, secondary, warn — no nulls)
  - Sticky channels are present (should match primary channel count)
  - Highlights configured (numberOfHighlights=4, carousel=true)
If verify_scope returns warnings about missing colors or channels, fix them
with update_scope before proceeding. Do not skip this phase silently.

═══════════════════════════════════════════════════════════════════════
PHASE 4 — HUB-WIDE POLISH
═══════════════════════════════════════════════════════════════════════
Call update_global_settings with EVERY field from the design:
  company_name, website_name,
  default_language, supported_languages,
  login_quick_links (3-5 real URLs from the profile),
  sidebar_navigation (ordered, reflects company priorities).
This is what ties the hub together — skipping it makes the demo feel unfinished.

═══════════════════════════════════════════════════════════════════════
PHASE 5 — FINAL HANDOVER
═══════════════════════════════════════════════════════════════════════
Deliver a presentation-ready summary:
  Company    : [name]
  Scope      : [name] — id: [SCOPE_ID] — route: /scope/[route]
  Brand      : primary [hex] / secondary [hex] / warn [hex]
  Channels   : [count] total — [list each: name, id, type, sticky=T/F]
  Highlights : [list the 3-4 is_main_highlight items with title]
  Content   : [total count] published across [N] channels
  Policies   : [count] items with require_read_confirmation=True
  Hub polish : company_name=[name], language=[tag], [N] login quick links
Close with: 'The workspace is live. Want to iterate — rename a channel,
add regional content, seed more content, or adjust branding?'

═══════════════════════════════════════════════════════════════════════
HARD RULES — build-specific
═══════════════════════════════════════════════════════════════════════
• NEVER leave description, welcome message, or body null or empty.
• NEVER use placeholder text ('Lorem ipsum', 'TODO', 'TBD').
• NEVER fabricate facts not in the profile — omit or reason plausibly.
• NEVER use generic channel names ('News', 'HR') when the profile
  suggests better ones ('Engineering Pulse', 'People & Belonging').
• NEVER guess a channel id — use the channel_map from create_scope_with_channels.
• Exactly 3-4 items must have is_main_highlight=True across the hub.
• At least one content item per compliance/legal/IT/security channel must have
  require_read_confirmation=True.
• Primary channels must have is_sticky=True in their ChannelConfig.
• Include 1-2 Events total across the hub where context supports it.
• If any tool call fails mid-build: stop, report what succeeded, ask how to proceed.
• The finished workspace must feel built BY this company, FOR this company.
• Follow each phase in order. Do not skip ahead."""

    return [
        PromptMessage(role="assistant", content=TextContent(type="text", text=INSTRUCTIONS)),
        PromptMessage(role="user",      content=TextContent(type="text", text=company_profile)),
    ]


if __name__ == "__main__":
    mcp.run(transport="stdio")