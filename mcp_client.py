"""
mcp_client.py
=============
Multi-server MCP client.

Manages one or more MCP server connections simultaneously.
Each server runs as its own stdio subprocess. Tools, resources,
and prompts from all servers are merged into a single pool that
the rest of the app sees — so App, Model, and CLI are unaware
of how many servers are running.

Server registry format (passed to MCPClient.__init__):
    [
        {"name": "mock",  "command": "python", "args": ["mcp_server.py"]},
        {"name": "ms365", "command": "npx",    "args": ["-y", "@softeria/ms-365-mcp-server", "--org-mode"]},
    ]

The "name" field is used as a human-readable label in logs and
is stored on each tool so call_tool() knows which session to use.
"""

import json
import time
from contextlib import AsyncExitStack
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.types import TextContent
from pydantic import AnyUrl


# ── Per-server connection record ─────────────────────────────────────────────

class _ServerConnection:
    """Holds the live session + metadata for one connected MCP server."""

    def __init__(self, name: str):
        self.name         = name
        self.session      : Optional[ClientSession] = None
        self.tools        : list[dict] = []      # OpenAI-format tool schemas
        self.resources    : list[dict] = []      # {uri, name, description, mime_type}
        self.prompts      : list[types.Prompt] = []
        self.instructions : Optional[str] = None  # from server's initialize response


# ── Main client ──────────────────────────────────────────────────────────────

class MCPClient:
    """
    Drop-in replacement for the single-server MCPClient.
    Public interface is identical — tools / resources / call_tool() etc.
    now transparently span multiple servers.
    """

    def __init__(self, servers: list[dict]):
        """
        Args:
            servers: list of server descriptors, each with:
                     - name    (str)  — human label, e.g. "mock" or "ms365"
                     - command (str)  — executable, e.g. "python" or "npx"
                     - args    (list) — arguments list
                     - env     (dict, optional) — extra environment variables
        """
        self._server_configs  = servers
        self._connections     : dict[str, _ServerConnection] = {}
        self._exit_stack      = AsyncExitStack()

        # Merged views (populated by connect())
        self.tools    : list[dict]         = []
        self.resources: list[dict]         = []
        self._prompts : list[types.Prompt] = []

        # tool_name → server_name  (for routing call_tool)
        self._tool_router: dict[str, str] = {}

    # ── Connection ────────────────────────────────────────────────────────────

    async def connect(self) -> dict:
        """
        Launch all server subprocesses, discover tools/resources/prompts,
        and merge them.

        Returns {"tools": [...names], "resources": [...uris]} for display.
        """
        await self._exit_stack.__aenter__()

        for cfg in self._server_configs:
            name    = cfg["name"]
            command = cfg["command"]
            args    = cfg.get("args", [])
            env     = cfg.get("env", None)

            conn = _ServerConnection(name)
            self._connections[name] = conn

            params = StdioServerParameters(command=command, args=args, env=env)

            try:
                read, write = await self._exit_stack.enter_async_context(
                    stdio_client(params)
                )
                conn.session = await self._exit_stack.enter_async_context(
                    ClientSession(read, write)
                )
                init_result = await conn.session.initialize()
                # Capture server-declared instructions (set via FastMCP instructions=).
                # These are the authoritative system prompt for this server.
                conn.instructions = getattr(init_result, "instructions", None) or None
            except Exception as e:
                print(f"[warn] Could not connect to server '{name}': {e}")
                continue

            # Discover tools
            try:
                tools_result = await conn.session.list_tools()
                for t in tools_result.tools:
                    schema = self._to_openai_format(t, server_name=name)
                    conn.tools.append(schema)
                    self._tool_router[t.name] = name
            except Exception as e:
                print(f"[warn] Could not list tools for '{name}': {e}")

            # Discover resources
            try:
                res_result = await conn.session.list_resources()
                for r in res_result.resources:
                    conn.resources.append({
                        "uri"        : str(r.uri),
                        "name"       : r.name,
                        "description": r.description or "",
                        "mime_type"  : r.mimeType or "application/json",
                        "_server"    : name,  # internal routing hint
                    })
            except Exception as e:
                print(f"[warn] Could not list resources for '{name}': {e}")

            # Discover prompts
            try:
                prompts_result = await conn.session.list_prompts()
                conn.prompts = prompts_result.prompts
            except Exception as e:
                print(f"[warn] Could not list prompts for '{name}': {e}")

        # Merge into flat views
        self.tools     = [t for c in self._connections.values() for t in c.tools]
        self.resources = [r for c in self._connections.values() for r in c.resources]
        self._prompts  = [p for c in self._connections.values() for p in c.prompts]

        # Merge server instructions into a single system prompt string.
        # Multiple servers may each declare instructions — concatenate them.
        self.system_prompt: Optional[str] = "\n\n".join(
            c.instructions
            for c in self._connections.values()
            if c.instructions
        ) or None

        return {
            "tools"    : [t["function"]["name"] for t in self.tools],
            "resources": [r["uri"] for r in self.resources],
            "servers"  : list(self._connections.keys()),
        }

    # ── Tool calls ────────────────────────────────────────────────────────────

    async def call_tool(self, tool_name: str, arguments: dict) -> tuple[str, float]:
        """Execute a tool on whichever server owns it. Returns (result_text, latency_ms)."""
        server_name = self._tool_router.get(tool_name)
        if server_name is None:
            return json.dumps({"error": f"Unknown tool '{tool_name}'"}), 0.0

        session = self._connections[server_name].session
        if session is None:
            return json.dumps({"error": f"Server '{server_name}' is not connected"}), 0.0

        start  = time.perf_counter()
        result = await session.call_tool(tool_name, arguments)
        latency_ms = (time.perf_counter() - start) * 1000

        result_text = "".join(
            block.text for block in result.content
            if isinstance(block, TextContent)
        )
        return result_text, latency_ms

    # ── Resource access ───────────────────────────────────────────────────────

    async def list_resources(self) -> list[dict]:
        return self.resources

    async def read_resource(self, uri: str) -> Any:
        """
        Read a resource by URI, routing to the correct server.
        Falls back to trying all servers if the URI isn't in the known list.
        """
        # Find the server that owns this URI
        server_name = None
        for r in self.resources:
            if r["uri"] == uri:
                server_name = r.get("_server")
                break

        # If not found by exact match (e.g. templated URI like company://pages/page_001)
        # try to match by URI prefix / scheme
        if server_name is None:
            scheme = uri.split("://")[0] if "://" in uri else ""
            for r in self.resources:
                if r["uri"].split("://")[0] == scheme:
                    server_name = r.get("_server")
                    break

        # Last resort: try all servers
        candidate_names = (
            [server_name] if server_name
            else list(self._connections.keys())
        )

        last_exc: Optional[Exception] = None
        for name in candidate_names:
            session = self._connections[name].session
            if session is None:
                continue
            try:
                result   = await session.read_resource(AnyUrl(uri))
                resource = result.contents[0]
                if isinstance(resource, types.TextResourceContents):
                    if resource.mimeType == "application/json":
                        return json.loads(resource.text)
                    return resource.text
                return ""
            except Exception as e:
                last_exc = e

        raise last_exc or RuntimeError(f"Could not read resource '{uri}'")

    # ── Prompts ───────────────────────────────────────────────────────────────

    async def list_prompts(self) -> list[types.Prompt]:
        return self._prompts

    async def get_prompt(self, prompt_name: str, args: dict[str, str]) -> list:
        # Find which server has this prompt
        for name, conn in self._connections.items():
            if conn.session is None:
                continue
            if any(p.name == prompt_name for p in conn.prompts):
                result = await conn.session.get_prompt(prompt_name, args)
                return result.messages
        raise ValueError(f"Prompt '{prompt_name}' not found on any connected server")

    # ── Cleanup ───────────────────────────────────────────────────────────────

    async def disconnect(self):
        await self._exit_stack.aclose()
        for conn in self._connections.values():
            conn.session = None

    # ── Context manager ───────────────────────────────────────────────────────

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _to_openai_format(self, tool, server_name: str = "") -> dict:
        return {
            "type": "function",
            "function": {
                "name"       : tool.name,
                "description": tool.description or tool.name,
                "parameters" : tool.inputSchema or {"type": "object", "properties": {}},
            },
            "_server": server_name,  # internal metadata, not sent to OpenAI
        }