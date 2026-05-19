"""
main.py
=======
Entry point. Loads config, defines the server registry, wires App + CliApp.

SERVER REGISTRY
───────────────
Add or remove servers by editing SERVER_CONFIGS below.
Each entry needs:
    name    — label used in logs and tool-routing display
    command — the executable (python, npx, node, …)
    args    — list of CLI arguments

PROVIDER SELECTION
──────────────────
Set MODEL_PROVIDER in your .env to switch between models:
    MODEL_PROVIDER=openai      → uses AZURE_OPENAI_* vars (default)
    MODEL_PROVIDER=anthropic   → uses ANTHROPIC_FOUNDRY_* vars

MS365 AUTHENTICATION
─────────────────────
The MS365 server handles its own auth via device-code flow.
On first run, call `npx @softeria/ms-365-mcp-server --login` in your
terminal BEFORE starting this app, then the cached token is reused.

Alternatively, set MS365_MCP_OAUTH_TOKEN in your .env to skip interactive auth.
"""

import asyncio
import os

from dotenv import load_dotenv
from app import App
from cli import CliApp

load_dotenv()

# ── Model provider config ─────────────────────────────────────────────────────

provider = os.getenv("MODEL_PROVIDER", "openai").lower()

if provider == "anthropic":
    deployment = os.getenv("ANTHROPIC_FOUNDRY_DEPLOYMENT", "")
    endpoint   = os.getenv("ANTHROPIC_FOUNDRY_ENDPOINT", "")
    api_key    = os.getenv("ANTHROPIC_FOUNDRY_API_KEY", "")
    assert deployment, "ANTHROPIC_FOUNDRY_DEPLOYMENT is not set"
    assert endpoint,   "ANTHROPIC_FOUNDRY_ENDPOINT is not set"
    assert api_key,    "ANTHROPIC_FOUNDRY_API_KEY is not set"
else:
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
    endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key    = os.getenv("AZURE_OPENAI_API_KEY", "")
    assert deployment, "AZURE_OPENAI_DEPLOYMENT is not set"
    assert endpoint,   "AZURE_OPENAI_ENDPOINT is not set"
    assert api_key,    "AZURE_OPENAI_API_KEY is not set"

# ── MCP Server registry ───────────────────────────────────────────────────────

SERVER_CONFIGS = [
    # {
    #     "name"   : "mock",
    #     "command": "python",
    #     "args"   : ["mcp_server.py"],
    # },
    # {
    #     "name"   : "ms365",
    #     "command": "npx",
    #     "args"   : [
    #         "-y",
    #         "@softeria/ms-365-mcp-server",
    #         "--org-mode",
    #         "--enabled-tools", "list-mail-messages|get-mail-message"
    #     ],
    #     "env": {
    #         **dict(os.environ),
    #         "MS365_MCP_TOKEN_CACHE_PATH"     : os.getenv("MS365_MCP_TOKEN_CACHE_PATH", ""),
    #         "MS365_MCP_SELECTED_ACCOUNT_PATH": os.getenv("MS365_MCP_SELECTED_ACCOUNT_PATH", ""),
    #     },
    # },
    {
        "name"   : "diggspace",
        "command": "python",
        "args"   : ["diggspace_mcp_server.py"],
        "env": {
            **dict(os.environ),
            "DIGGSPACE_API_BASE_URL": os.getenv("DIGGSPACE_API_BASE_URL", ""),
            "DIGGSPACE_BEARER_TOKEN": os.getenv("DIGGSPACE_BEARER_TOKEN", ""),
            "DIGGSPACE_VERIFY_SSL"  : os.getenv("DIGGSPACE_VERIFY_SSL", "false"),
        },
    },
]


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    app = App(
        endpoint  =endpoint,
        api_key   =api_key,
        deployment=deployment,
        servers   =SERVER_CONFIGS,
    )

    discovered = await app.start()

    # ── Startup banner ────────────────────────────────────────────────────────
    connected_servers = discovered.get("servers", [])
    print(f"\nModel provider : {provider}")
    print(f"Model          : {deployment}")
    print(f"Connected servers: {', '.join(connected_servers)}")
    print(f"Total tools available: {len(discovered['tools'])}\n")

    for server_name, conn in app.client._connections.items():
        if conn.tools:
            print(f"[{server_name}] tools ({len(conn.tools)}):")
            for t in conn.tools:
                print(f"  • {t['function']['name']}")
            print()

    if discovered["resources"]:
        print("Resources (type @ to browse):")
        for uri in discovered["resources"]:
            print(f"  @ {uri.split('://', 1)[-1]}")
        print()

    print("Type 'quit' to exit.\n")

    # ── CLI setup ─────────────────────────────────────────────────────────────
    cli = CliApp(app)
    await cli.load_resources(discovered)
    await cli.load_prompts()

    if cli._prompts:
        print("Prompts (type / to run):")
        for p in cli._prompts:
            print(f"  /{p.name:<26} {p.description or ''}")
        print()

    # ── Start conversation log ────────────────────────────────────────────────
    app.logger.start_conversation(model=app.model.deployment)

    try:
        await cli.run()
    finally:
        app.logger.end_conversation()   
        await app.stop()


if __name__ == "__main__":
    asyncio.run(main())