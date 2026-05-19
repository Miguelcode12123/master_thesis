"""
cli.py
======
Terminal UI using prompt_toolkit.

Multi-server update:
  - load_resources() now iterates over ALL connected servers and
    populates @mention completions from each one's resources.
  - MS365 resources (mailbox folders, OneDrive drives, etc.) are
    surfaced alongside mock-server resources under their own URI prefix.
  - Everything else (autocomplete, /command, ghost-text) is unchanged.
"""

import json

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document

from app import App


# ── Autosuggester ─────────────────────────────────────────────────────────────

class PromptAutoSuggest(AutoSuggest):
    def __init__(self):
        self.prompt_args: dict[str, str] = {}

    def update_prompts(self, prompts: list):
        self.prompt_args = {}
        for p in prompts:
            if p.arguments:
                self.prompt_args[p.name] = p.arguments[0].name

    def get_suggestion(self, buffer: Buffer, document: Document) -> Suggestion | None:
        text = document.text
        if not text.startswith("/"):
            return None
        parts = text[1:].split()
        if len(parts) == 1 and text.endswith(" "):
            arg = self.prompt_args.get(parts[0])
            return Suggestion(arg) if arg else None
        return None


# ── Unified completer ─────────────────────────────────────────────────────────

class UnifiedCompleter(Completer):

    def __init__(self):
        self.collections: list[str] = []
        self.children   : dict[str, list[str]] = {}
        self.prompts    : list = []

    def update_resources(self, uris: list[str], concrete_ids: list[str] | None = None):
        """
        uris          — all resource URIs across all servers (scheme stripped)
        concrete_ids  — pre-fetched child IDs like "pages/page_001", "employees/emp_001"
        """
        self.collections = [uri.split("://", 1)[-1] for uri in uris if "{" not in uri]
        self.children    = {}
        for full_id in (concrete_ids or []):
            parent = full_id.split("/")[0]
            self.children.setdefault(parent, []).append(full_id)

    def update_prompts(self, prompts: list):
        self.prompts = prompts

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # ── / commands ────────────────────────────────────────
        if text.startswith("/"):
            parts = text[1:].split()
            if not text.endswith(" ") and len(parts) <= 1:
                prefix = parts[0] if parts else ""
                for p in self.prompts:
                    if p.name.startswith(prefix):
                        yield Completion(
                            p.name,
                            start_position=-len(prefix),
                            display=f"/{p.name}",
                            display_meta=p.description or "",
                        )
            return

        # ── @ mentions ─────────────────────────────────────────
        if "@" not in text:
            return

        prefix = text[text.rfind("@") + 1:]

        if "/" in prefix:
            parent = prefix.split("/")[0]
            for full_id in self.children.get(parent, []):
                if full_id.lower().startswith(prefix.lower()):
                    yield Completion(
                        full_id,
                        start_position=-len(prefix),
                        display=full_id.split("/", 1)[-1],
                        display_meta=parent,
                    )
        else:
            for name in self.collections:
                has_children = name in self.children
                yield Completion(
                    name + ("/" if has_children else ""),
                    start_position=-len(prefix),
                    display=name,
                    display_meta="↳ select id" if has_children else "resource",
                )


# ── CliApp ────────────────────────────────────────────────────────────────────

class CliApp:

    def __init__(self, app: App):
        self.app           = app
        self.completer     = UnifiedCompleter()
        self.autosuggester = PromptAutoSuggest()
        self._prompts: list = []

        kb = KeyBindings()

        @kb.add("@")
        def _(event):
            buf = event.app.current_buffer
            buf.insert_text("@")
            if buf.document.is_cursor_at_the_end:
                buf.start_completion(select_first=False)

        @kb.add("/")
        def _(event):
            buf  = event.app.current_buffer
            text = buf.document.text_before_cursor
            buf.insert_text("/")
            if not text.strip():
                buf.start_completion(select_first=False)
            elif "@" in text and text.endswith("/"):
                buf.start_completion(select_first=False)

        self.session = PromptSession(
            completer          =self.completer,
            auto_suggest       =self.autosuggester,
            history            =InMemoryHistory(),
            key_bindings       =kb,
            complete_while_typing=True,
            complete_in_thread =True,
            style=Style.from_dict({
                "prompt"                             : "#888888",
                "completion-menu.completion"         : "bg:#1e1e1e #cccccc",
                "completion-menu.completion.current" : "bg:#3a3a3a #ffffff bold",
                "completion-menu.meta.completion"    : "bg:#1e1e1e #555555 italic",
            }),
        )

    async def load_resources(self, discovered: dict):
        """
        Load @mention completions from ALL connected servers.
        For each server's resources, try to fetch concrete child IDs
        where the resource is a known collection type.
        """
        uris         = discovered["resources"]
        concrete_ids : list[str] = []

        # # ── Mock server children ──────────────────────────────
        # try:
        #     pages = await self.app.client.read_resource("company://pages")
        #     if isinstance(pages, list):
        #         concrete_ids += [f"pages/{p['id']}" for p in pages]
        # except Exception:
        #     pass

        # try:
        #     employees = await self.app.client.read_resource("company://employees")
        #     if isinstance(employees, list):
        #         concrete_ids += [f"employees/{e['id']}" for e in employees]
        # except Exception:
        #     pass

        # ── MS365: no templated child resources to pre-load ───
        # The MS365 server exposes data via tools, not URI templates,
        # so @mention for MS365 works at the collection level only
        # (e.g. if the server registers any static resources).

        # self.completer.update_resources(uris, concrete_ids)

         # ── Diggspace children ────────────────────────────────
        # The diggspace://scopes resource returns an envelope
        # {"status_code": 200, "data": [ {id, name, ...}, ... ]}
        # so we unwrap and collect scope IDs for @scopes/<id> completion.
        try:
            scopes_raw = await self.app.client.read_resource("diggspace://scopes")
            scopes_list = self._unwrap_diggspace_payload(scopes_raw)
            if isinstance(scopes_list, list):
                concrete_ids += [
                    f"scopes/{s['id']}" for s in scopes_list
                    if isinstance(s, dict) and "id" in s
                ]
        except Exception:
            pass

    @staticmethod
    def _unwrap_diggspace_payload(raw):
        """
        Diggspace resources wrap responses as
            {"status_code": 200, "data": <payload>}
        Parse if string, then unwrap to the inner payload.
        """
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return raw
        if isinstance(raw, dict) and "data" in raw:
            return raw["data"]
        return raw

    async def load_prompts(self):
        try:
            self._prompts = await self.app.client.list_prompts()
            self.completer.update_prompts(self._prompts)
            self.autosuggester.update_prompts(self._prompts)
        except Exception as e:
            print(f"[warn] could not load prompts: {e}")

    def _resolve_prompt_command(self, text: str) -> tuple[str, str] | None:
        if not text.startswith("/"):
            return None
        parts = text[1:].split(maxsplit=1)
        if not parts:
            return None
        return (parts[0], parts[1].strip() if len(parts) > 1 else "")

    async def run(self):
        print("  Type your question, use @mention to inject resources, or /command to run a prompt.\n")

        while True:
            try:
                user_input = await self.session.prompt_async("You: ")
            except (EOFError, KeyboardInterrupt):
                break

            user_input = user_input.strip()
            if not user_input or user_input.lower() == "quit":
                break

            # ── Prompt command path ───────────────────────────
            resolved = self._resolve_prompt_command(user_input)
            if resolved:
                prompt_name, arg_value = resolved
                prompt_obj = next((p for p in self._prompts if p.name == prompt_name), None)

                if prompt_obj is None:
                    print(f"Unknown command /{prompt_name}. Available: {[p.name for p in self._prompts]}\n")
                    continue

                if prompt_obj.arguments and not arg_value:
                    arg_name = prompt_obj.arguments[0].name
                    arg_desc = prompt_obj.arguments[0].description or arg_name
                    try:
                        arg_value = (await self.session.prompt_async(f"  {arg_desc}: ")).strip()
                    except (EOFError, KeyboardInterrupt):
                        break
                    if not arg_value:
                        print("Cancelled.\n")
                        continue

                args = {}
                if prompt_obj.arguments:
                    args[prompt_obj.arguments[0].name] = arg_value

                try:
                    messages = await self.app.client.get_prompt(prompt_name, args)
                    await self.app.inject_prompt_as_task(messages)
                except Exception as e:
                    print(f"[error] prompt failed: {e}\n")
                continue

            # ── Normal query path ─────────────────────────────
            await self.app.handle_query(user_input)