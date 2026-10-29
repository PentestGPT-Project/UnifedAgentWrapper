"""Groq backend: OpenAI-compatible API with MCP tool execution loop."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import uuid
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from ..events import (
    AgentEvent,
    AssistantText,
    Reasoning,
    SessionStarted,
    ToolCall,
    ToolResult,
    TurnCompleted,
)
from ..types import (
    AgentAuthError,
    AgentRunError,
    RunOptions,
    UnifiedUsage,
)

TOOL_TIMEOUT = 300
MAX_TURNS_DEFAULT = 20


class MCPClient:
    def __init__(self, spec):
        self.spec = spec
        self.proc = None
        self._req_id = 0
        self._tools = []

    async def __aenter__(self):
        env = dict(os.environ)
        env.update(self.spec.env)
        self.proc = await asyncio.create_subprocess_exec(
            *self.spec.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        await self._call("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "groq-backend", "version": "0.1.0"},
        })
        await self._notify("notifications/initialized", {})
        result = await self._call("tools/list", {})
        self._tools = result.get("tools", [])
        return self

    async def __aexit__(self, *exc):
        if self.proc and self.proc.returncode is None:
            try:
                self.proc.stdin.close()
                await self.proc.wait()
            except Exception:
                self.proc.kill()

    async def _send(self, msg: dict):
        if not self.proc or not self.proc.stdin:
            raise RuntimeError("MCP not started")
        self.proc.stdin.write((json.dumps(msg) + "\n").encode())
        await self.proc.stdin.drain()

    async def _recv(self) -> dict:
        if not self.proc or not self.proc.stdout:
            raise RuntimeError("MCP not started")
        line = await asyncio.wait_for(self.proc.stdout.readline(), timeout=TOOL_TIMEOUT)
        if not line:
            raise RuntimeError("MCP stdout closed")
        return json.loads(line)

    async def _call(self, method: str, params: dict) -> dict:
        self._req_id += 1
        await self._send({"jsonrpc": "2.0", "id": self._req_id, "method": method, "params": params})
        while True:
            resp = await self._recv()
            if resp.get("id") == self._req_id:
                if "error" in resp:
                    raise RuntimeError(f"MCP error: {resp['error']}")
                return resp.get("result", {})

    async def _notify(self, method: str, params: dict):
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def openai_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("inputSchema", {"type": "object", "properties": {}}),
                },
            }
            for t in self._tools
        ]

    async def call_tool(self, name: str, args: dict) -> str:
        result = await self._call("tools/call", {"name": name, "arguments": args})
        parts = []
        for c in result.get("content", []):
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(c.get("text", ""))
            else:
                parts.append(str(c))
        return "\n".join(parts) if parts else "(no output)"


class GroqBackend:
    name = "groq"

    def __init__(self, model: str = "openai/gpt-oss-120b", **kw):
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise AgentAuthError("GROQ_API_KEY not set")
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        )
        self.model = model

    @staticmethod
    def _usage(resp) -> UnifiedUsage:
        u = resp.usage
        if u is None:
            return UnifiedUsage()
        return UnifiedUsage(
            input_tokens=u.prompt_tokens or 0,
            cached_input_tokens=0,
            output_tokens=u.completion_tokens or 0,
        )

    async def stream(self, prompt: str, opts: RunOptions) -> AsyncIterator[AgentEvent]:
        yield SessionStarted(session_id=str(uuid.uuid4()))
        mcp = None
        try:
            tools = []
            if opts.tool_server is not None:
                mcp = MCPClient(opts.tool_server)
                await mcp.__aenter__()
                tools = mcp.openai_tools()

            if opts.output_schema:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": "StructuredOutput",
                        "description": "Return the final structured JSON decision.",
                        "parameters": opts.output_schema,
                    },
                })

            messages: list[dict] = [{"role": "user", "content": prompt}]
            max_turns = opts.max_turns or MAX_TURNS_DEFAULT

            for turn in range(max_turns):
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools or None,
                    tool_choice="auto" if tools else None,
                    temperature=0.2,
                )
                msg = resp.choices[0].message
                finish = resp.choices[0].finish_reason

                if getattr(msg, "reasoning", None):
                    yield Reasoning(text=msg.reasoning)
                if msg.content:
                    yield AssistantText(text=msg.content)

                if not msg.tool_calls:
                    yield TurnCompleted(
                        success=True,
                        final_text=msg.content or "",
                        structured_output=None,
                        usage=self._usage(resp),
                        stop_reason=finish,
                        error=None,
                    )
                    return

                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name,
                                      "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls
                    ],
                })

                for tc in msg.tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}

                    yield ToolCall(name=name, input=args, call_id=tc.id)

                    if name == "StructuredOutput":
                        yield ToolResult(call_id=tc.id, output="accepted", is_error=False)
                        yield TurnCompleted(
                            success=True,
                            final_text=msg.content or "",
                            structured_output=args,
                            usage=self._usage(resp),
                            stop_reason="structured_output",
                            error=None,
                        )
                        return

                    if mcp is None:
                        err = f"No tool server available for {name!r}"
                        yield ToolResult(call_id=tc.id, output=err, is_error=True)
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": err})
                        continue

                    try:
                        result = await mcp.call_tool(name, args)
                        yield ToolResult(call_id=tc.id, output=result, is_error=False)
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                    except Exception as e:
                        err = f"{type(e).__name__}: {e}"
                        yield ToolResult(call_id=tc.id, output=err, is_error=True)
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": err})

            yield TurnCompleted(
                success=False,
                final_text="",
                structured_output=None,
                usage=UnifiedUsage(),
                stop_reason="max_turns",
                error=f"reached max_turns ({max_turns})",
            )
        except Exception as e:
            yield TurnCompleted(
                success=False,
                final_text="",
                structured_output=None,
                usage=UnifiedUsage(),
                stop_reason="error",
                error=f"{type(e).__name__}: {e}",
            )
        finally:
            if mcp:
                await mcp.__aexit__(None, None, None)
