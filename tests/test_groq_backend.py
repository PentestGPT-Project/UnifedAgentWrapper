"""Tests for the Groq backend (no network; mocks OpenAI client)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from unified_agent.agent import BACKEND_NAMES, _make_backend
from unified_agent.backends.groq import GroqBackend, MCPClient
from unified_agent.events import AssistantText, ToolCall, ToolResult, TurnCompleted
from unified_agent.types import RunOptions


def test_groq_in_backend_names():
    assert "groq" in BACKEND_NAMES


def test_make_backend_groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    backend = _make_backend("groq")
    assert isinstance(backend, GroqBackend)
    assert backend.name == "groq"


def test_groq_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(Exception):
        GroqBackend()


@pytest.mark.asyncio
async def test_stream_plain_text(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    backend = GroqBackend()

    fake_resp = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="hello", reasoning=None, tool_calls=None),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    backend.client = MagicMock()
    backend.client.chat = MagicMock()
    backend.client.chat.completions = MagicMock()
    backend.client.chat.completions.create = AsyncMock(return_value=fake_resp)

    opts = RunOptions(workspace=tmp_path)
    events = [e async for e in backend.stream("hi", opts)]
    kinds = [type(e).__name__ for e in events]
    assert "AssistantText" in kinds
    assert isinstance(events[-1], TurnCompleted)
    assert events[-1].success is True


@pytest.mark.asyncio
async def test_stream_structured_output(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    backend = GroqBackend()

    tc = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="StructuredOutput", arguments='{"answer": 42}'),
    )
    fake_resp = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=None, reasoning=None, tool_calls=[tc]),
            finish_reason="tool_calls",
        )],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    backend.client = MagicMock()
    backend.client.chat = MagicMock()
    backend.client.chat.completions = MagicMock()
    backend.client.chat.completions.create = AsyncMock(return_value=fake_resp)

    opts = RunOptions(workspace=tmp_path, output_schema={"type": "object"})
    events = [e async for e in backend.stream("hi", opts)]
    assert any(isinstance(e, ToolCall) and e.name == "StructuredOutput" for e in events)
    assert any(isinstance(e, ToolResult) for e in events)
    final = events[-1]
    assert isinstance(final, TurnCompleted)
    assert final.structured_output == {"answer": 42}
