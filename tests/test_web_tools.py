from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent import loop as loop_module
from nanobot.agent.loop import AgentLoop
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.web import WebSearchTool
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse


def _make_loop(tmp_path: Path, *, web_search_enabled: bool) -> AgentLoop:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    return AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        web_search_enabled=web_search_enabled,
    )


def test_agent_loop_registers_web_search_by_default(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, web_search_enabled=True)

    assert loop.tools.has("web_search") is True
    assert loop.subagents.web_search_enabled is True


def test_agent_loop_can_disable_web_search(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, web_search_enabled=False)

    assert loop.tools.has("web_search") is False
    assert loop.subagents.web_search_enabled is False


def test_web_search_description_mentions_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)

    tool = WebSearchTool(api_key=None)

    assert "Currently unavailable" in tool.description


@pytest.mark.asyncio
async def test_web_search_execute_returns_setup_error_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)

    tool = WebSearchTool(api_key=None)
    result = await tool.execute(query="nanobot")

    assert "Brave Search API key not configured" in result


def test_agent_loop_does_not_log_missing_key_when_env_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAVE_API_KEY", "env-key")

    logged: list[str] = []
    monkeypatch.setattr(loop_module.logger, "error", lambda message, *args: logged.append(message))

    _make_loop(tmp_path, web_search_enabled=True)

    assert logged == []


@pytest.mark.asyncio
async def test_subagent_skips_web_search_registration_when_disabled(tmp_path: Path) -> None:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="done", tool_calls=[]))
    mgr = SubagentManager(
        provider=provider,
        workspace=tmp_path,
        bus=bus,
        web_search_enabled=False,
    )
    mgr._announce_result = AsyncMock()

    registered: list[str] = []
    original_register = ToolRegistry.register

    def spy_register(self, tool):
        registered.append(tool.name)
        return original_register(self, tool)

    try:
        ToolRegistry.register = spy_register
        await mgr._run_subagent("sub-1", "do task", "label", {"channel": "test", "chat_id": "c1"})
    finally:
        ToolRegistry.register = original_register

    assert "web_search" not in registered
