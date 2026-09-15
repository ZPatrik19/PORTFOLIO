"""Unit tests for the allowlisted tool registry."""

from __future__ import annotations

import pytest
from tkip.tools import ToolRegistry, ToolSpec

pytestmark = pytest.mark.unit


def _registry_with_increment_tool() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "increment",
            "Increment an integer by one.",
            {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
            lambda value: value + 1,
        )
    )
    return registry


def test_registered_tool_executes_with_valid_arguments() -> None:
    registry = _registry_with_increment_tool()

    result = registry.execute("increment", {"value": 2})

    assert result == 3


def test_unregistered_tool_name_is_rejected() -> None:
    registry = _registry_with_increment_tool()

    with pytest.raises(ValueError):
        registry.execute("delete_everything", {})


def test_invalid_tool_arguments_are_rejected_before_execution() -> None:
    registry = _registry_with_increment_tool()

    with pytest.raises(ValueError):
        registry.execute("increment", {"wrong_field": 2})
