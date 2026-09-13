from __future__ import annotations

import time
from typing import Any

from pydantic import ValidationError

from .attractions import AttractionSearchTool
from .base import BaseTool
from .calculator import CalculatorTool
from .currency import CurrencyTool
from .hotels import HotelSearchTool
from .location import LocationTool
from .restaurants import RestaurantSearchTool
from .transport import TransportTool
from .weather import WeatherTool


class ToolRegistry:
    """Single source of truth for tool discovery, schema generation, validation and execution."""

    def __init__(self, tools: list[BaseTool] | None = None) -> None:
        tools = tools or [
            LocationTool(), WeatherTool(), CurrencyTool(), HotelSearchTool(),
            AttractionSearchTool(), RestaurantSearchTool(), TransportTool(), CalculatorTool(),
        ]
        self._tools = {tool.name: tool for tool in tools}

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.openai_schema() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], float, bool, str | None]:
        started = time.perf_counter()
        try:
            if name not in self._tools:
                raise KeyError(f"Unknown tool: {name}")
            validated = self._tools[name].validate(arguments)
            output = self._tools[name].execute(validated)
            if "error" in output:
                raise ValueError(output["error"])
            success, error = True, None
        except (ValidationError, ValueError, KeyError, SyntaxError, ZeroDivisionError, FileNotFoundError) as exc:
            output = {"error": str(exc)}
            success, error = False, str(exc)
        return output, (time.perf_counter() - started) * 1000, success, error
