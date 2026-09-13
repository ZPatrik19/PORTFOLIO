from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

InputT = TypeVar("InputT", bound=BaseModel)


def _strictify_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make Pydantic JSON Schema friendlier to strict function calling.

    OpenAI strict tool schemas expect object properties to reject unknown fields and
    every declared property to be present. Nullable fields remain nullable, but the
    model should explicitly send null when it does not need them.
    """
    schema = deepcopy(schema)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("default", None)
            node.pop("title", None)
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                props = node.get("properties", {})
                if props:
                    node["required"] = list(props.keys())
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema)
    return schema


class BaseTool(ABC, Generic[InputT]):
    name: str
    description: str
    input_model: type[InputT]

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": _strictify_schema(self.input_model.model_json_schema()),
            "strict": True,
        }

    def validate(self, arguments: dict[str, Any]) -> InputT:
        return self.input_model.model_validate(arguments)

    @abstractmethod
    def execute(self, arguments: InputT) -> dict[str, Any]:
        raise NotImplementedError
