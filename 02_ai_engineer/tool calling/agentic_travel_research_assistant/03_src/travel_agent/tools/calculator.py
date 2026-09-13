from __future__ import annotations

import ast
import operator
from pydantic import BaseModel, ConfigDict, Field
from .base import BaseTool

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval_node(node.left), _safe_eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_safe_eval_node(node.operand))
    raise ValueError("Only numeric arithmetic with +, -, *, /, %, and ** is allowed")


class CalculatorInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expression: str = Field(min_length=1, max_length=300, description="Numeric arithmetic expression")


class CalculatorTool(BaseTool[CalculatorInput]):
    name = "calculate"
    description = "Safely evaluate numeric arithmetic. Use it for budget totals and non-trivial calculations."
    input_model = CalculatorInput

    def execute(self, arguments: CalculatorInput) -> dict:
        tree = ast.parse(arguments.expression, mode="eval")
        result = _safe_eval_node(tree)
        return {"expression": arguments.expression, "result": round(result, 4)}
