"""Safe symbolic-expression parsing and evaluation.

LLM-proposed formulas are untrusted text. We never call ``eval``/``exec``.
Instead we parse with :mod:`ast`, validate against a strict whitelist, and
evaluate the tree ourselves with NumPy.

Grammar (informal):
    expr  := number | variable | constant | expr (+|-|*|/|**) expr
           | -expr | func(expr, ...)
Free constants are named ``c0, c1, ...`` and are fitted to data later.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np

MAX_NODES = 40
MAX_DEPTH = 12
_CONST_RE = re.compile(r"^c(\d+)$")

_FUNCS: Dict[str, Callable[..., np.ndarray]] = {
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "tanh": np.tanh,
    "abs": np.abs,
}
_NAMED_CONSTANTS = {"pi": math.pi, "e": math.e}


class ExprError(ValueError):
    """Raised for any expression that is malformed or not allowed."""


@dataclass(frozen=True)
class Expr:
    """A validated expression, ready to be evaluated with arrays."""

    source: str
    tree: ast.AST
    variables: tuple  # variable names actually used
    n_constants: int  # number of free constants c0..c{n-1}
    n_nodes: int  # complexity measure

    def __call__(self, env: Dict[str, np.ndarray], consts: Sequence[float]) -> np.ndarray:
        if len(consts) < self.n_constants:
            raise ExprError("not enough constants supplied")
        with np.errstate(all="ignore"):
            out = _eval(self.tree.body, env, consts)  # type: ignore[attr-defined]
        out = np.asarray(out, dtype=float)
        if out.ndim == 0:
            any_var = next(iter(env.values()))
            out = np.full(np.shape(any_var), float(out))
        return out


def _normalise(source: str) -> str:
    s = source.strip()
    # Tolerate the notation LLMs commonly emit.
    s = s.replace("^", "**").replace("×", "*").replace("−", "-")
    if "=" in s:  # "y = c0*x" -> keep right-hand side only
        s = s.split("=")[-1].strip()
    return s


def parse(source: str, variables: Sequence[str]) -> Expr:
    """Parse and validate ``source``; raise :class:`ExprError` if unacceptable."""
    if not isinstance(source, str) or not source.strip():
        raise ExprError("empty expression")
    if len(source) > 300:
        raise ExprError("expression too long")
    text = _normalise(source)
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ExprError(f"syntax error: {exc.msg}") from exc
    except (ValueError, RecursionError, MemoryError) as exc:
        # Older Pythons raise ValueError for null bytes; very deep nesting raises
        # RecursionError. Untrusted text must never be able to crash the agent.
        raise ExprError(f"cannot parse expression ({type(exc).__name__})") from exc

    allowed_vars = set(variables)
    used_vars: set = set()
    const_ids: set = set()
    count = 0

    def walk(node: ast.AST, depth: int) -> None:
        nonlocal count
        count += 1
        if count > MAX_NODES:
            raise ExprError(f"expression exceeds {MAX_NODES} nodes")
        if depth > MAX_DEPTH:
            raise ExprError("expression nested too deeply")

        if isinstance(node, ast.Expression):
            walk(node.body, depth + 1)
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
                raise ExprError(f"operator {type(node.op).__name__} not allowed")
            walk(node.left, depth + 1)
            walk(node.right, depth + 1)
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd)):
                raise ExprError(f"operator {type(node.op).__name__} not allowed")
            walk(node.operand, depth + 1)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
                raise ExprError("only whitelisted functions may be called: " + ", ".join(sorted(_FUNCS)))
            if node.keywords or len(node.args) != 1:
                raise ExprError(f"{node.func.id} takes exactly one positional argument")
            walk(node.args[0], depth + 1)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ExprError("only numeric literals are allowed")
            if not math.isfinite(node.value) or abs(node.value) > 1e9:
                raise ExprError("numeric literal out of range")
        elif isinstance(node, ast.Name):
            m = _CONST_RE.match(node.id)
            if m:
                const_ids.add(int(m.group(1)))
            elif node.id in allowed_vars:
                used_vars.add(node.id)
            elif node.id in _NAMED_CONSTANTS:
                pass
            else:
                raise ExprError(
                    f"unknown name '{node.id}' (variables: {sorted(allowed_vars)}, constants: c0, c1, ...)"
                )
        else:
            raise ExprError(f"{type(node).__name__} is not allowed")

    walk(tree, 0)

    n_consts = (max(const_ids) + 1) if const_ids else 0
    if n_consts > 8:
        raise ExprError("at most 8 free constants (c0..c7) are allowed")
    if const_ids != set(range(n_consts)):
        raise ExprError("constants must be numbered contiguously from c0")
    return Expr(text, tree, tuple(sorted(used_vars)), n_consts, count)


def _eval(node: ast.AST, env: Dict[str, np.ndarray], consts: Sequence[float]):
    if isinstance(node, ast.BinOp):
        a = _eval(node.left, env, consts)
        b = _eval(node.right, env, consts)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div):
            return a / b
        # Pow: clip exponents so adversarial input cannot blow up memory/time.
        return np.power(a, np.clip(b, -50.0, 50.0))
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, env, consts)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.Call):
        return _FUNCS[node.func.id](_eval(node.args[0], env, consts))  # type: ignore[attr-defined]
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        m = _CONST_RE.match(node.id)
        if m:
            return float(consts[int(m.group(1))])
        if node.id in env:
            return env[node.id]
        return _NAMED_CONSTANTS[node.id]
    raise ExprError(f"cannot evaluate {type(node).__name__}")  # unreachable after validation


def canonical_key(expr: Expr) -> str:
    """A whitespace-insensitive key used to deduplicate candidates."""
    return ast.dump(expr.tree.body)  # type: ignore[attr-defined]
