"""Constant fitting and candidate evaluation.

Given a validated expression with free constants c0..ck, find constants that
minimise squared error on the training set (multi-start least squares), then
score on the validation set. Test/extrapolation scoring is done separately by
the benchmark so that solvers cannot see those numbers.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

from .expr import Expr, ExprError, parse
from .tasks import Arrays, r2_score

COMPLEXITY_PENALTY = 1e-4  # score = val_r2 - penalty * nodes  (mild Occam's razor)
_BIG = 1e6


@dataclass
class Candidate:
    """The outcome of fitting and scoring one proposed expression."""

    source: str
    ok: bool
    error: str = ""
    consts: List[float] = field(default_factory=list)
    train_r2: float = float("-inf")
    val_r2: float = float("-inf")
    nodes: int = 0
    expr: Optional[Expr] = field(default=None, repr=False)

    @property
    def score(self) -> float:
        if not self.ok or not np.isfinite(self.val_r2):
            return float("-inf")
        return self.val_r2 - COMPLEXITY_PENALTY * self.nodes

    def predict(self, env: Arrays) -> Optional[np.ndarray]:
        if not self.ok or self.expr is None:
            return None
        return self.expr(env, self.consts)

    def short(self) -> str:
        if not self.ok:
            return f"{self.source}  ->  INVALID: {self.error}"
        cs = ", ".join(f"c{i}={c:.4g}" for i, c in enumerate(self.consts))
        return f"{self.expr.source if self.expr else self.source}  [{cs}]  train R2={self.train_r2:.4f}, val R2={self.val_r2:.4f}"


def _starts(n_consts: int, n_starts: int, rng: np.random.Generator, y_scale: float) -> List[np.ndarray]:
    starts = [np.ones(n_consts), np.full(n_consts, y_scale)]
    while len(starts) < n_starts:
        if rng.random() < 0.5:  # log-uniform magnitude, random sign
            s = rng.choice([-1.0, 1.0], n_consts) * 10.0 ** rng.uniform(-2, 2, n_consts)
        else:  # plain uniform: helps frequencies / phases
            s = rng.uniform(-6, 6, n_consts)
        starts.append(s)
    return starts


def fit_constants(
    expr: Expr, env: Arrays, y: np.ndarray, rng: np.random.Generator, n_starts: int = 12
) -> Optional[np.ndarray]:
    """Return the best constants found, or None if every attempt failed."""
    if expr.n_constants == 0:
        return np.zeros(0)

    uses_trig = any(f in expr.source for f in ("sin", "cos", "tan"))
    if uses_trig:
        n_starts = max(n_starts, 40)
    y_scale = float(np.mean(np.abs(y))) or 1.0

    def resid(c: np.ndarray) -> np.ndarray:
        r = expr(env, c) - y
        return np.nan_to_num(r, nan=_BIG, posinf=_BIG, neginf=-_BIG)

    best_c, best_cost = None, float("inf")
    sst = float(np.sum((y - y.mean()) ** 2)) or 1.0
    for c0 in _starts(expr.n_constants, n_starts, rng, y_scale):
        try:
            # scipy's trust-region step emits RuntimeWarnings on degenerate Jacobians
            # (flat regions of a bad candidate). Results are NaN-checked below, so mute
            # only that warning class, only here.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sol = least_squares(resid, c0, method="trf", max_nfev=200, x_scale="jac")
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        if np.isfinite(sol.cost) and sol.cost < best_cost:
            best_cost, best_c = sol.cost, sol.x
            if 2 * best_cost / sst < 1e-6:  # essentially perfect: stop early
                break
    return best_c


def evaluate(
    source: str,
    variables: Sequence[str],
    target: str,
    train: Arrays,
    val: Arrays,
    rng: np.random.Generator,
) -> Candidate:
    """Parse, fit and score one candidate. Never raises on bad candidates."""
    try:
        expr = parse(source, variables)
    except ExprError as exc:
        return Candidate(source=source, ok=False, error=str(exc))

    try:
        consts = fit_constants(expr, train, train[target], rng)
        if consts is None:
            return Candidate(source=source, ok=False, error="could not fit constants (optimiser failed)")
        p_tr = expr(train, consts)
        p_va = expr(val, consts)
    except (ExprError, FloatingPointError, ValueError) as exc:
        return Candidate(source=source, ok=False, error=f"evaluation failed: {exc}")

    tr, va = r2_score(train[target], p_tr), r2_score(val[target], p_va)
    if not np.isfinite(tr):
        return Candidate(source=source, ok=False, error="expression produced NaN/inf on the data")
    return Candidate(
        source=source,
        ok=True,
        consts=[float(c) for c in consts],
        train_r2=tr,
        val_r2=va,
        nodes=expr.n_nodes,
        expr=expr,
    )


def residual_report(cand: Candidate, task_vars: Sequence[str], target: str, train: Arrays) -> str:
    """Describe where the best candidate is still wrong, as feedback for the agent.

    For each input variable we split the training points into three equal-count
    groups (low/mid/high) and report the mean residual (data - prediction) in
    each. A systematic trend means the model is missing structure in that variable.
    """
    pred = cand.predict(train)
    if pred is None:
        return ""
    res = train[target] - pred
    scale = float(np.std(train[target])) or 1.0
    lines = [f"RMS residual = {np.sqrt(np.mean(res ** 2)):.4g} (std of target = {scale:.4g})"]
    for v in task_vars:
        order = np.argsort(train[v])
        thirds = np.array_split(order, 3)
        means = [float(np.mean(res[idx])) / scale for idx in thirds]
        lines.append(
            f"mean residual / std(target) by {v} (low, mid, high third): "
            + ", ".join(f"{m:+.3f}" for m in means)
        )
    return "\n".join(lines)
