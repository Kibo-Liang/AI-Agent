"""Common solver interface plus three non-LLM baselines.

Every solver receives a task and the *visible* part of a dataset (train/val) and
returns a :class:`SolveResult`. The benchmark then scores the returned predictor
on the hidden test and extrapolation sets.

Baselines
---------
power_law   c0 * v1**c1 * v2**c2 ...                    (classic log-log regression idea)
template    small hand-written library of common forms   (classical "try a menu of shapes")
poly3       degree-3 polynomial in standardised inputs   (flexible curve fit, no law)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .fitting import Candidate, evaluate
from .tasks import Arrays, Dataset, Task, r2_score

Predictor = Callable[[Arrays], Optional[np.ndarray]]


@dataclass
class SolveResult:
    solver: str
    formula: str
    predict: Predictor
    val_r2: float
    n_candidates: int
    trace: List[Dict[str, Any]] = field(default_factory=list)
    llm_calls: int = 0


class Solver:
    name = "solver"

    def solve(self, task: Task, data: Dataset, seed: int) -> SolveResult:  # pragma: no cover
        raise NotImplementedError


def result_from_candidates(name: str, cands: List[Candidate], trace=None, llm_calls: int = 0) -> SolveResult:
    ok = [c for c in cands if c.ok]
    if not ok:
        return SolveResult(name, "<none>", lambda env: None, float("-inf"), len(cands), trace or [], llm_calls)
    best = max(ok, key=lambda c: c.score)
    cs = ", ".join(f"c{i}={v:.4g}" for i, v in enumerate(best.consts))
    return SolveResult(
        name,
        f"{best.expr.source}   [{cs}]" if best.expr else best.source,
        best.predict,
        best.val_r2,
        len(cands),
        trace or [],
        llm_calls,
    )


class PowerLawBaseline(Solver):
    name = "power_law"

    def solve(self, task: Task, data: Dataset, seed: int) -> SolveResult:
        rng = np.random.default_rng(seed)
        names = list(task.variables)
        src = "c0*" + "*".join(f"{v}**c{i + 1}" for i, v in enumerate(names))
        cand = evaluate(src, names, task.target, data.train, data.val, rng)
        return result_from_candidates(self.name, [cand])


# Each template uses c0.. for its own constants and the placeholder {v} for the
# "main" variable. Remaining variables enter as a power-law factor.
_TEMPLATES = [
    ("c0*{v}", 1),
    ("c0*{v}+c1", 2),
    ("c0*{v}**c1", 2),
    ("c0*sqrt({v})", 1),
    ("c0/{v}", 1),
    ("c0/(c1+{v})", 2),
    ("c0*{v}/(c1+{v})", 2),
    ("c0*exp(c1*{v})", 2),
    ("c0+c1*exp(c2*{v})", 3),
    ("c0*log({v})+c1", 2),
    ("c0*sin(c1*{v})", 2),
    ("c0*cos(c1*{v}+c2)", 3),
]


class TemplateLibraryBaseline(Solver):
    name = "template"

    def solve(self, task: Task, data: Dataset, seed: int) -> SolveResult:
        rng = np.random.default_rng(seed)
        names = list(task.variables)
        cands: List[Candidate] = []
        for main in names:
            others = [v for v in names if v != main]
            for tmpl, k in _TEMPLATES:
                src = tmpl.format(v=main)
                src += "".join(f"*{u}**c{k + i}" for i, u in enumerate(others))
                cands.append(evaluate(src, names, task.target, data.train, data.val, rng))
        return result_from_candidates(self.name, cands)


class PolynomialBaseline(Solver):
    """Degree-3 polynomial least squares: fits the data without discovering a law."""

    name = "poly3"

    def __init__(self, degree: int = 3):
        self.degree = degree

    def solve(self, task: Task, data: Dataset, seed: int) -> SolveResult:
        names = list(task.variables)
        mu = {v: data.train[v].mean() for v in names}
        sd = {v: data.train[v].std() or 1.0 for v in names}
        powers = [p for p in itertools.product(range(self.degree + 1), repeat=len(names)) if sum(p) <= self.degree]

        def design(env: Arrays) -> np.ndarray:
            z = {v: (env[v] - mu[v]) / sd[v] for v in names}
            return np.column_stack([np.prod([z[v] ** e for v, e in zip(names, p)], axis=0) for p in powers])

        coef, *_ = np.linalg.lstsq(design(data.train), data.train[task.target], rcond=None)

        def predict(env: Arrays) -> np.ndarray:
            return design(env) @ coef

        val_r2 = r2_score(data.val[task.target], predict(data.val))
        return SolveResult(self.name, f"degree-{self.degree} polynomial ({len(coef)} coefficients)", predict, val_r2, 1)


BASELINES: Dict[str, Callable[[], Solver]] = {
    "power_law": PowerLawBaseline,
    "template": TemplateLibraryBaseline,
    "poly3": PolynomialBaseline,
}
