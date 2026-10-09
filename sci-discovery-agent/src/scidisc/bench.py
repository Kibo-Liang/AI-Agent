"""Benchmark runner: score solvers on hidden in-range and extrapolation data."""

from __future__ import annotations

import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .solvers import Solver
from .tasks import TASKS, Task, nrmse, r2_score

SUCCESS_NRMSE = 0.05  # "found the law": extrapolation RMSE <= 5% of the in-range std of the target


@dataclass
class Row:
    task: str
    solver: str
    seed: int
    formula: str
    val_r2: float
    test_r2: float
    extrap_r2: float
    extrap_nrmse: float
    n_candidates: int
    llm_calls: int
    seconds: float

    @property
    def success(self) -> bool:
        return self.extrap_nrmse <= SUCCESS_NRMSE


def run_one(solver: Solver, task: Task, seed: int, trace_dir: Optional[Path] = None) -> Row:
    data = task.make_dataset(seed)
    t0 = time.time()
    res = solver.solve(task, data, seed)
    dt = time.time() - t0
    test_r2 = r2_score(data.test[task.target], res.predict(data.test))
    extrap_r2 = r2_score(data.extrap[task.target], res.predict(data.extrap))
    scale = float(np.std(data.test[task.target]))  # in-range natural size of the signal
    extrap_nrmse = nrmse(data.extrap[task.target], res.predict(data.extrap), scale)
    if trace_dir is not None and res.trace:
        trace_dir.mkdir(parents=True, exist_ok=True)
        safe = res.solver.replace("[", "_").replace("]", "")
        (trace_dir / f"{task.name}__{safe}__seed{seed}.json").write_text(
            json.dumps({"task": task.name, "formula": res.formula, "trace": res.trace}, indent=2)
        )
    return Row(task.name, res.solver, seed, res.formula, res.val_r2, test_r2, extrap_r2, extrap_nrmse, res.n_candidates, res.llm_calls, dt)


def run_benchmark(
    solvers: Sequence[Solver],
    task_names: Sequence[str],
    seeds: Sequence[int],
    trace_dir: Optional[Path] = None,
    progress: bool = True,
) -> List[Row]:
    rows: List[Row] = []
    for tn in task_names:
        task = TASKS[tn]
        for solver in solvers:
            for seed in seeds:
                row = run_one(solver, task, seed, trace_dir)
                rows.append(row)
                if progress:
                    print(
                        f"{tn:18s} {row.solver:22s} seed={seed} test R2={row.test_r2:8.4f} "
                        f"extrap NRMSE={row.extrap_nrmse:9.4f} ({row.seconds:.1f}s)",
                        flush=True,
                    )
    return rows


def _fmt(x: float) -> str:
    if x == float("inf") or x != x:
        return "fail"
    return f"{x:.4f}" if x < 100 else ">100"


def summarise_markdown(rows: List[Row]) -> str:
    solvers = sorted({r.solver for r in rows})
    tasks = [t for t in TASKS if any(r.task == t for r in rows)]
    out = ["| Task (difficulty) | " + " | ".join(solvers) + " |", "|---|" + "---|" * len(solvers)]
    for t in tasks:
        cells = []
        for s in solvers:
            rs = [r for r in rows if r.task == t and r.solver == s]
            if not rs:
                cells.append("-")
                continue
            med = statistics.median(r.extrap_nrmse for r in rs)
            ok = sum(r.success for r in rs)
            cells.append(f"{_fmt(med)} ({ok}/{len(rs)})")
        out.append(f"| {t} ({TASKS[t].difficulty}) | " + " | ".join(cells) + " |")
    total = []
    for s in solvers:
        rs = [r for r in rows if r.solver == s]
        total.append(f"**{sum(r.success for r in rs)}/{len(rs)}**")
    out.append("| **Total successes** | " + " | ".join(total) + " |")
    return "\n".join(out)


def write_outputs(rows: List[Row], out_dir: Path, stem: str = "benchmark") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{stem}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "solver", "seed", "val_r2", "test_r2", "extrap_r2", "extrap_nrmse", "success", "n_candidates", "llm_calls", "seconds", "formula"])
        for r in rows:
            w.writerow([r.task, r.solver, r.seed, r.val_r2, r.test_r2, r.extrap_r2, r.extrap_nrmse, int(r.success), r.n_candidates, r.llm_calls, f"{r.seconds:.2f}", r.formula])
    (out_dir / f"{stem}.md").write_text(
        f"Median extrapolation NRMSE over seeds (lower is better; success = NRMSE <= {SUCCESS_NRMSE}).\n\n" + summarise_markdown(rows) + "\n"
    )
