"""The LLM discovery agent: propose -> fit -> evaluate -> reflect -> repeat.

Division of labour (deliberate):
  * The LLM supplies *structure*: scientifically motivated candidate formulas.
  * Deterministic code supplies *numbers and truth*: it validates the formula in a
    sandbox, fits the free constants, scores on held-out data, and reports back
    exactly where the best formula still fails (residual diagnostics).
The LLM never executes code and never sees the hidden test/extrapolation sets.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .expr import ExprError, canonical_key, parse
from .fitting import Candidate, evaluate, residual_report
from .llm import BackendError, LLMBackend
from .solvers import Solver, SolveResult, result_from_candidates
from .tasks import Dataset, Task

SYSTEM_PROMPT = """You are a scientist's assistant specialising in discovering closed-form laws from measurements.

You propose candidate formulas; a separate program fits their numeric constants and scores them on held-out data. Your job is to find the simplest formula that explains the data AND would keep working outside the measured range.

RULES FOR EVERY FORMULA
- Write only the right-hand side, e.g.  c0*x**c1 + c2
- Use ONLY the variable names given in the task.
- Write free constants as c0, c1, c2, ... numbered from c0 with no gaps (at most 8). They are fitted for you, so do not guess their values. You may also use numeric literals and pi.
- Allowed operators: + - * / **   Allowed functions: sqrt, exp, log, sin, cos, tan, tanh, abs
- Prefer simple, physically meaningful forms. Think about units, limits, scaling, saturation, decay, periodicity.

OUTPUT FORMAT
Reply with ONLY a JSON object, no other text:
{"candidates": [{"expr": "<formula>", "why": "<one short sentence>"}, ...]}"""


def _data_summary(task: Task, data: Dataset, n_rows: int = 10) -> str:
    cols = list(task.variables) + [task.target]
    lines = ["variable | meaning | min | max | mean"]
    for v in cols:
        arr = data.train[v]
        meaning = task.variables.get(v, task.target_desc + " (TARGET to predict)")
        lines.append(f"{v} | {meaning} | {arr.min():.4g} | {arr.max():.4g} | {arr.mean():.4g}")
    first = list(task.variables)[0]
    order = np.argsort(data.train[first])
    idx = order[np.linspace(0, len(order) - 1, n_rows).astype(int)]
    lines.append("")
    lines.append("sample rows (sorted by %s):" % first)
    lines.append(" ".join(f"{c:>10}" for c in cols))
    for i in idx:
        lines.append(" ".join(f"{data.train[c][i]:>10.4g}" for c in cols))
    return "\n".join(lines)


def build_prompt(
    task: Task,
    data: Dataset,
    n_candidates: int,
    leaderboard: List[Candidate],
    last_round: List[Candidate],
    residuals: str,
    tried_dupes: List[str],
    parse_note: str,
) -> str:
    parts = [
        f"TASK: {task.title}\n{task.description}",
        f"Predict `{task.target}` ({task.target_desc}) from: " + ", ".join(task.variables) + ".",
        "DATA (train split):\n" + _data_summary(task, data),
    ]
    if leaderboard:
        parts.append(
            "BEST FORMULAS SO FAR (fitted; higher val R2 is better):\n"
            + "\n".join(f"{i + 1}. {c.short()}" for i, c in enumerate(leaderboard))
        )
        parts.append("WHERE THE BEST FORMULA STILL FAILS:\n" + residuals)
    bad = [c for c in last_round if not c.ok]
    if bad:
        parts.append(
            "INVALID CANDIDATES LAST ROUND (fix the problem, do not repeat it):\n"
            + "\n".join(f"- {c.short()}" for c in bad)
        )
    if tried_dupes:
        parts.append("ALREADY TRIED (do not repeat): " + "; ".join(tried_dupes[:8]))
    if parse_note:
        parts.append("NOTE: " + parse_note)
    ask = (
        f"Propose {n_candidates} NEW candidate formulas"
        + (", improving on the best so far and diversifying structure." if leaderboard else ", covering different plausible structures.")
        + " Reply with the JSON object only."
    )
    parts.append(ask)
    return "\n\n".join(parts)


def parse_candidates(text: str) -> Tuple[List[str], str]:
    """Extract formula strings from a model reply. Returns (formulas, note_if_failed)."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()

    def try_load(s: str) -> Optional[Any]:
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return None

    obj = try_load(cleaned)
    if obj is None:
        m = re.search(r"\{.*\}", cleaned, re.S)
        obj = try_load(m.group(0)) if m else None
    if obj is None:
        m = re.search(r"\[.*\]", cleaned, re.S)
        obj = try_load(m.group(0)) if m else None

    items: Any = obj.get("candidates") if isinstance(obj, dict) else obj
    out: List[str] = []
    if isinstance(items, list):
        for it in items:
            if isinstance(it, str):
                out.append(it)
            elif isinstance(it, dict) and isinstance(it.get("expr"), str):
                out.append(it["expr"])
    if not out:  # last resort: pull "expr": "..." pairs out of malformed JSON
        out = re.findall(r'"expr"\s*:\s*"([^"]+)"', cleaned)
    if not out:
        return [], "Your previous reply was not valid JSON of the required form. Reply with ONLY the JSON object."
    return out, ""


class DiscoveryAgent(Solver):
    name = "llm_agent"

    def __init__(
        self,
        backend: LLMBackend,
        max_iters: int = 6,
        n_per_iter: int = 5,
        target_val_r2: float = 0.9995,
        verbose: bool = False,
    ):
        self.backend = backend
        self.max_iters, self.n_per_iter = max_iters, n_per_iter
        self.target_val_r2, self.verbose = target_val_r2, verbose
        self.name = f"llm_agent[{backend.name}]"

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def solve(self, task: Task, data: Dataset, seed: int) -> SolveResult:
        rng = np.random.default_rng(seed)
        names = list(task.variables)
        seen: Dict[str, Candidate] = {}
        all_cands: List[Candidate] = []
        trace: List[Dict[str, Any]] = []
        last_round: List[Candidate] = []
        last_skipped: List[str] = []
        parse_note = ""
        start_calls = self.backend.calls

        for it in range(1, self.max_iters + 1):
            ranked = sorted((c for c in all_cands if c.ok), key=lambda c: c.score, reverse=True)
            board = ranked[:5]
            resid = residual_report(board[0], names, task.target, data.train) if board else ""
            prompt = build_prompt(task, data, self.n_per_iter, board, last_round, resid, last_skipped, parse_note)

            try:
                reply = self.backend.complete(SYSTEM_PROMPT, [{"role": "user", "content": prompt}])
            except BackendError as exc:
                self._log(f"[iter {it}] backend error: {exc}")
                trace.append({"iter": it, "error": str(exc)})
                break

            exprs, parse_note = parse_candidates(reply)
            round_cands: List[Candidate] = []
            skipped: List[str] = []
            for src in exprs[: self.n_per_iter * 2]:
                try:
                    key = canonical_key(parse(src, names))
                except ExprError:
                    key = None
                if key is not None and key in seen:
                    skipped.append(src)
                    continue
                cand = evaluate(src, names, task.target, data.train, data.val, rng)
                if key is not None:
                    seen[key] = cand
                round_cands.append(cand)
                all_cands.append(cand)
            last_round, last_skipped = round_cands, skipped

            best = max((c for c in all_cands if c.ok), key=lambda c: c.score, default=None)
            self._log(
                f"[iter {it}] proposed={len(exprs)} new={len(round_cands)} dup={len(skipped)} "
                f"best={best.short() if best else 'none'}"
            )
            trace.append(
                {
                    "iter": it,
                    "reply": reply[:2500],
                    "candidates": [
                        {"expr": c.source, "ok": c.ok, "error": c.error, "val_r2": c.val_r2, "nodes": c.nodes}
                        for c in round_cands
                    ],
                    "skipped_duplicates": skipped,
                    "best_val_r2": best.val_r2 if best else None,
                }
            )
            if best is not None and best.val_r2 >= self.target_val_r2:
                break

        return result_from_candidates(self.name, all_cands, trace, self.backend.calls - start_calls)
