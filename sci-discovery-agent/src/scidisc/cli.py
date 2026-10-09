"""Command line interface.

Examples
--------
  # Offline, no LLM needed: how well do classical baselines do?
  python -m scidisc bench --solvers power_law,template,poly3 --seeds 5

  # Run the LLM agent on one task with a LOCAL model served by Ollama
  python -m scidisc run --task kepler --backend openai --model qwen2.5:7b --verbose

  # Same with Claude
  export ANTHROPIC_API_KEY=...
  python -m scidisc run --task cooling --backend anthropic --verbose

  # Full comparison, baselines + agent, with per-run traces saved
  python -m scidisc bench --solvers power_law,template,poly3,agent --backend openai --model qwen2.5:7b --seeds 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from .agent import DiscoveryAgent
from .bench import run_benchmark, summarise_markdown, write_outputs
from .llm import AnthropicBackend, BackendError, LLMBackend, OpenAICompatBackend
from .solvers import BASELINES, Solver
from .tasks import TASKS


def _backend(args: argparse.Namespace) -> LLMBackend:
    if args.backend == "anthropic":
        return AnthropicBackend(model=args.model)
    if not args.model:
        raise BackendError("--model is required for --backend openai (e.g. qwen2.5:7b for Ollama)")
    return OpenAICompatBackend(model=args.model, base_url=args.base_url)


def _solvers(args: argparse.Namespace) -> List[Solver]:
    out: List[Solver] = []
    for name in args.solvers.split(","):
        name = name.strip()
        if name == "agent":
            out.append(DiscoveryAgent(_backend(args), max_iters=args.iters, n_per_iter=args.per_iter, verbose=args.verbose))
        elif name in BASELINES:
            out.append(BASELINES[name]())
        else:
            raise SystemExit(f"unknown solver '{name}'. Choose from: {', '.join(BASELINES)}, agent")
    return out


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="scidisc", description="LLM agent for scientific law discovery")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--backend", choices=["openai", "anthropic"], default="openai")
        sp.add_argument("--model", default=None, help="model name (Ollama tag, or Claude model id)")
        sp.add_argument("--base-url", default="http://localhost:11434/v1", help="OpenAI-compatible server URL")
        sp.add_argument("--iters", type=int, default=6, help="max agent rounds")
        sp.add_argument("--per-iter", type=int, default=5, help="candidates requested per round")
        sp.add_argument("--verbose", action="store_true")

    sp_list = sub.add_parser("tasks", help="list benchmark tasks")

    sp_run = sub.add_parser("run", help="run the LLM agent on one task")
    common(sp_run)
    sp_run.add_argument("--task", required=True, choices=list(TASKS))
    sp_run.add_argument("--seed", type=int, default=0)

    sp_b = sub.add_parser("bench", help="benchmark solvers across tasks and seeds")
    common(sp_b)
    sp_b.add_argument("--solvers", default="power_law,template,poly3")
    sp_b.add_argument("--tasks", default=",".join(TASKS))
    sp_b.add_argument("--seeds", type=int, default=5)
    sp_b.add_argument("--out", default="results")
    sp_b.add_argument("--stem", default="benchmark")

    args = p.parse_args(argv)

    if args.cmd == "tasks":
        for t in TASKS.values():
            print(f"{t.name:18s} [{t.difficulty:6s}] {t.title}")
        return 0

    try:
        if args.cmd == "run":
            args.solvers = "agent"
            args.verbose = True
            rows = run_benchmark(_solvers(args), [args.task], [args.seed], trace_dir=Path("results/traces"))
            r = rows[0]
            print(f"\nFinal formula : {r.formula}\nHidden truth  : {TASKS[args.task].truth}")
            print(f"val R2={r.val_r2:.5f}  test R2={r.test_r2:.5f}  extrapolation R2={r.extrap_r2:.5f}  LLM calls={r.llm_calls}")
            return 0

        solvers = _solvers(args)
        tasks = [t.strip() for t in args.tasks.split(",")]
        rows = run_benchmark(solvers, tasks, list(range(args.seeds)), trace_dir=Path(args.out) / "traces")
        write_outputs(rows, Path(args.out), args.stem)
        print("\n" + summarise_markdown(rows))
        print(f"\nSaved to {args.out}/{args.stem}.csv and .md")
        return 0
    except BackendError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
