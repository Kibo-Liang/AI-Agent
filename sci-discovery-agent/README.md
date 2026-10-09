# scidisc — an LLM agent that discovers scientific laws from data

An LLM proposes candidate formulas for noisy measurements; deterministic code **sandboxes, fits and scores** them on held-out data and feeds the failures back; the loop repeats. Success is judged on data **outside the measured range**, so memorising the training points doesn't count as discovering a law.

Works with a **fully local model** (Ollama / llama.cpp / vLLM / LM Studio) or Claude. No required API key. Only `numpy` and `scipy` are needed to run the offline parts.

```
        task description + noisy train/val rows
                        │
                        ▼
   ┌──────────────────────────────────────────────┐
   │  LLM  — proposes N candidate formulas (JSON) │◄───────────────┐
   └──────────────────────┬───────────────────────┘                │
                          ▼                                        │
   ┌──────────────────────────────────────────────┐      feedback: leaderboard,
   │ Sandbox: parse with `ast`, whitelist only    │      invalid-candidate reasons,
   │ (never eval/exec)                            │      residual diagnostics
   └──────────────────────┬───────────────────────┘      ("model is too low at
                          ▼                                high t"), duplicates
   ┌──────────────────────────────────────────────┐                │
   │ Fit constants c0..ck (multi-start LSQ)       │                │
   │ Score on validation set + mild complexity    ├────────────────┘
   │ penalty                                      │
   └──────────────────────┬───────────────────────┘
                          ▼
              best formula  ──►  scored ONCE on hidden test +
                                  extrapolation data (never shown to the LLM)
```

## Design decisions

| Decision | Why |
|---|---|
| LLM proposes *structure*, code fits *numbers* | LLMs are good at physically-motivated forms and bad at arithmetic/curve fitting. Constants are written `c0, c1, …` and fitted numerically. |
| Formulas parsed with `ast` + whitelist, evaluated by a tree-walker | LLM output is untrusted text. No `eval`/`exec`; only `+ - * / **`, 8 math functions, ≤40 nodes, bounded exponents. Injection, attribute access, lambdas, null bytes and deep nesting are covered by tests. |
| Three-way data split + extrapolation set | The agent only sees train/val. Final scoring uses a clean in-range test set and a clean set **outside** the training range. A polynomial fits in-range data well and fails here; the true law does not. |
| Residual diagnostics fed back to the LLM | Instead of just "score = 0.93", the agent is told *where* the best formula is wrong (mean residual by low/mid/high third of each variable). |
| Stateless prompts rebuilt each round | Cheaper and more robust for small local models than an ever-growing chat history. |
| Backend = one method `complete(system, messages)` | Adding a provider is ~15 lines. |

## Quickstart

```bash
pip install -e .            # numpy + scipy
pip install -e .[anthropic] # only if you want Claude

# tests (stdlib unittest, no extra deps)
PYTHONPATH=src python -m unittest discover -s tests

# list tasks
python -m scidisc tasks

# 1) Offline baselines, no LLM:
python -m scidisc bench --solvers power_law,template,poly3 --seeds 5

# 2) The agent with a LOCAL model (Ollama):
ollama pull qwen2.5:7b
python -m scidisc run --task cooling --backend openai --model qwen2.5:7b

# 3) The agent with Claude:
export ANTHROPIC_API_KEY=...
python -m scidisc run --task damped_oscillator --backend anthropic

# 4) Full comparison (baselines + agent), traces saved to results/traces/
python -m scidisc bench --solvers power_law,template,poly3,agent \
    --backend openai --model qwen2.5:7b --seeds 3 --stem with_agent
```

Per-run JSON traces (every prompt reply, every candidate, every score) are written to `results/traces/`.

## Benchmark

Eight tasks, 1% noise, from easy (`kepler`, `pendulum`) to hard (`damped_oscillator`). The laws are hidden; the agent sees only a plain-language description, variable meanings/units and noisy rows.

**Success** = extrapolation RMSE ≤ 5% of the target's in-range standard deviation (NRMSE ≤ 0.05).

### Non-LLM baselines (measured, 5 seeds each, `results/baselines.csv`)

Median extrapolation NRMSE (lower is better) and successes out of 5 seeds:

| Task (difficulty) | poly3 | power_law | template |
|---|---|---|---|
| kepler (easy) | 4.4636 (0/5) | 0.0210 (4/5) | 0.0210 (4/5) |
| pendulum (easy) | 0.7123 (0/5) | 0.0026 (5/5) | 0.0014 (5/5) |
| ideal_gas (easy) | 1.0259 (0/5) | 0.0029 (5/5) | 0.0024 (5/5) |
| inverse_square (easy) | 2.5611 (0/5) | 0.0010 (5/5) | 0.0011 (5/5) |
| projectile (medium) | 1.0481 (0/5) | 3.7046 (0/5) | 0.0075 (5/5) |
| michaelis_menten (medium) | >100 (0/5) | 1.6935 (0/5) | 0.0019 (5/5) |
| cooling (medium) | 1.7419 (0/5) | 0.1707 (0/5) | 0.0004 (5/5) |
| damped_oscillator (hard) | 3.5914 (0/5) | 0.0783 (0/5) | 0.7705 (0/5) |
| **Total successes** | **0/40** | **19/40** | **34/40** |

* `poly3`: degree-3 polynomial. Fits the in-range data (test R² > 0.99 on several tasks) yet never extrapolates — it is the "memorised, didn't discover" control.
* `power_law`: `c0·v1^c1·v2^c2…`. Solves pure scaling laws, fails anything with saturation, decay or oscillation.
* `template`: a hand-written menu of 12 common shapes (× each variable as "main" variable). Strong on this benchmark — see the caveat below.

### LLM agent results

**Not yet measured.** I have not run the agent against a real LLM, so there are no agent numbers here and none are implied. Run command (4) above and paste the table into this section. The interesting rows are the ones the baselines leave open: `damped_oscillator` (0/5 for every baseline).

## How to read these numbers (caveats)

* **The `template` baseline is a strong, hand-tuned opponent.** Its library happens to contain the functional form of 7 of the 8 tasks (it was written by someone who knew the benchmark), so 34/40 is a ceiling-ish reference, not a typical classical method. The `damped_oscillator` form is deliberately *not* in the library.
* **The tasks are textbook laws.** A language model may have memorised Kepler's third law, so success on these tasks is not evidence of *discovery*. A fair claim about discovery needs tasks the model cannot have seen (see Roadmap).
* **The success metric was changed once, openly.** I first scored extrapolation with R². For saturating laws that is misleading: for `cooling` at late times the true curve is almost flat, its variance collapses, and a 0.05 °C error produced R² = 0.89 for a correct formula. I switched to RMSE normalised by the *in-range* standard deviation (`tasks.nrmse`) and added a regression test. The change applied to every solver; it also moved `kepler` from 5/5 to 4/5 (see below).
* **Threshold edge effects exist.** `kepler`/`power_law` seed 2 recovered `c0=1.008, c1=1.497` (truth 1, 1.5) — the right law — but missed the 0.05 line with NRMSE 0.0510, because a 0.2% exponent error is amplified when extrapolating 3×. I did not tune the threshold after seeing this.
* **NRMSE is lenient for decaying signals.** `power_law` on `damped_oscillator` scores NRMSE 0.078 (close to passing) even though its in-range test R² is ≈ 0, because the true signal has mostly decayed in the extrapolation window. Look at `test_r2` in the CSV alongside NRMSE.

## Limitations

* Only 1-D target, ≤3 inputs, closed-form expressions; no units/dimensional-analysis checking yet.
* Constant fitting is multi-start least squares: it can miss frequencies/phases of oscillatory forms (the reason `damped_oscillator` is hard for classical baselines).
* Tested on Python 3.11 only so far (full suite: 37 tests). A GitHub Actions matrix for 3.9–3.12 is included in `.github/workflows/test.yml` but has not been run yet. I verified directly that `ast.parse` raises `ValueError` (not `SyntaxError`) on null bytes under Python 3.10 and handle it, but did not run the full suite there.
* The HTTP backend is tested against a local fake server (plumbing only), not against a real Ollama/vLLM instance.

## Roadmap (ideas that make the science stronger)

1. **Contamination-proof tasks:** perturbed laws (`T = 0.8·a^1.7`), renamed/anonymised variables, invented systems. If the agent only wins on the originals, it recalled; if it wins on both, it discovered.
2. **Stronger baselines:** genetic-programming symbolic regression (e.g. PySR) at an equal candidate budget.
3. **Ablations:** no residual feedback vs. feedback; one-shot vs. iterative; model size sweep (local 3B → 70B → Claude).
4. **Noise/data-size sweeps** and **dimensional analysis** as an additional validity check on proposed formulas.
5. **Tool use:** let the agent request specific plots/statistics instead of receiving fixed summaries.

## Layout

```
src/scidisc/
  expr.py      safe parser/evaluator (the security boundary)
  tasks.py     hidden-law tasks, splits, noise, metrics
  fitting.py   constant fitting, candidate scoring, residual diagnostics
  llm.py       Anthropic / OpenAI-compatible (local) / scripted backends
  agent.py     prompt building, robust JSON parsing, the discovery loop
  solvers.py   solver interface + 3 baselines
  bench.py     benchmark runner and tables
  cli.py       `python -m scidisc ...`
tests/         37 tests: sandbox attacks, parsing, fitting, agent edge cases, HTTP e2e
```
