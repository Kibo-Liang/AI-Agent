"""Benchmark tasks: noisy measurements generated from *hidden* physical laws.

The agent sees a plain-language description, variable names/units and noisy
training/validation rows. It never sees the law, nor the clean test and
extrapolation sets that are used only for final scoring.

Extrapolation matters: a flexible curve fit can match in-range data yet fail
outside it, while the true law keeps working. Scoring on an extrapolation set
therefore separates "found the law" from "memorised the data".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np

Arrays = Dict[str, np.ndarray]


@dataclass
class Dataset:
    train: Arrays  # noisy, visible to the solver
    val: Arrays  # noisy, visible to the solver (model selection)
    test: Arrays  # clean, hidden, in-range
    extrap: Arrays  # clean, hidden, outside the training range


@dataclass(frozen=True)
class Task:
    name: str
    title: str
    description: str
    variables: Dict[str, str]  # input name -> meaning and unit
    target: str
    target_desc: str
    ranges: Dict[str, Tuple[float, float]]
    law: Callable[[Arrays], np.ndarray] = field(repr=False)
    truth: str  # human-readable ground truth; never shown to solvers
    sampling: Dict[str, str] = field(default_factory=dict)  # name -> "log" | "uniform"
    extrap_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    noise: float = 0.01  # noise std as a fraction of std(y)
    difficulty: str = "easy"

    def _sample(self, rng: np.random.Generator, n: int, ranges: Dict[str, Tuple[float, float]]) -> Arrays:
        out: Arrays = {}
        for name, (lo, hi) in ranges.items():
            if self.sampling.get(name) == "log":
                out[name] = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
            else:
                out[name] = rng.uniform(lo, hi, n)
        return out

    def _extrap_ranges(self) -> Dict[str, Tuple[float, float]]:
        res = {}
        for name, (lo, hi) in self.ranges.items():
            if name in self.extrap_ranges:
                res[name] = self.extrap_ranges[name]
            elif self.sampling.get(name) == "log":
                res[name] = (hi, hi * (hi / lo) ** 0.25)
            else:
                res[name] = (hi, hi + 0.5 * (hi - lo))
        return res

    def make_dataset(
        self, seed: int, n_train: int = 120, n_val: int = 40, n_test: int = 100, n_extrap: int = 100
    ) -> Dataset:
        rng = np.random.default_rng(seed)

        def build(n: int, ranges: Dict[str, Tuple[float, float]], noisy: bool) -> Arrays:
            x = self._sample(rng, n, ranges)
            y = self.law(x)
            if noisy:
                y = y + self.noise * np.std(y) * rng.standard_normal(n)
            x[self.target] = y
            return x

        return Dataset(
            train=build(n_train, self.ranges, True),
            val=build(n_val, self.ranges, True),
            test=build(n_test, self.ranges, False),
            extrap=build(n_extrap, self._extrap_ranges(), False),
        )


def _tasks() -> Dict[str, Task]:
    t: Dict[str, Task] = {}

    t["kepler"] = Task(
        name="kepler",
        title="Orbital period of planets",
        description="Planets orbit a star. We measured how long one orbit takes for planets at different distances.",
        variables={"a": "semi-major axis of the orbit, in astronomical units (AU)"},
        target="T",
        target_desc="orbital period in years",
        ranges={"a": (0.4, 30.0)},
        sampling={"a": "log"},
        law=lambda x: x["a"] ** 1.5,
        truth="T = a**1.5",
    )
    t["pendulum"] = Task(
        name="pendulum",
        title="Simple pendulum",
        description="A small-angle pendulum swings on Earth. We timed the swing for different string lengths.",
        variables={"L": "string length in metres"},
        target="T",
        target_desc="period of one full swing in seconds",
        ranges={"L": (0.1, 5.0)},
        law=lambda x: 2 * np.pi * np.sqrt(x["L"] / 9.81),
        truth="T = 2*pi*sqrt(L/9.81)",
    )
    t["ideal_gas"] = Task(
        name="ideal_gas",
        title="Gas in a sealed container",
        description="A gas in a sealed, rigid container was measured while varying how much gas there is, how hot it is and the container volume.",
        variables={
            "n": "amount of gas in moles",
            "T": "absolute temperature in kelvin",
            "V": "container volume in cubic metres",
        },
        target="P",
        target_desc="pressure in pascals",
        ranges={"n": (0.5, 5.0), "T": (250.0, 500.0), "V": (0.01, 0.1)},
        law=lambda x: 8.314 * x["n"] * x["T"] / x["V"],
        truth="P = 8.314*n*T/V",
    )
    t["inverse_square"] = Task(
        name="inverse_square",
        title="Attraction between two bodies",
        description="The attractive force between two bodies was measured for different masses and separations (scaled, dimensionless units).",
        variables={
            "m1": "mass of body 1 (scaled units)",
            "m2": "mass of body 2 (scaled units)",
            "r": "separation between the bodies (scaled units)",
        },
        target="F",
        target_desc="attractive force (scaled units)",
        ranges={"m1": (1.0, 10.0), "m2": (1.0, 10.0), "r": (1.0, 8.0)},
        law=lambda x: 6.674 * x["m1"] * x["m2"] / x["r"] ** 2,
        truth="F = 6.674*m1*m2/r**2",
    )
    t["projectile"] = Task(
        name="projectile",
        title="Projectile range",
        description="A projectile is launched over flat ground, ignoring air resistance. We measured how far it lands for different launch speeds and angles.",
        variables={
            "v": "launch speed in metres per second",
            "theta": "launch angle above horizontal in radians",
        },
        target="R",
        target_desc="horizontal distance travelled in metres",
        ranges={"v": (5.0, 30.0), "theta": (0.1, 0.9)},
        extrap_ranges={"v": (30.0, 40.0), "theta": (0.9, 1.4)},
        law=lambda x: x["v"] ** 2 * np.sin(2 * x["theta"]) / 9.81,
        truth="R = v**2*sin(2*theta)/9.81",
        difficulty="medium",
    )
    t["michaelis_menten"] = Task(
        name="michaelis_menten",
        title="Enzyme reaction rate",
        description="An enzyme converts a substrate into product. We measured the initial reaction rate at different substrate concentrations. The rate levels off at high concentration.",
        variables={"S": "substrate concentration in mM"},
        target="v",
        target_desc="initial reaction rate in mM/min",
        ranges={"S": (0.05, 10.0)},
        sampling={"S": "log"},
        law=lambda x: 2.5 * x["S"] / (0.8 + x["S"]),
        truth="v = 2.5*S/(0.8+S)",
        difficulty="medium",
    )
    t["cooling"] = Task(
        name="cooling",
        title="Cooling of a hot object",
        description="A hot object cools in a room. We recorded its temperature over time.",
        variables={"t": "time in minutes"},
        target="temp",
        target_desc="object temperature in degrees Celsius",
        ranges={"t": (0.0, 30.0)},
        law=lambda x: 20.0 + 60.0 * np.exp(-0.15 * x["t"]),
        truth="temp = 20 + 60*exp(-0.15*t)",
        difficulty="medium",
    )
    t["damped_oscillator"] = Task(
        name="damped_oscillator",
        title="Damped oscillation",
        description="A mass on a spring oscillates while friction slowly removes its energy. We recorded its displacement over time.",
        variables={"t": "time in seconds"},
        target="x",
        target_desc="displacement from rest position in centimetres",
        ranges={"t": (0.0, 15.0)},
        law=lambda x: 3.0 * np.exp(-0.2 * x["t"]) * np.cos(2.0 * x["t"]),
        truth="x = 3*exp(-0.2*t)*cos(2*t)",
        difficulty="hard",
    )
    return t


TASKS: Dict[str, Task] = _tasks()


def get_task(name: str) -> Task:
    try:
        return TASKS[name]
    except KeyError as exc:
        raise KeyError(f"unknown task '{name}'. Available: {', '.join(TASKS)}") from exc


def nrmse(y_true: np.ndarray, y_pred: Optional[np.ndarray], scale: float) -> float:
    """RMSE divided by a fixed reference ``scale`` (inf for invalid predictions).

    We normalise by the in-range std of the target rather than by the variance of
    the set being scored. R^2 on an extrapolation set collapses for saturating laws
    (e.g. cooling at late times is almost flat, so SST ~ 0 and a 0.05 C error looks
    catastrophic). A fixed scale measures the error in units of the signal's natural size.
    """
    if y_pred is None:
        return float("inf")
    y_pred = np.asarray(y_pred, dtype=float)
    if not np.all(np.isfinite(y_pred)) or scale <= 0:
        return float("inf")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)) / scale)


def r2_score(y_true: np.ndarray, y_pred: Optional[np.ndarray]) -> float:
    """Coefficient of determination; -inf for invalid predictions."""
    if y_pred is None:
        return float("-inf")
    y_pred = np.asarray(y_pred, dtype=float)
    if not np.all(np.isfinite(y_pred)):
        return float("-inf")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else float("-inf")
    return 1.0 - ss_res / ss_tot
