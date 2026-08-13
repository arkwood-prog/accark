"""Turn a score-probability matrix into prices for every market.

The design rule here: **one object, many markets**. Everything is expressed as
a boolean mask over the joint score matrix, so

    P(selection)          = sum(matrix * mask)
    P(sel_a AND sel_b)    = sum(matrix * mask_a * mask_b)

That second line is why same-game combinations get an *exact* joint
probability instead of the wrong answer you get by multiplying two correlated
legs together. "Home win" and "Over 2.5" are not independent, and this treats
them properly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np

from ..data.schema import Selection

# Markets whose outcomes are mutually exclusive and exhaustive. These are the
# only ones that can be devigged or blended directly; everything else is
# derived from them so the whole price set stays internally consistent.
PARTITIONS: dict[str, tuple[str, ...]] = {
    "1X2": ("1X2:H", "1X2:D", "1X2:A"),
    "BTTS": ("BTTS:Y", "BTTS:N"),
}


def ou_partition(line: float) -> tuple[str, ...]:
    return (f"OU{line:g}:O", f"OU{line:g}:U")


@lru_cache(maxsize=512)
def _grids(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    home = np.arange(size)[:, None] * np.ones((1, size), dtype=int)
    away = np.ones((size, 1), dtype=int) * np.arange(size)[None, :]
    return home, away, home + away


@lru_cache(maxsize=2048)
def selection_mask(selection_id: str, size: int) -> np.ndarray:
    """Boolean mask over the (size x size) score matrix for one selection."""
    sel = Selection.parse(selection_id)
    home, away, total = _grids(size)

    if sel.market == "1X2":
        return {"H": home > away, "D": home == away, "A": home < away}[sel.pick]
    if sel.market == "DC":
        return {"1X": home >= away, "12": home != away, "X2": home <= away}[sel.pick]
    if sel.market == "OU":
        if sel.line is None:
            raise ValueError(f"over/under selection needs a line: {selection_id}")
        return total > sel.line if sel.pick == "O" else total < sel.line
    if sel.market == "BTTS":
        both = (home > 0) & (away > 0)
        return both if sel.pick == "Y" else ~both
    if sel.market == "CS":
        home_goals, _, away_goals = sel.pick.partition("-")
        return (home == int(home_goals)) & (away == int(away_goals))
    raise ValueError(f"unsupported market: {sel.market}")


def probability(matrix: np.ndarray, selection_id: str) -> float:
    return float(matrix[selection_mask(selection_id, matrix.shape[0])].sum())


def joint_probability(matrix: np.ndarray, selection_ids: Sequence[str]) -> float:
    """Exact joint probability of several selections *in the same match*."""
    size = matrix.shape[0]
    mask = np.ones((size, size), dtype=bool)
    for selection_id in selection_ids:
        mask &= selection_mask(selection_id, size)
    return float(matrix[mask].sum())


def price_markets(
    matrix: np.ndarray,
    markets: Iterable[str] = ("1X2", "DC", "OU", "BTTS"),
    ou_lines: Iterable[float] = (1.5, 2.5, 3.5),
) -> dict[str, float]:
    """Probabilities for every selection the engine supports."""
    markets = set(markets)
    probs: dict[str, float] = {}
    if "1X2" in markets:
        for sel in PARTITIONS["1X2"]:
            probs[sel] = probability(matrix, sel)
    if "DC" in markets:
        for sel in ("DC:1X", "DC:12", "DC:X2"):
            probs[sel] = probability(matrix, sel)
    if "OU" in markets:
        for line in ou_lines:
            for sel in ou_partition(line):
                probs[sel] = probability(matrix, sel)
    if "BTTS" in markets:
        for sel in PARTITIONS["BTTS"]:
            probs[sel] = probability(matrix, sel)
    return probs


def top_scorelines(matrix: np.ndarray, count: int = 5) -> list[tuple[str, float]]:
    """Most likely exact scores, highest first."""
    flat = matrix.ravel()
    size = matrix.shape[1]
    order = np.argsort(flat)[::-1][:count]
    return [(f"{int(i) // size}-{int(i) % size}", float(flat[i])) for i in order]


def partitions_for(probs: dict[str, float]) -> list[tuple[str, ...]]:
    """Which exhaustive partitions are present in a probability dict."""
    found: list[tuple[str, ...]] = []
    for group in PARTITIONS.values():
        if all(sel in probs for sel in group):
            found.append(group)
    lines = sorted({Selection.parse(sel).line for sel in probs
                    if sel.startswith("OU") and Selection.parse(sel).line is not None})
    for line in lines:
        group = ou_partition(float(line))
        if all(sel in probs for sel in group):
            found.append(group)
    return found


def calibrate_matrix(
    matrix: np.ndarray,
    targets: dict[str, float],
    iterations: int = 60,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Reshape a score matrix so its market margins match `targets`.

    Iterative proportional fitting. After blending model and market opinions we
    have improved probabilities for 1X2, the over/under lines and BTTS — but
    those live outside the score matrix. This pushes them back *into* the
    matrix, so correct scores and same-game combinations inherit the
    market-anchored view instead of the raw model's.
    """
    groups = [group for group in partitions_for(targets)]
    if not groups:
        return matrix

    adjusted = matrix.astype(float).copy()
    size = adjusted.shape[0]
    masks = {sel: selection_mask(sel, size) for group in groups for sel in group}

    for _ in range(iterations):
        largest_shift = 0.0
        for group in groups:
            current = np.array([adjusted[masks[sel]].sum() for sel in group])
            wanted = np.array([targets[sel] for sel in group], dtype=float)
            wanted = wanted / wanted.sum()
            for sel, now, goal in zip(group, current, wanted):
                if now <= 1e-12:
                    continue
                scale = goal / now
                adjusted[masks[sel]] *= scale
                largest_shift = max(largest_shift, abs(scale - 1.0))
            adjusted /= adjusted.sum()
        if largest_shift < tolerance:
            break
    return adjusted


def expected_goals_from_matrix(matrix: np.ndarray) -> tuple[float, float]:
    size = matrix.shape[0]
    goals = np.arange(size)
    return float((matrix.sum(axis=1) * goals).sum()), float((matrix.sum(axis=0) * goals).sum())
