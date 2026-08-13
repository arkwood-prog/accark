"""Removing the bookmaker's margin from a set of prices.

Decimal odds imply probabilities that sum to more than 1; the excess is the
overround (vig). How you strip it matters, because the margin is *not* spread
evenly across outcomes — books load more of it onto longshots.

Three methods, in increasing order of realism:

* ``multiplicative``  divide by the total. Simple, but it systematically
  overstates longshots and understates favourites.
* ``power``           solve ``sum(q_i ** k) = 1``. Handles the bias better.
* ``shin``            Shin (1993), which models the margin as protection
  against insider trading. The standard choice for 3-way football markets.
"""

from __future__ import annotations

import math
from typing import Sequence

from scipy.optimize import brentq


def overround(odds: Sequence[float]) -> float:
    """Bookmaker margin, e.g. 0.05 for a 5% book."""
    return sum(1.0 / o for o in odds) - 1.0


def implied(odds: Sequence[float]) -> list[float]:
    return [1.0 / o for o in odds]


def fair_odds(probabilities: Sequence[float]) -> list[float]:
    return [float("inf") if p <= 0 else 1.0 / p for p in probabilities]


def devig_multiplicative(odds: Sequence[float]) -> list[float]:
    raw = implied(odds)
    total = sum(raw)
    return [q / total for q in raw]


def devig_power(odds: Sequence[float]) -> list[float]:
    """Find the exponent k with sum(q_i ** k) == 1."""
    raw = implied(odds)
    if abs(sum(raw) - 1.0) < 1e-12:
        return list(raw)

    def residual(k: float) -> float:
        return sum(q**k for q in raw) - 1.0

    try:
        k = brentq(residual, 0.5, 3.0, xtol=1e-12, maxiter=200)
    except (ValueError, RuntimeError):
        return devig_multiplicative(odds)
    adjusted = [q**k for q in raw]
    total = sum(adjusted)
    return [p / total for p in adjusted]


def devig_shin(odds: Sequence[float]) -> list[float]:
    """Shin's model: solve for the insider-trading proportion z."""
    raw = implied(odds)
    booksum = sum(raw)
    if booksum <= 1.0 + 1e-12 or len(raw) < 2:
        return devig_multiplicative(odds)

    def probs_for(z: float) -> list[float]:
        if z <= 1e-9:
            return [q / booksum for q in raw]
        return [
            (math.sqrt(z * z + 4.0 * (1.0 - z) * q * q / booksum) - z) / (2.0 * (1.0 - z))
            for q in raw
        ]

    def residual(z: float) -> float:
        return sum(probs_for(z)) - 1.0

    try:
        z = brentq(residual, 1e-9, 0.6, xtol=1e-12, maxiter=200)
    except (ValueError, RuntimeError):
        return devig_power(odds)
    result = probs_for(z)
    total = sum(result)
    return [p / total for p in result]


_METHODS = {
    "multiplicative": devig_multiplicative,
    "power": devig_power,
    "shin": devig_shin,
}


def devig(odds: Sequence[float], method: str = "shin") -> list[float]:
    """Strip the margin from a complete set of prices for one market."""
    clean = [float(o) for o in odds]
    if len(clean) < 2 or any(o <= 1.0 for o in clean):
        raise ValueError(f"need at least two valid decimal odds, got {odds!r}")
    if method not in _METHODS:
        raise ValueError(f"unknown devig method {method!r}; pick one of {sorted(_METHODS)}")
    # Shin and power are only defined for an over-round book.
    if sum(implied(clean)) <= 1.0:
        return devig_multiplicative(clean)
    return _METHODS[method](clean)
