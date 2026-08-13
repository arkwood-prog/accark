"""Market-side pricing: margin removal and consensus fair prices."""

from .consensus import MarketQuote, market_view
from .devig import devig, devig_multiplicative, devig_power, devig_shin, fair_odds, overround

__all__ = [
    "MarketQuote",
    "market_view",
    "devig",
    "devig_multiplicative",
    "devig_power",
    "devig_shin",
    "fair_odds",
    "overround",
]
