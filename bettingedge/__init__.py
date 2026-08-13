"""bettingedge — data-driven football betting analysis.

    from bettingedge import Engine, Config
    from bettingedge.data import synthetic

    matches, fixtures = synthetic.generate()
    slate = Engine(Config()).build_slate(matches, fixtures)
    print(slate.singles[0].analysis)

Nothing in this package is a guarantee of profit. It is a modelling and
discipline tool: it estimates probabilities, compares them with the price on
offer, and sizes bets so that a real edge can survive variance.
"""

from .config import Config, MarketConfig, ModelConfig, ParlayConfig, SelectionConfig, StakingConfig
from .pipeline import Engine, Slate
from .verify import VerificationReport, sharp_only, verify

__version__ = "1.0.0"

__all__ = [
    "Config",
    "ModelConfig",
    "MarketConfig",
    "SelectionConfig",
    "ParlayConfig",
    "StakingConfig",
    "Engine",
    "Slate",
    "verify",
    "sharp_only",
    "VerificationReport",
    "__version__",
]
