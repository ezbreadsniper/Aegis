"""
Trading strategies for Polymarket.

Usage:
    from strategies import create_strategy, AVAILABLE_STRATEGIES

    strategy = create_strategy("mean_revert")
    action = strategy.act(state)
"""
import sys
from .base import Strategy, MarketState, Action
from .random_strat import RandomStrategy
from .mean_revert import MeanRevertStrategy
from .momentum import MomentumStrategy
from .fade_spike import FadeSpikeStrategy
from .gating import GatingStrategy
from .earnhft import EarnHFTStrategy, BetaAgent, AgentPool
from .lacuna_pytorch import LacunaStrategy

# Auto-detect platform and select RL implementation
# - Windows/Linux: Use PyTorch (cross-platform)
# - Mac: Try MLX first (Apple Silicon optimized), fall back to PyTorch
_rl_backend = None

if sys.platform == "darwin":
    # Mac - try MLX first
    try:
        from .rl_mlx import RLStrategy
        _rl_backend = "mlx"
    except ImportError:
        try:
            from .rl_pytorch import RLStrategy
            _rl_backend = "pytorch"
        except ImportError:
            RLStrategy = None
            _rl_backend = None
else:
    # Windows/Linux - use PyTorch
    try:
        from .rl_pytorch import RLStrategy
        _rl_backend = "pytorch"
    except ImportError:
        try:
            from .rl_mlx import RLStrategy
            _rl_backend = "mlx"
        except ImportError:
            RLStrategy = None
            _rl_backend = None

if RLStrategy is None:
    print("Warning: No RL backend available. Install PyTorch (pip install torch) to enable RL strategy.")


AVAILABLE_STRATEGIES = [
    "random",
    "mean_revert",
    "momentum",
    "fade_spike",
    "rl",
    "aegis",  # AEGIS v1 - alias for RL
    "gating",
    "earnhft",
    "lacuna",
    "oddyssey", # Oddyssey v1 - alias for Lacuna
]


def create_strategy(name: str, **kwargs) -> Strategy:
    """Factory function to create strategies."""
    strategies = {
        "random": RandomStrategy,
        "mean_revert": MeanRevertStrategy,
        "momentum": MomentumStrategy,
        "fade_spike": FadeSpikeStrategy,
        "rl": RLStrategy,
        "aegis": RLStrategy,  # AEGIS v1 - same as RL
        "lacuna": LacunaStrategy,
        "oddyssey": LacunaStrategy, # Oddyssey v1 - same as Lacuna
    }

    if name == "gating":
        # Create gating with default experts
        experts = [
            MeanRevertStrategy(),
            MomentumStrategy(),
            FadeSpikeStrategy(),
        ]
        return GatingStrategy(experts, **kwargs)
    
    if name == "earnhft":
        return EarnHFTStrategy(**kwargs)

    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}")

    strategy_class = strategies[name]
    if strategy_class is None:
        if name == "rl":
            raise ImportError(
                f"RL strategy not available. Install PyTorch: pip install torch\n"
                f"Or on Apple Silicon Mac, install MLX: pip install mlx"
            )
        raise ValueError(f"Strategy {name} is not available")

    return strategy_class(**kwargs)


def get_rl_backend() -> str:
    """Get the current RL backend being used."""
    return _rl_backend or "none"


__all__ = [
    # Base
    "Strategy",
    "MarketState",
    "Action",
    # Strategies
    "RandomStrategy",
    "MeanRevertStrategy",
    "MomentumStrategy",
    "FadeSpikeStrategy",
    "RLStrategy",
    "GatingStrategy",
    "EarnHFTStrategy",
    "BetaAgent",
    "AgentPool",
    "LacunaStrategy",
    # Factory
    "create_strategy",
    "AVAILABLE_STRATEGIES",
    "get_rl_backend",
]
