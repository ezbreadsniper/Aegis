"""
EarnHFT: Efficient Hierarchical Reinforcement Learning for High Frequency Trading.

Based on AAAI 2024 paper - achieves 30%+ higher profitability through:
1. Beta Agents: PPO with configurable risk preferences
2. Agent Pool: Specialized agents for different market regimes
3. Router: Meta-controller that dynamically selects best agent

Usage:
    from strategies.earnhft import EarnHFTStrategy, BetaAgent, AgentPool
    
    strategy = EarnHFTStrategy()
    action = strategy.act(state)
"""
from .beta_agent import BetaAgent
from .agent_pool import AgentPool
from .router import Router
from .earnhft_strategy import EarnHFTStrategy

__all__ = [
    "BetaAgent",
    "AgentPool", 
    "Router",
    "EarnHFTStrategy",
]
