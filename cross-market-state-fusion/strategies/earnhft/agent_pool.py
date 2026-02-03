#!/usr/bin/env python3
"""
AgentPool: Manages pool of specialized BetaAgents for different market regimes.

Part of EarnHFT (AAAI 2024) Stage 2:
1. Trains multiple agents with different beta preferences
2. Evaluates each agent on different market regimes
3. Builds a pool mapping regime -> best agent
"""
import os
import json
from glob import glob
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .beta_agent import BetaAgent


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    RANGING = "ranging"


@dataclass
class AgentPerformance:
    """Performance metrics for an agent on a regime."""
    agent_path: str
    beta: float
    regime: str
    total_pnl: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    trade_count: int = 0
    
    @property
    def score(self) -> float:
        """Composite score for ranking agents."""
        # Weight: 40% PnL, 30% Sharpe, 20% Win Rate, 10% Drawdown penalty
        drawdown_penalty = self.max_drawdown * 0.1
        return (
            self.total_pnl * 0.4 +
            self.sharpe_ratio * 0.3 +
            self.win_rate * 0.2 -
            drawdown_penalty
        )


class AgentPool:
    """
    Manages a pool of specialized BetaAgents.
    
    The pool contains the best-performing agent for each market regime,
    selected through evaluation on historical data.
    
    Usage:
        pool = AgentPool()
        pool.train_all_betas([0.1, 0.3, 0.5, 0.7, 0.9])
        pool.build_pool(validation_data)
        agent = pool.get_agent("trending_up")
    """
    
    DEFAULT_BETAS = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    def __init__(self, agents_dir: str = "earnhft_agents"):
        self.agents_dir = agents_dir
        os.makedirs(agents_dir, exist_ok=True)
        
        # Pool: regime -> (agent, performance)
        self.pool: Dict[str, Tuple[BetaAgent, AgentPerformance]] = {}
        
        # All trained agents
        self.trained_agents: Dict[float, str] = {}  # beta -> path
        
        # Current loaded agent cache
        self._loaded_agents: Dict[str, BetaAgent] = {}
        
    def get_agent_path(self, beta: float) -> str:
        """Get path for an agent with given beta."""
        return os.path.join(self.agents_dir, f"agent_beta_{beta:.1f}")
    
    def create_agent(self, beta: float) -> BetaAgent:
        """Create a new BetaAgent with given preference."""
        return BetaAgent(beta=beta)
    
    def save_agent(self, agent: BetaAgent):
        """Save an agent to the pool directory."""
        path = self.get_agent_path(agent.beta)
        agent.save(path)
        self.trained_agents[agent.beta] = path
        
    def load_agent(self, beta: float) -> Optional[BetaAgent]:
        """Load an agent by beta value."""
        path = self.get_agent_path(beta)
        if not os.path.exists(path + ".pt"):
            return None
            
        agent = BetaAgent(beta=beta)
        agent.load(path)
        return agent
    
    def list_trained_agents(self) -> List[Tuple[float, str]]:
        """List all trained agents in the directory."""
        pattern = os.path.join(self.agents_dir, "agent_beta_*.pt")
        agents = []
        for path in glob(pattern):
            # Extract beta from filename
            basename = os.path.basename(path)
            try:
                beta_str = basename.replace("agent_beta_", "").replace(".pt", "")
                beta = float(beta_str)
                agents.append((beta, path.replace(".pt", "")))
            except ValueError:
                continue
        return sorted(agents)
    
    def classify_regime(
        self,
        returns_1m: float,
        returns_5m: float,
        volatility: float,
        avg_volatility: float,
    ) -> MarketRegime:
        """
        Classify current market regime.
        
        Uses short-term returns and volatility to determine regime.
        """
        vol_ratio = volatility / max(avg_volatility, 1e-6)
        
        # Trend detection
        if returns_5m > 0.003:  # 0.3% up
            if vol_ratio > 1.5:
                return MarketRegime.HIGH_VOLATILITY
            return MarketRegime.TRENDING_UP
        elif returns_5m < -0.003:  # 0.3% down
            if vol_ratio > 1.5:
                return MarketRegime.HIGH_VOLATILITY
            return MarketRegime.TRENDING_DOWN
        
        # Volatility detection
        if vol_ratio > 1.5:
            return MarketRegime.HIGH_VOLATILITY
        elif vol_ratio < 0.7:
            return MarketRegime.LOW_VOLATILITY
            
        return MarketRegime.RANGING
    
    def build_pool_from_trained(self, default_mapping: Optional[Dict[str, float]] = None):
        """
        Build pool from trained agents using default regime mapping.
        
        Default mapping (can be overridden by evaluation):
        - trending_up: aggressive (beta=0.7)
        - trending_down: aggressive (beta=0.7)  
        - high_volatility: conservative (beta=0.3)
        - low_volatility: balanced (beta=0.5)
        - ranging: conservative (beta=0.3)
        """
        if default_mapping is None:
            default_mapping = {
                MarketRegime.TRENDING_UP.value: 0.7,
                MarketRegime.TRENDING_DOWN.value: 0.7,
                MarketRegime.HIGH_VOLATILITY.value: 0.3,
                MarketRegime.LOW_VOLATILITY.value: 0.5,
                MarketRegime.RANGING.value: 0.3,
            }
        
        trained = self.list_trained_agents()
        available_betas = {beta for beta, _ in trained}
        
        for regime_name, preferred_beta in default_mapping.items():
            # Find closest available beta
            if preferred_beta in available_betas:
                beta = preferred_beta
            else:
                beta = min(available_betas, key=lambda b: abs(b - preferred_beta))
                
            agent = self.load_agent(beta)
            if agent:
                perf = AgentPerformance(
                    agent_path=self.get_agent_path(beta),
                    beta=beta,
                    regime=regime_name,
                )
                self.pool[regime_name] = (agent, perf)
                print(f"[Pool] {regime_name} -> beta={beta:.1f}")
    
    def get_agent(self, regime: str) -> BetaAgent:
        """Get the best agent for a regime."""
        if regime in self.pool:
            return self.pool[regime][0]
        
        # Fallback to ranging (most conservative)
        if MarketRegime.RANGING.value in self.pool:
            return self.pool[MarketRegime.RANGING.value][0]
        
        # Last resort: create a balanced agent
        return BetaAgent(beta=0.5)
    
    def get_agent_for_state(
        self,
        returns_1m: float,
        returns_5m: float,
        volatility: float,
        avg_volatility: float,
    ) -> Tuple[BetaAgent, MarketRegime]:
        """Get appropriate agent based on current market state."""
        regime = self.classify_regime(returns_1m, returns_5m, volatility, avg_volatility)
        agent = self.get_agent(regime.value)
        return agent, regime
    
    def save_pool_config(self, path: str = "pool_config.json"):
        """Save pool configuration to JSON."""
        config = {
            regime: {
                "beta": perf.beta,
                "path": perf.agent_path,
                "score": perf.score,
            }
            for regime, (agent, perf) in self.pool.items()
        }
        
        full_path = os.path.join(self.agents_dir, path)
        with open(full_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"[Pool] Config saved to {full_path}")
    
    def load_pool_config(self, path: str = "pool_config.json") -> bool:
        """Load pool configuration from JSON."""
        full_path = os.path.join(self.agents_dir, path)
        if not os.path.exists(full_path):
            return False
            
        with open(full_path) as f:
            config = json.load(f)
            
        for regime, info in config.items():
            beta = info["beta"]
            agent = self.load_agent(beta)
            if agent:
                perf = AgentPerformance(
                    agent_path=info["path"],
                    beta=beta,
                    regime=regime,
                )
                self.pool[regime] = (agent, perf)
                
        print(f"[Pool] Loaded {len(self.pool)} agents from {full_path}")
        return True
    
    def __len__(self) -> int:
        return len(self.pool)
    
    def __iter__(self):
        return iter(self.pool.items())
