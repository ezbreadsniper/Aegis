#!/usr/bin/env python3
"""
EarnHFTStrategy: Complete hierarchical RL trading strategy.

Combines all EarnHFT components:
1. AgentPool: Collection of specialized BetaAgents
2. Router: Meta-controller that selects which agent to use
3. Dynamic regime-based switching

This is the main strategy class to use in production.
"""
import time
import numpy as np
from typing import Dict, Optional
import os

from .beta_agent import BetaAgent
from .agent_pool import AgentPool, MarketRegime
from .router import Router

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from strategies.base import Strategy, MarketState, Action


class EarnHFTStrategy(Strategy):
    """
    Complete EarnHFT hierarchical reinforcement learning strategy.
    
    Architecture:
        Router (minute-level) → selects from → AgentPool → selected Agent → Action
    
    The router periodically (default: every 60s) evaluates the market state
    and selects the most appropriate agent from the pool. The selected agent
    then makes trading decisions at the normal tick frequency.
    
    Usage:
        strategy = EarnHFTStrategy(agents_dir="earnhft_agents")
        action = strategy.act(state)
    
    Args:
        agents_dir: Directory containing trained agents
        use_router: If True, use learned router; if False, use regime-based selection
        routing_interval: Seconds between routing decisions
    """
    
    def __init__(
        self,
        agents_dir: str = "earnhft_agents",
        use_router: bool = True,
        routing_interval: float = 60.0,
        fallback_beta: float = 0.5,
    ):
        super().__init__("earnhft")
        
        self.agents_dir = agents_dir
        self.use_router = use_router
        self.fallback_beta = fallback_beta
        
        # Initialize components
        self.pool = AgentPool(agents_dir)
        self.router: Optional[Router] = None
        self.current_agent: Optional[BetaAgent] = None
        
        # Routing state
        self._last_route_time = 0.0
        self.routing_interval = routing_interval
        self._current_regime: Optional[MarketRegime] = None
        
        # Stats
        self.routing_count = 0
        self.actions_since_route = 0
        
        # Load pool if available
        self._initialize()
        
    def _initialize(self):
        """Initialize pool and router from saved state or bootstrap defaults."""
        # Try to load existing pool
        if self.pool.load_pool_config():
            print(f"[EarnHFT] Loaded pool with {len(self.pool)} agents")
        else:
            # Bootstrap a default pool if none exists
            print("[EarnHFT] No saved pool found. Bootstrapping default agent pool...")
            default_betas = [0.1, 0.3, 0.5, 0.7, 0.9]
            
            # Create agents for each beta
            for beta in default_betas:
                agent = self.pool.create_agent(beta)
                self.pool.save_agent(agent)
                print(f"  - Created agent beta={beta}")
            
            # Map regimes to agents (default mapping)
            self.pool.build_pool_from_trained()
            self.pool.save_pool_config()
            print(f"[EarnHFT] Bootstrapped pool with {len(self.pool)} agents")

        # Initialize router (now that pool exists)
        if self.use_router:
            num_agents = len(self.pool)
            self.router = Router(num_agents=num_agents, routing_interval=self.routing_interval)
            
            # Try to load saved router
            router_path = os.path.join(self.agents_dir, "router.pt")
            if os.path.exists(router_path):
                try:
                    self.router.load(router_path)
                    print("[EarnHFT] Loaded router from checkpoint")
                except Exception as e:
                    print(f"[EarnHFT] Could not load router (starting fresh): {e}")

    def _should_reroute(self) -> bool:
        """Check if it's time for a routing decision."""
        now = time.time()
        return (now - self._last_route_time) >= self.routing_interval
    
    def _route_with_router(self, state: MarketState):
        """Use learned router to select agent."""
        features = state.to_features()
        agent_idx = self.router.select_agent(features, training=self.training)
        
        # Get agent from pool (map index to regime)
        regimes = list(self.pool.pool.keys())
        if agent_idx < len(regimes):
            regime_name = regimes[agent_idx]
            self.current_agent = self.pool.get_agent(regime_name)
            self._current_regime = MarketRegime(regime_name)
        else:
            self.current_agent = BetaAgent(beta=self.fallback_beta)
            
        self.routing_count += 1
        self.actions_since_route = 0
        self._last_route_time = time.time()
    
    def _route_with_regime(self, state: MarketState):
        """Use regime classification to select agent."""
        # Get market indicators
        returns_1m = getattr(state, 'returns_1m', 0.0)
        returns_5m = getattr(state, 'returns_5m', 0.0)
        volatility = getattr(state, 'realized_vol_5m', 0.01)
        avg_volatility = 0.01  # Default average
        
        old_regime = self._current_regime
        self.current_agent, self._current_regime = self.pool.get_agent_for_state(
            returns_1m, returns_5m, volatility, avg_volatility
        )
        
        # Log regime switch
        if old_regime != self._current_regime:
            print(f"  [EarnHFT] Regime: {self._current_regime.value if self._current_regime else 'none'} → β={self.current_agent.beta:.1f}")
        
        self.routing_count += 1
        self.actions_since_route = 0
        self._last_route_time = time.time()
    
    def act(self, state: MarketState) -> Action:
        """
        Select action using hierarchical decision making.
        
        1. If time to reroute, select new agent (via router or regime)
        2. Use current agent to select action
        """
        # Check if we need to (re)route
        if self.current_agent is None or self._should_reroute():
            if self.use_router and self.router is not None:
                self._route_with_router(state)
            elif len(self.pool) > 0:
                self._route_with_regime(state)
            else:
                # No pool, create fallback
                self.current_agent = BetaAgent(beta=self.fallback_beta)
                
        # Use current agent to act
        action = self.current_agent.act(state)
        self.actions_since_route += 1
        
        return action
    
    def store(self, state: MarketState, action: Action, reward: float,
              next_state: MarketState, done: bool):
        """Store experience for both agent and router training."""
        if self.current_agent is None:
            return
            
        # Store for agent training
        self.current_agent.store(state, action, reward, next_state, done)
        
        # Accumulate reward for router
        if self.router is not None:
            self.router.add_reward(reward)
    
    def update(self) -> Optional[Dict[str, float]]:
        """Update both agent and router."""
        metrics = {}
        
        # Update current agent
        if self.current_agent is not None:
            agent_metrics = self.current_agent.update()
            if agent_metrics:
                metrics.update(agent_metrics)
                metrics["current_beta"] = self.current_agent.beta
        
        # Update router (less frequently)
        if self.router is not None and len(self.router.experiences) >= 10:
            router_metrics = self.router.update()
            if router_metrics:
                metrics["router_loss"] = router_metrics.get("router_loss", 0)
                metrics["avg_routing_reward"] = router_metrics.get("avg_routing_reward", 0)
        
        if self._current_regime:
            metrics["current_regime"] = self._current_regime.value
        metrics["routing_count"] = self.routing_count
        metrics["actions_since_route"] = self.actions_since_route
        
        return metrics if metrics else None
    
    def save(self, path: str):
        """Save all EarnHFT components."""
        # Save pool config
        self.pool.save_pool_config()
        
        # Save router
        if self.router is not None:
            router_path = os.path.join(self.agents_dir, "router.pt")
            self.router.save(router_path)
        
        # Save current agent
        if self.current_agent is not None:
            self.current_agent.save(os.path.join(self.agents_dir, "current_agent"))
            
        print(f"[EarnHFT] Saved to {self.agents_dir}")
    
    def load(self, path: str):
        """Load EarnHFT state (re-initialize)."""
        self._initialize()
    
    def reset(self):
        """Reset strategy state."""
        self._last_route_time = 0.0
        self.routing_count = 0
        self.actions_since_route = 0
        
        if self.current_agent:
            self.current_agent.reset()
        if self.router:
            self.router.reset_stats()
    
    def get_status(self) -> Dict:
        """Get current strategy status."""
        return {
            "strategy": "earnhft",
            "pool_size": len(self.pool),
            "current_agent_beta": self.current_agent.beta if self.current_agent else None,
            "current_regime": self._current_regime.value if self._current_regime else None,
            "routing_count": self.routing_count,
            "use_router": self.use_router,
        }
