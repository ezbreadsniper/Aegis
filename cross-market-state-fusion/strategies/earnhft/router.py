#!/usr/bin/env python3
"""
Router: Minute-level meta-controller that selects which agent to use.

Part of EarnHFT (AAAI 2024) Stage 3:
- Operates at a slower timescale (minute-level vs second-level)
- Takes current market state as input
- Outputs probability distribution over agents in the pool
- Learns which agent is best for current conditions

The router is itself trained with RL to maximize cumulative reward.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass
import time
import os


# Auto-detect device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


@dataclass
class RouterExperience:
    """Experience for router training."""
    state: np.ndarray  # Market state features
    agent_idx: int     # Selected agent index
    reward: float      # Cumulative reward over routing period
    next_state: np.ndarray
    done: bool
    log_prob: float


class RouterNetwork(nn.Module):
    """
    Neural network that selects which agent to use.
    
    Architecture:
        Market state (28) → 64 → ReLU → 32 → ReLU → num_agents (softmax)
    
    Output is a probability distribution over the agent pool.
    """
    
    def __init__(self, state_dim: int = 31, hidden_size: int = 64, num_agents: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_agents)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns agent selection probabilities."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


class Router:
    """
    Minute-level router that dynamically selects which agent to use.
    
    The router operates on a slower timescale than the agents:
    - Agents make decisions every tick (500ms)
    - Router re-evaluates every N seconds (default 60s)
    
    The router is trained using REINFORCE to maximize cumulative reward
    over the routing period.
    
    Args:
        num_agents: Number of agents in the pool
        state_dim: Dimension of state features
        routing_interval: Seconds between routing decisions
        lr: Learning rate for router optimization
    """
    
    def __init__(
        self,
        num_agents: int = 5,
        state_dim: int = 31,  # v7.5: 31 features (removed position_pnl)
        routing_interval: float = 60.0,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 100,
    ):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.routing_interval = routing_interval
        self.gamma = gamma
        self.buffer_size = buffer_size
        
        # Network
        self.device = DEVICE
        self.network = RouterNetwork(state_dim, 64, num_agents).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Experience buffer
        self.experiences: List[RouterExperience] = []
        
        # Routing state
        self._last_route_time = 0.0
        self._current_agent_idx = 0
        self._last_log_prob = 0.0
        self._cumulative_reward = 0.0
        self._route_start_state: Optional[np.ndarray] = None
        
        # Stats
        self.agent_selection_counts = np.zeros(num_agents)
        
    def should_reroute(self) -> bool:
        """Check if it's time to make a new routing decision."""
        now = time.time()
        return (now - self._last_route_time) >= self.routing_interval
    
    @torch.no_grad()
    def select_agent(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select which agent to use based on current state.
        
        Args:
            state: Current market state features (28,)
            training: If True, sample from distribution; else take argmax
            
        Returns:
            Index of selected agent
        """
        # Store previous routing experience
        if self._route_start_state is not None:
            exp = RouterExperience(
                state=self._route_start_state,
                agent_idx=self._current_agent_idx,
                reward=self._cumulative_reward,
                next_state=state,
                done=False,
                log_prob=self._last_log_prob,
            )
            self.experiences.append(exp)
            if len(self.experiences) > self.buffer_size:
                self.experiences = self.experiences[-self.buffer_size:]
        
        # Get agent probabilities
        state_t = torch.tensor(state.reshape(1, -1), dtype=torch.float32, device=self.device)
        self.network.eval()
        probs = self.network(state_t)[0].cpu().numpy()
        
        # Select agent
        if training:
            agent_idx = np.random.choice(self.num_agents, p=probs)
        else:
            agent_idx = int(np.argmax(probs))
        
        # Update state for next routing
        self._current_agent_idx = agent_idx
        self._last_log_prob = float(np.log(probs[agent_idx] + 1e-8))
        self._last_route_time = time.time()
        self._cumulative_reward = 0.0
        self._route_start_state = state.copy()
        
        # Stats
        self.agent_selection_counts[agent_idx] += 1
        
        return agent_idx
    
    def add_reward(self, reward: float):
        """Add reward to current routing period."""
        self._cumulative_reward += reward
    
    def get_current_agent_idx(self) -> int:
        """Get the currently selected agent index."""
        return self._current_agent_idx
    
    def update(self) -> Optional[Dict[str, float]]:
        """
        Update router using REINFORCE.
        
        Returns:
            Training metrics or None if not enough data
        """
        if len(self.experiences) < 10:
            return None
            
        self.network.train()
        
        # Compute discounted returns
        returns = []
        R = 0
        for exp in reversed(self.experiences):
            R = exp.reward + self.gamma * R
            returns.insert(0, R)
        returns = np.array(returns)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # REINFORCE loss
        total_loss = 0.0
        for exp, R in zip(self.experiences, returns):
            state_t = torch.tensor(exp.state.reshape(1, -1), dtype=torch.float32, device=self.device)
            probs = self.network(state_t)[0]
            log_prob = torch.log(probs[exp.agent_idx] + 1e-8)
            loss = -log_prob * R
            total_loss += loss
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        # Clear experiences
        avg_reward = np.mean([e.reward for e in self.experiences])
        self.experiences.clear()
        self._route_start_state = None
        
        return {
            "router_loss": total_loss.item(),
            "avg_routing_reward": avg_reward,
            "selection_distribution": self.agent_selection_counts / max(1, self.agent_selection_counts.sum()),
        }
    
    def save(self, path: str):
        """Save router state."""
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_agents": self.num_agents,
            "state_dim": self.state_dim,
            "routing_interval": self.routing_interval,
            "selection_counts": self.agent_selection_counts,
        }, path)
        print(f"[Router] Saved to {path}")
    
    def load(self, path: str):
        """Load router state with weight surgery for dimension changes."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Check for dimension mismatch
        saved_state_dim = checkpoint.get("state_dim", 31)
        if saved_state_dim != self.state_dim:
            print(f"[Router] ⚠️  Dimension mismatch: Saved state_dim={saved_state_dim}, Current={self.state_dim}")
            print(f"[Router] 🔄 Performing weight surgery to adapt model...")
            
            # Get saved weights
            saved_state = checkpoint["network_state_dict"]
            new_state = self.network.state_dict()
            
            # Adapt fc1.weight (first layer needs to match input dimension)
            for key in saved_state:
                if key == "fc1.weight":
                    old_weight = saved_state[key]
                    new_weight = new_state[key]
                    if old_weight.shape != new_weight.shape:
                        # Expand by adding zeros for new features
                        diff = new_weight.shape[1] - old_weight.shape[1]
                        if diff > 0:
                            # Pad with small random values for new features
                            padding = torch.randn(old_weight.shape[0], diff, device=self.device) * 0.01
                            adapted_weight = torch.cat([old_weight.to(self.device), padding], dim=1)
                            saved_state[key] = adapted_weight
                            print(f"  - Adapted {key} from {old_weight.shape} to {adapted_weight.shape}")
                        else:
                            print(f"  - ⚠️ Cannot shrink {key}, using fresh initialization")
                    else:
                        saved_state[key] = old_weight
                elif key in new_state:
                    if saved_state[key].shape == new_state[key].shape:
                        pass  # Compatible, keep as is
                    else:
                        print(f"  - ⚠️ Skipping incompatible layer {key}: {saved_state[key].shape} vs {new_state[key].shape}")
                        saved_state[key] = new_state[key]  # Use fresh initialization
                        
            self.network.load_state_dict(saved_state, strict=False)
            # Reset optimizer due to architecture change
            self.optimizer = optim.Adam(self.network.parameters(), lr=1e-3)
            print(f"[Router] Optimizer reset due to architecture change.")
        else:
            self.network.load_state_dict(checkpoint["network_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
        if "selection_counts" in checkpoint:
            self.agent_selection_counts = checkpoint["selection_counts"]
        print(f"[Router] Loaded from {path}")
    
    def reset_stats(self):
        """Reset selection statistics."""
        self.agent_selection_counts = np.zeros(self.num_agents)
