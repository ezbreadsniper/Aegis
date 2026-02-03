#!/usr/bin/env python3
"""
BetaAgent: PPO agent with configurable risk preference.

The beta parameter controls the risk-return tradeoff:
- beta=0.1: Conservative (minimize drawdowns, fewer but safer trades)
- beta=0.5: Balanced (current default behavior)
- beta=0.9: Aggressive (maximize returns, accept higher risk)

Reward function: R = beta * PnL - (1 - beta) * drawdown_penalty

Based on EarnHFT (AAAI 2024) - trains multiple agents with different
preferences, then selects the best one for each market regime.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from collections import deque
import os

# Import base strategy components
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from strategies.base import Strategy, MarketState, Action


# Auto-detect device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


@dataclass
class Experience:
    """Single experience tuple."""
    state: np.ndarray
    temporal_state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_temporal_state: np.ndarray
    done: bool
    log_prob: float
    value: float


class TemporalEncoder(nn.Module):
    """Encodes temporal state history."""
    
    def __init__(self, input_dim: int = 31, history_len: int = 5, output_dim: int = 32):
        super().__init__()
        self.history_len = history_len
        self.temporal_input = input_dim * history_len
        self.fc1 = nn.Linear(self.temporal_input, 64)
        self.ln1 = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, output_dim)
        self.ln2 = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.ln1(self.fc1(x)))
        h = torch.tanh(self.ln2(self.fc2(h)))
        return h


class Actor(nn.Module):
    """Policy network."""
    
    def __init__(self, input_dim: int = 31, hidden_size: int = 64, output_dim: int = 3,
                 history_len: int = 5, temporal_dim: int = 32):
        super().__init__()
        self.temporal_encoder = TemporalEncoder(input_dim, history_len, temporal_dim)
        combined_dim = input_dim + temporal_dim
        self.fc1 = nn.Linear(combined_dim, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_dim)

    def forward(self, current_state: torch.Tensor, temporal_state: torch.Tensor) -> torch.Tensor:
        temporal_features = self.temporal_encoder(temporal_state)
        combined = torch.cat([current_state, temporal_features], dim=-1)
        h = torch.tanh(self.ln1(self.fc1(combined)))
        h = torch.tanh(self.ln2(self.fc2(h)))
        logits = self.fc3(h)
        return F.softmax(logits, dim=-1)


class Critic(nn.Module):
    """Value network."""
    
    def __init__(self, input_dim: int = 31, hidden_size: int = 96,
                 history_len: int = 5, temporal_dim: int = 32):
        super().__init__()
        self.temporal_encoder = TemporalEncoder(input_dim, history_len, temporal_dim)
        combined_dim = input_dim + temporal_dim
        self.fc1 = nn.Linear(combined_dim, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, current_state: torch.Tensor, temporal_state: torch.Tensor) -> torch.Tensor:
        temporal_features = self.temporal_encoder(temporal_state)
        combined = torch.cat([current_state, temporal_features], dim=-1)
        h = torch.tanh(self.ln1(self.fc1(combined)))
        h = torch.tanh(self.ln2(self.fc2(h)))
        return self.fc3(h)


class BetaAgent(Strategy):
    """
    PPO agent with configurable risk preference (beta).
    
    Beta controls the risk-return tradeoff in the reward function:
        R = beta * pnl - (1 - beta) * drawdown_penalty
    
    Lower beta = more conservative (avoid losses)
    Higher beta = more aggressive (chase gains)
    
    Args:
        beta: Risk preference in [0, 1]. Default 0.5 (balanced)
        input_dim: Feature dimension (28 for current MarketState)
        **kwargs: Additional PPO hyperparameters
    """
    
    def __init__(
        self,
        beta: float = 0.5,
        input_dim: int = 31,  # v7.5: 31 features (removed position_pnl)
        hidden_size: int = 64,
        critic_hidden_size: int = 96,
        history_len: int = 5,
        temporal_dim: int = 32,
        lr_actor: float = 1e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.95,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.03,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        buffer_size: int = 256,
        batch_size: int = 64,
        n_epochs: int = 10,
        target_kl: float = 0.02,
    ):
        super().__init__(f"beta_{beta:.1f}")
        
        # Beta preference (core EarnHFT parameter)
        assert 0.0 <= beta <= 1.0, "Beta must be in [0, 1]"
        self.beta = beta
        
        # Device
        self.device = DEVICE
        print(f"[BetaAgent β={beta:.1f}] Using device: {self.device}")
        
        # Architecture params
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.critic_hidden_size = critic_hidden_size
        self.history_len = history_len
        self.temporal_dim = temporal_dim
        self.output_dim = 3  # BUY, HOLD, SELL
        
        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.target_kl = target_kl
        
        # Networks
        self.actor = Actor(input_dim, hidden_size, self.output_dim, history_len, temporal_dim).to(self.device)
        self.critic = Critic(input_dim, critic_hidden_size, history_len, temporal_dim).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Experience buffer
        self.experiences: List[Experience] = []
        
        # Temporal state history
        self._state_history: Dict[str, deque] = {}
        
        # Running stats
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
        
        # Drawdown tracking for beta-adjusted rewards
        self.peak_pnl = 0.0
        self.current_pnl = 0.0
        
        # Last action info
        self._last_log_prob = 0.0
        self._last_value = 0.0
        self._last_temporal_state: Optional[np.ndarray] = None
        
    def compute_beta_reward(self, pnl: float) -> float:
        """
        Compute reward signal.
        
        v7.6 FIX: Removed drawdown penalty based on original repo analysis.
        Their TRAINING_JOURNAL.md explicitly warns:
        "Reward shaping is risky - When shaping rewards are gameable and
        similar magnitude to the real signal, agents optimize the wrong thing."
        
        Pure PnL rewards are more honest and prevent gaming the reward function.
        """
        # Update PnL tracking (for stats only, not used in reward)
        self.current_pnl += pnl
        self.peak_pnl = max(self.peak_pnl, self.current_pnl)
        
        # Pure PnL reward - no shaping, no drawdown penalty
        return pnl
        
    def _get_temporal_state(self, asset: str, current_features: np.ndarray) -> np.ndarray:
        """Get stacked temporal state for an asset."""
        if asset not in self._state_history:
            self._state_history[asset] = deque(maxlen=self.history_len)

        history = self._state_history[asset]
        history.append(current_features.copy())

        if len(history) < self.history_len:
            padding = [np.zeros(self.input_dim, dtype=np.float32)] * (self.history_len - len(history))
            stacked = np.concatenate(padding + list(history))
        else:
            stacked = np.concatenate(list(history))

        return stacked.astype(np.float32)

    @torch.no_grad()
    def act(self, state: MarketState) -> Action:
        """Select action using current policy."""
        features = state.to_features()
        temporal_state = self._get_temporal_state(state.asset, features)

        features_t = torch.tensor(features.reshape(1, -1), dtype=torch.float32, device=self.device)
        temporal_t = torch.tensor(temporal_state.reshape(1, -1), dtype=torch.float32, device=self.device)

        self.actor.eval()
        self.critic.eval()
        probs = self.actor(features_t, temporal_t)
        value = self.critic(features_t, temporal_t)

        probs_np = probs[0].cpu().numpy()
        value_np = float(value[0, 0].cpu().numpy())

        if self.training:
            action_idx = np.random.choice(self.output_dim, p=probs_np)
        else:
            action_idx = int(np.argmax(probs_np))

        self._last_log_prob = float(np.log(probs_np[action_idx] + 1e-8))
        self._last_value = value_np
        self._last_temporal_state = temporal_state

        return Action(action_idx)

    def store(self, state: MarketState, action: Action, raw_pnl: float,
              next_state: MarketState, done: bool):
        """Store experience with beta-adjusted reward."""
        # Apply beta preference to reward
        beta_reward = self.compute_beta_reward(raw_pnl)
        
        # Normalize
        self.reward_count += 1
        delta = beta_reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        self.reward_std = np.sqrt(
            ((self.reward_count - 1) * self.reward_std**2 + delta * (beta_reward - self.reward_mean))
            / max(1, self.reward_count)
        )
        norm_reward = (beta_reward - self.reward_mean) / (self.reward_std + 1e-8)

        next_features = next_state.to_features()
        next_temporal_state = self._get_temporal_state(next_state.asset, next_features)

        exp = Experience(
            state=state.to_features(),
            temporal_state=self._last_temporal_state if self._last_temporal_state is not None else np.zeros(self.history_len * self.input_dim, dtype=np.float32),
            action=action.value,
            reward=norm_reward,
            next_state=next_features,
            next_temporal_state=next_temporal_state,
            done=done,
            log_prob=self._last_log_prob,
            value=self._last_value,
        )
        self.experiences.append(exp)

        if len(self.experiences) > self.buffer_size:
            self.experiences = self.experiences[-self.buffer_size:]

    def _compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation."""
        n = len(rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)
        gae = 0
        
        for t in reversed(range(n)):
            if t == n - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            
        return advantages, returns

    def update(self) -> Optional[Dict[str, float]]:
        """PPO update step."""
        if len(self.experiences) < self.buffer_size:
            return None

        self.actor.train()
        self.critic.train()

        # Convert to arrays
        states = np.array([e.state for e in self.experiences], dtype=np.float32)
        temporal_states = np.array([e.temporal_state for e in self.experiences], dtype=np.float32)
        actions = np.array([e.action for e in self.experiences], dtype=np.int64)
        rewards = np.array([e.reward for e in self.experiences], dtype=np.float32)
        dones = np.array([e.done for e in self.experiences], dtype=np.float32)
        old_log_probs = np.array([e.log_prob for e in self.experiences], dtype=np.float32)
        old_values = np.array([e.value for e in self.experiences], dtype=np.float32)

        # Compute GAE
        with torch.no_grad():
            next_state_t = torch.tensor(self.experiences[-1].next_state.reshape(1, -1), dtype=torch.float32, device=self.device)
            next_temporal_t = torch.tensor(self.experiences[-1].next_temporal_state.reshape(1, -1), dtype=torch.float32, device=self.device)
            next_value = float(self.critic(next_state_t, next_temporal_t)[0, 0].cpu().numpy())

        advantages, returns = self._compute_gae(rewards, old_values, dones, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        temporal_t = torch.tensor(temporal_states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        old_values_t = torch.tensor(old_values, dtype=torch.float32, device=self.device)

        n_samples = len(self.experiences)
        metrics = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_fraction": []}

        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n_samples)
            epoch_kl = 0.0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_idx = indices[start:end]

                batch_states = states_t[batch_idx]
                batch_temporal = temporal_t[batch_idx]
                batch_actions = actions_t[batch_idx]
                batch_old_log_probs = old_log_probs_t[batch_idx]
                batch_advantages = advantages_t[batch_idx]
                batch_returns = returns_t[batch_idx]
                batch_old_values = old_values_t[batch_idx]

                # Actor update
                probs = self.actor(batch_states, batch_temporal)
                batch_size_local = batch_actions.shape[0]
                selected_probs = probs[torch.arange(batch_size_local, device=self.device), batch_actions]
                log_probs = torch.log(selected_probs + 1e-8)

                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.mean(torch.min(surr1, surr2))

                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                entropy_mean = torch.mean(entropy)
                policy_loss = policy_loss - self.entropy_coef * entropy_mean

                with torch.no_grad():
                    approx_kl = torch.mean(batch_old_log_probs - log_probs)
                    clip_frac = torch.mean(((ratio < 1 - self.clip_epsilon) | (ratio > 1 + self.clip_epsilon)).float())

                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Critic update
                values = self.critic(batch_states, batch_temporal).squeeze()
                values_clipped = batch_old_values + torch.clamp(values - batch_old_values, -self.clip_epsilon, self.clip_epsilon)
                value_loss1 = (batch_returns - values) ** 2
                value_loss2 = (batch_returns - values_clipped) ** 2
                value_loss = 0.5 * torch.mean(torch.max(value_loss1, value_loss2))

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy_mean.item())
                metrics["approx_kl"].append(approx_kl.item())
                metrics["clip_fraction"].append(clip_frac.item())

                epoch_kl += approx_kl.item()
                n_batches += 1

            if epoch_kl / max(1, n_batches) > self.target_kl:
                break

        self.experiences.clear()

        y_pred = old_values
        y_true = returns
        var_y = np.var(y_true)
        explained_var = 1 - np.var(y_true - y_pred) / (var_y + 1e-8) if var_y > 0 else 0.0

        result = {
            "policy_loss": np.mean(metrics["policy_loss"]),
            "value_loss": np.mean(metrics["value_loss"]),
            "entropy": np.mean(metrics["entropy"]),
            "approx_kl": np.mean(metrics["approx_kl"]),
            "clip_fraction": np.mean(metrics["clip_fraction"]),
            "explained_variance": explained_var,
            "beta": self.beta,
        }
        
        # Log training progress
        print(f"  [BetaAgent β={self.beta:.1f}] loss={result['policy_loss']:.4f} v_loss={result['value_loss']:.4f} ent={result['entropy']:.3f} kl={result['approx_kl']:.4f} ev={explained_var:.2f}")
        
        return result

    def save(self, path: str):
        """Save agent with beta metadata."""
        weights_path = path.replace(".npz", "").replace(".pt", "") + ".pt"
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "beta": self.beta,
            "input_dim": self.input_dim,
            "hidden_size": self.hidden_size,
            "history_len": self.history_len,
        }, weights_path)
        
        # Save stats
        stats_path = path.replace(".npz", "").replace(".pt", "") + "_stats.npz"
        np.savez(
            stats_path,
            reward_mean=self.reward_mean,
            reward_std=self.reward_std,
            reward_count=self.reward_count,
            peak_pnl=self.peak_pnl,
            current_pnl=self.current_pnl,
            beta=self.beta,
        )
        print(f"[BetaAgent β={self.beta:.1f}] Saved to {weights_path}")

    def load(self, path: str):
        """Load agent from checkpoint, adapting to new dimensions if needed."""
        pt_path = path.replace(".npz", "").replace(".pt", "") + ".pt"
        checkpoint = torch.load(pt_path, map_location=self.device)
        
        # Check architecture compatibility
        saved_input_dim = checkpoint.get("input_dim", 28)
        current_input_dim = self.input_dim
        
        if saved_input_dim != current_input_dim:
            print(f"[BetaAgent β={self.beta:.1f}] ⚠️  Architecture mismatch: Saved input_dim={saved_input_dim}, Current={current_input_dim}")
            print(f"[BetaAgent β={self.beta:.1f}] 🔄 Performing weight surgery to adapt model...")
            
            # Helper to expand weights
            def expand_weights(sd_key, target_model, src_state_dict):
                """Expand weights for input layers to match new dimension."""
                for key, param in target_model.named_parameters():
                    src_key = f"{key}"
                    if src_key not in src_state_dict:
                        continue
                        
                    src_tensor = src_state_dict[src_key]
                    
                    # Check if this is a temporal encoder input layer (fc1)
                    if "temporal_encoder.fc1.weight" in key:
                         # Temporal input is input_dim * history_len
                        old_in = saved_input_dim * self.history_len
                        new_in = current_input_dim * self.history_len
                        
                        if src_tensor.shape[1] == old_in and param.shape[1] == new_in:
                            # Create new tensor with correct shape
                            new_tensor = torch.zeros_like(param.data)
                            
                            # Copy weights (spaced out because interleave)
                            # Actually, temporal features are stacked: [t-4, t-3, t-2, t-1, t]
                            # Each block grew from 28 -> 29.
                            # We need to copy block by block.
                            
                            for h in range(self.history_len):
                                start_old = h * saved_input_dim
                                end_old = start_old + saved_input_dim
                                start_new = h * current_input_dim
                                end_new = start_new + saved_input_dim
                                
                                # Copy old weights
                                new_tensor[:, start_new:end_new] = src_tensor[:, start_old:end_old]
                                # New column remains 0.0 (neutral initialization)
                                
                            # Load adapted weights
                            param.data.copy_(new_tensor)
                            print(f"  - Adapted {key} from {src_tensor.shape} to {new_tensor.shape}")
                            continue

                    # Direct copy for other layers
                    if param.shape == src_tensor.shape:
                         param.data.copy_(src_tensor)
                    else:
                        print(f"  - ⚠️ Skipping incompatbile layer {key}: {src_tensor.shape} vs {param.shape}")

            # Apply surgery
            expand_weights("actor", self.actor, checkpoint["actor_state_dict"])
            expand_weights("critic", self.critic, checkpoint["critic_state_dict"])
            
            # Load optimizers (skip if mismatch, they will reset)
            print(f"[BetaAgent β={self.beta:.1f}] Optimizers reset due to architecture change.")
            
        else:
            # Normal load
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            if "actor_optimizer" in checkpoint:
                self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            if "critic_optimizer" in checkpoint:
                self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

        if "beta" in checkpoint:
            self.beta = checkpoint["beta"]
            self.name = f"beta_{self.beta:.1f}"
            
        print(f"[BetaAgent β={self.beta:.1f}] Loaded from {pt_path}")

        stats_path = path.replace(".npz", "").replace(".pt", "") + "_stats.npz"
        try:
            stats = np.load(stats_path)
            self.reward_mean = float(stats["reward_mean"])
            self.reward_std = float(stats["reward_std"])
            self.reward_count = int(stats["reward_count"])
        except FileNotFoundError:
            pass

    def reset(self):
        """Reset agent state."""
        self.experiences.clear()
        self._state_history.clear()
        self._last_temporal_state = None
        self.peak_pnl = 0.0
        self.current_pnl = 0.0
