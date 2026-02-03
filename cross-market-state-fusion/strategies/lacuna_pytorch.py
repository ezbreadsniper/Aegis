#!/usr/bin/env python3
"""
HumanPlane/LACUNA Adapter Strategy (PyTorch Version).

This strategy wraps the pre-trained LACUNA v5 model (29 features)
to work within the current v7.5 environment (31 features).

It performs "Feature Adaption" by slicing off the new features (VPIN/Spoofing)
before passing the state to the network.

UPDATE: Added training support (PPO) for fine-tuning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from typing import Optional, Dict, List, Any
from collections import deque, namedtuple
from .base import Strategy, MarketState, Action

# Named tuple for RL experiences
Experience = namedtuple('Experience', [
    'state', 'temporal_state', 'action', 'reward', 
    'next_state', 'next_temporal_state', 'done', 
    'log_prob', 'value'
])

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int = 29, history_len: int = 5, output_dim: int = 32):
        super().__init__()
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
    def __init__(self, input_dim: int = 29, hidden_size: int = 64, output_dim: int = 3,
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
        return F.softmax(logits + 1e-10, dim=-1)

class Critic(nn.Module):
    def __init__(self, input_dim: int = 29, hidden_size: int = 96,
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

class LacunaStrategy(Strategy):
    """Fine-tunable Adapter for HumanPlane/LACUNA (29 features)."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        hidden_size: int = 64,
        critic_hidden_size: int = 96,
        history_len: int = 5,
        temporal_dim: int = 32,
        lr_actor: float = 5e-5,  # Lower LR for fine-tuning
        lr_critic: float = 1e-4,
        gamma: float = 0.95,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01, # Low entropy to preserve pre-trained policy
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        buffer_size: int = 256,
        batch_size: int = 64,
        n_epochs: int = 10,
    ):
        super().__init__("oddyssey")
        
        # LACUNA specs
        self.input_dim = 29
        self.output_dim = 3
        self.history_len = history_len
        self.temporal_dim = temporal_dim
        
        # Hyperparams
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        # Reward Normalization
        self.reward_mean = 0
        self.reward_std = 1
        self.reward_count = 0

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🧬 [LACUNA] Initializing on {self.device}")

        # Initialize Architecture
        self.actor = Actor(self.input_dim, hidden_size, self.output_dim, history_len, temporal_dim).to(self.device)
        self.critic = Critic(self.input_dim, critic_hidden_size, history_len, temporal_dim).to(self.device)
        
        # Optimizers
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Experience Buffer
        self.experiences: List[Experience] = []
        
        # Temporal state tracking
        self._state_history: Dict[str, deque] = {}
        self._last_temporal_state: Optional[np.ndarray] = None
        self._last_log_prob: float = 0.0
        self._last_value: float = 0.0
        
        if model_path:
            self.load(model_path)
        else:
            print("⚠️ [LACUNA] No model path provided. Running with UNTRAINED weights.")

    def _adapt_state(self, state: MarketState) -> np.ndarray:
        return state.to_features()[:29]

    def _get_temporal_state(self, asset: str, current_features: np.ndarray) -> np.ndarray:
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

    def act(self, state: MarketState) -> Action:
        features_29 = self._adapt_state(state)
        temporal_state = self._get_temporal_state(state.asset, features_29)
        
        features_t = torch.tensor(features_29.reshape(1, -1), dtype=torch.float32, device=self.device)
        temporal_t = torch.tensor(temporal_state.reshape(1, -1), dtype=torch.float32, device=self.device)
        
        if self.training:
            self.actor.train()
            probs = self.actor(features_t, temporal_t)
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor([action_idx], device=self.device)).item()
            
            self.critic.eval()
            with torch.no_grad():
                value = self.critic(features_t, temporal_t).item()
                
            self._last_log_prob = log_prob
            self._last_value = value
        else:
            self.actor.eval()
            with torch.no_grad():
                probs = self.actor(features_t, temporal_t)
                action_idx = torch.argmax(probs[0]).item()

        self._last_temporal_state = temporal_state
        return Action(action_idx)

    def store(self, state: MarketState, action: Action, reward: float, 
              next_state: MarketState, done: bool):
        # Normalize reward
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        self.reward_std = np.sqrt(((self.reward_count-1)*self.reward_std**2 + delta*(reward-self.reward_mean))/max(1, self.reward_count))
        norm_reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)

        # Get next temporal state
        next_features = self._adapt_state(next_state)
        next_temporal = self._get_temporal_state(next_state.asset, next_features)

        exp = Experience(
            state=self._adapt_state(state),
            temporal_state=self._last_temporal_state if self._last_temporal_state is not None else np.zeros(self.history_len * self.input_dim),
            action=action.value,
            reward=norm_reward,
            next_state=next_features,
            next_temporal_state=next_temporal,
            done=done,
            log_prob=self._last_log_prob,
            value=self._last_value
        )
        self.experiences.append(exp)
        if len(self.experiences) > self.buffer_size:
            self.experiences.pop(0)

    def update(self) -> Optional[Dict[str, float]]:
        if len(self.experiences) < self.buffer_size:
            return None

        # Prepare tensors
        states = torch.tensor(np.array([e.state for e in self.experiences]), dtype=torch.float32, device=self.device)
        temp_states = torch.tensor(np.array([e.temporal_state for e in self.experiences]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array([e.action for e in self.experiences]), dtype=torch.long, device=self.device)
        rewards = torch.tensor(np.array([e.reward for e in self.experiences]), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array([e.done for e in self.experiences], dtype=np.float32), dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(np.array([e.log_prob for e in self.experiences]), dtype=torch.float32, device=self.device)
        old_values = torch.tensor(np.array([e.value for e in self.experiences]), dtype=torch.float32, device=self.device)

        # GAE Advantages
        with torch.no_grad():
            next_features = torch.tensor(self.experiences[-1].next_state.reshape(1, -1), dtype=torch.float32, device=self.device)
            next_temp = torch.tensor(self.experiences[-1].next_temporal_state.reshape(1, -1), dtype=torch.float32, device=self.device)
            next_value = self.critic(next_features, next_temp).item()
            
            values = torch.cat([old_values, torch.tensor([next_value], device=self.device)])
            advantages = torch.zeros_like(rewards)
            last_gae = 0
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
                advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            returns = advantages + old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO Epochs
        metrics = {"policy_loss": 0, "value_loss": 0, "entropy": 0, "approx_kl": 0}
        
        for _ in range(self.n_epochs):
            indices = torch.randperm(len(self.experiences))
            for i in range(0, len(indices), self.batch_size):
                idx = indices[i:i+self.batch_size]
                
                # New log probs & values
                probs = self.actor(states[idx], temp_states[idx])
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()
                new_values = self.critic(states[idx], temp_states[idx]).squeeze()

                # Ratio & Policy Loss
                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Loss
                value_loss = F.mse_loss(new_values, returns[idx])

                # Total Loss
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Update Actor
                self.optimizer_actor.zero_grad()
                policy_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer_actor.step()

                # Update Critic
                self.optimizer_critic.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer_critic.step()

                # Tracking
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.item()
                metrics["approx_kl"] += (old_log_probs[idx] - new_log_probs).mean().item()

        # Average metrics
        num_updates = self.n_epochs * (len(self.experiences) // self.batch_size)
        for k in metrics: metrics[k] /= num_updates

        self.experiences = []
        return metrics

    def save(self, path: str):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'input_dim': self.input_dim,
            'reward_mean': self.reward_mean,
            'reward_std': self.reward_std,
            'reward_count': self.reward_count
        }, path if path.endswith('.pt') else f"{path}.pt")

    def load(self, path: str):
        print(f"🧬 [LACUNA] Loading weights from {path}...")
        try:
            if path.endswith(".safetensors"):
                from safetensors.torch import load_file
                checkpoint = load_file(path, device=str(self.device))
            else:
                checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            
            if isinstance(checkpoint, dict) and "actor_state_dict" in checkpoint:
                self.actor.load_state_dict(checkpoint["actor_state_dict"])
                self.critic.load_state_dict(checkpoint["critic_state_dict"])
                if 'reward_mean' in checkpoint:
                    self.reward_mean = checkpoint.get('reward_mean', 0)
                    self.reward_std = checkpoint.get('reward_std', 1)
                    self.reward_count = checkpoint.get('reward_count', 0)
            else:
                # Direct state dict load (common for HuggingFace/Safetensors)
                # Filter for actor keys
                actor_keys = {k: v for k, v in checkpoint.items() if k in self.actor.state_dict()}
                if actor_keys:
                    self.actor.load_state_dict(actor_keys, strict=False)
                    print(f"   - Loaded {len(actor_keys)} keys into Actor")
                
                # Filter for critic keys (if available)
                critic_keys = {k: v for k, v in checkpoint.items() if k in self.critic.state_dict()}
                if critic_keys:
                    self.critic.load_state_dict(critic_keys, strict=False)
                    print(f"   - Loaded {len(critic_keys)} keys into Critic")

            print("✅ [LACUNA] Weights loaded successfully!")
        except Exception as e:
            print(f"❌ [LACUNA] Failed to load: {e}")
            print("   (Ensure safetensors is installed and file path is correct)")
