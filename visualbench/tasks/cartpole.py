import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..benchmark import Benchmark


class _PolicyNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CartPole(Benchmark):
    def __init__(
        self,
        hidden_dim: int = 64,
        gamma: float = 0.99,
        baseline: bool = True,
        render_resolution: tuple[int, int] = (400, 600),
        max_episode_steps: int = 500,
        n_episodes_per_batch: int = 4,
    ):
        self.gamma = gamma
        self.baseline = baseline
        self.max_episode_steps = max_episode_steps
        self.n_episodes_per_batch = n_episodes_per_batch
        self.resolution = render_resolution
        self._running_reward = 0.0
        self._reward_history = []
        self._best_reward = 0.0
        self._current_frame = None
        self._synthetic_state = None

        obs_dim = 4
        action_dim = 2

        super().__init__()

        self.policy = _PolicyNet(obs_dim, hidden_dim, action_dim)

    def _run_episode(self) -> tuple[list[torch.Tensor], list[float], list[torch.Tensor]]:
        """Run a single episode and return (states, rewards, log_probs)."""
        states = []
        rewards = []
        log_probs = []

        state = self._reset_synthetic()

        for t in range(self.max_episode_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # (1, obs_dim)

            # Get action probabilities
            logits = self.policy(state_tensor)
            probs = F.softmax(logits, dim=-1)

            # Sample action
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Take action
            next_state, reward, done = self._step_synthetic(state, int(action.item()))

            states.append(state_tensor)
            rewards.append(reward)
            log_probs.append(log_prob)

            state = next_state

            if done:
                break

        return states, rewards, log_probs

    def _reset_synthetic(self) -> np.ndarray:
        """Reset synthetic CartPole environment."""
        # Random initial state (similar to gymnasium)
        self._synthetic_state = self.rng.numpy.uniform(-0.05, 0.05, size=4)
        return self._synthetic_state

    def _step_synthetic(self, state: np.ndarray, action: int) -> tuple[np.ndarray, float, bool]:
        """Simulate a CartPole step (simplified physics)."""
        # Simplified CartPole dynamics
        # action: 0 = left, 1 = right
        force = 10.0 if action == 1 else -10.0

        # Unpack state
        x, x_dot, theta, theta_dot = state

        # Physics parameters (simplified)
        gravity = 9.8
        cart_mass = 1.0
        pole_mass = 0.1
        pole_length = 0.5
        dt = 0.02

        # Apply force and gravity
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        total_mass = cart_mass + pole_mass
        pole_mass_length = pole_mass * pole_length

        acc_theta = (gravity * sin_theta - cos_theta * (force + pole_mass_length * theta_dot**2 * sin_theta) / total_mass) / \
                    (pole_length * (4.0/3.0 - pole_mass * cos_theta**2 / total_mass))
        acc_x = (force + pole_mass_length * (theta_dot**2 * sin_theta - acc_theta * cos_theta)) / total_mass

        # Euler integration
        x = x + dt * x_dot
        x_dot = x_dot + dt * acc_x
        theta = theta + dt * theta_dot
        theta_dot = theta_dot + dt * acc_theta

        self._synthetic_state = np.array([x, x_dot, theta, theta_dot])

        # Check termination conditions
        done = (
            x < -2.4 or x > 2.4 or
            theta < -0.2095 or theta > 0.2095  # ~12 degrees
        )

        # Always give +1 reward (standard CartPole)
        reward = 1.0

        return self._synthetic_state, reward, done

    def _compute_returns(self, rewards: list[float]) -> torch.Tensor:
        """Compute discounted returns with optional baseline."""
        returns = []
        R = 0.0

        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns (simple baseline)
        if self.baseline and len(returns) > 1:
            mean = returns.mean()
            std = returns.std()
            if std > 1e-6:
                returns = (returns - mean) / std

        return returns

    def get_loss(self) -> torch.Tensor:
        """Run multiple episodes and compute policy gradient loss."""
        all_log_probs = []
        all_returns = []
        episode_rewards = []

        for _ in range(self.n_episodes_per_batch):
            states, rewards, log_probs = self._run_episode()
            returns = self._compute_returns(rewards)

            all_log_probs.extend(log_probs)
            all_returns.append(returns)
            episode_rewards.append(sum(rewards))

        # Update running reward for logging
        batch_reward = np.mean(episode_rewards)
        self._running_reward = 0.99 * self._running_reward + 0.01 * batch_reward
        self._reward_history.append(batch_reward)
        self._best_reward = max(self._best_reward, batch_reward)

        # Concatenate all returns
        all_returns = torch.cat(all_returns)

        # Policy gradient loss: -sum(log_prob * return)
        log_probs = torch.cat(all_log_probs)
        loss = -(log_probs * all_returns.detach()).mean()

        return loss

    def reset(self):
        """Reset the benchmark and environment."""
        super().reset()
        self._running_reward = 0.0
        self._reward_history = []
        self._best_reward = 0.0
        return self
