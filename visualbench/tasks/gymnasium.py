import time
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from visualbench.benchmark import Benchmark


def _ensure_float(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu().item()
    return x

# 2. Define the Policy Network (Simple MLP)
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        action_probs = torch.softmax(self.fc2(x), dim=-1)
        return action_probs

# 3. Define the Agent (Policy Gradient - REINFORCE style)
class Agent(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim = 64):
        super().__init__()
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)

        self.gamma = 0.99
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0) # Add batch dimension
        probs = self.policy_net(state_tensor)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        self.log_probs.append(log_prob) # Store log prob for policy update
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def get_policy_loss(self):
        R = 0
        policy_loss = []
        returns = deque() # Efficient way to calculate returns in reverse

        # Calculate discounted returns
        for r in self.rewards[::-1]: # Iterate rewards in reverse
            R = r + self.gamma * R
            returns.appendleft(R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8) # Normalize returns

        for log_prob, R_t in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R_t) # REINFORCE loss

        policy_loss = torch.cat(policy_loss).sum() # Sum up losses

        self.log_probs = []
        self.rewards = []

        return policy_loss

class Gymnasium(Benchmark):
    def __init__(self, hidden_dim = 64, env_name = "CartPole-v1", avg_reward_len = 100):
        super().__init__()
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0] # type:ignore
        self.action_dim = self.env.action_space.n # type:ignore

        self.agent = Agent(self.state_dim, self.action_dim, hidden_dim=hidden_dim)
        self.avg_rewards_deque = deque(maxlen=avg_reward_len) # For moving average


    def get_loss(self):
        state, _ = self.env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = self.agent.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.agent.store_reward(reward)
            state = next_state
            episode_reward += reward # type:ignore

        loss = self.agent.get_policy_loss()
        self.avg_rewards_deque.append(_ensure_float(episode_reward))

        return loss, {"reward": episode_reward, "average reward": np.mean(self.avg_rewards_deque)}


# g = Gymnasium()
# g.run(torch.optim.SGD(g.parameters(), 1e-2), 200)