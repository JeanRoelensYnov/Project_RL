import torch
import torch.nn as nn
from carla_env import CarlaEnv
import numpy as np

import csv
from datetime import datetime

class ActorCritic(nn.Module):
    def __init__(self, obs_dim:int, action_dim: int, hidden_dim=64):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.actor_mean = nn.Linear(hidden_dim, action_dim)

        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        x = self.shared(obs)

        action_mean = self.actor_mean(x)
        action_std = self.actor_log_std.exp()

        value = self.critic(x)

        return action_mean, action_std, value

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def store(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

class PPO:
    def __init__(self, env : CarlaEnv, lr=3e-4, gamma=0.99, epsilon=0.2, n_steps=1024):
        # Prepare log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"logs/training_{timestamp}.csv"

        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'timesteps', 'avg_reward', 'num_episodes'])

        # PPO below
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon 
        self.n_steps = n_steps

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.policy = ActorCritic(obs_dim=obs_dim, action_dim=action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.buffer = RolloutBuffer()

    def collect_rollout(self):
        state, _ = self.env.reset()

        self.buffer.clear()
        episode_rewards = []
        current_episode_reward = 0

        for _ in range(self.n_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                action_mean, action_std, value = self.policy(state_tensor)
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            action_clipped = action.squeeze(0).numpy()
            action_clipped = np.clip(action_clipped, -1, 1)

            next_state, reward, terminated, truncated, _ = self.env.step(action_clipped)
            current_episode_reward += reward
            done = terminated or truncated
            self.buffer.store(
                state=state,
                action=action.squeeze(0).numpy(),
                reward=reward,
                done=done,
                log_prob=log_prob.item(),
                value=value.item()
            )

            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                state, _ = self.env.reset()
            else:
                state = next_state
        return episode_rewards # Used for logging


    def compute_advantages(self):
        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones

        advantages = []
        advantage = 0


        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            if dones[t]:
                next_value = 0
                advantage = 0
            
            td_error = rewards[t] + self.gamma * next_value - values[t]

            advantage = td_error + self.gamma * 0.95 * advantage
            advantages.insert(0, advantage)
        
        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns = advantages + torch.FloatTensor(values)

        return advantages, returns

    def update(self):
        states = torch.FloatTensor(np.array(self.buffer.states))
        actions = torch.FloatTensor(np.array(self.buffer.actions))
        old_log_probs = torch.FloatTensor(self.buffer.log_probs)

        advantages, returns = self.compute_advantages()

        for _ in range(4):
            action_mean, action_std, values = self.policy(states)
            values = values.squeeze(-1)
            
            dist = torch.distributions.Normal(action_mean, action_std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            
            ratio = (new_log_probs - old_log_probs).exp()
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = (returns - values).pow(2).mean()
            entropy = dist.entropy().mean()

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

    def learn(self, total_timesteps : int):
        timesteps_done = 0
        episode = 0

        while timesteps_done < total_timesteps:
            episode_rewards = self.collect_rollout()
            timesteps_done += self.n_steps

            self.update()

            episode += 1
            avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
            print(f"Episode {episode} | Timesteps: {timesteps_done}/{total_timesteps} | Avg Reward: {avg_reward:.1f}")
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([episode, timesteps_done, avg_reward, len(episode_rewards)])

        self.save("models/ppo_carla.pth")

    def save(self, path):
        torch.save(self.policy.state_dict(), path)
    
    def load(self, path):
        self.policy.load_state_dict(torch.load(path))