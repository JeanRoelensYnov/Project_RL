
import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple, List, Dict, Any



class QNetwork(nn.Module):

    def __init__(self, in_channels: int, state_dim: int, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            conv_out = self.conv(dummy)
            conv_flatten = int(np.prod(conv_out.shape[1:]))


        self.fc = nn.Sequential(
            nn.Linear(conv_flatten + state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, img: torch.Tensor, state: torch.Tensor) -> torch.Tensor:

        x = self.conv(img)
        x = x.reshape(x.size(0), -1)
        x = torch.cat([x, state], dim=1)
        q = self.fc(x)
        return q



class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, obs: Dict[str, Any], action: int, reward: float, next_obs: Dict[str, Any], done: bool):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)
        return obs_batch, np.array(action_batch, dtype=np.int64), np.array(reward_batch, dtype=np.float32), next_obs_batch, np.array(done_batch, dtype=np.uint8)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        observation_space,
        action_space,
        device: str = 'gpu',
        lr: float = 1e-4,
        gamma: float = 0.99,
        batch_size: int = 32,
        buffer_size: int = 100_000,
        min_replay_size: int = 2000,
        target_update_freq: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.05,
        epsilon_decay: int = 100_000,
    ):

        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.min_replay_size = min_replay_size

        self.actions = [
            [0.0, 0.6, 0.0],   # straight
            [0.0, 1.0, 0.0],   # full throttleA
            [0.0, 0.0, 0.8],   # brake
            [-0.5, 0.5, 0.0],  # left
            [0.5, 0.5, 0.0],   # right
            [-0.2, 0.6, 0.0],  # slight left
            [0.2, 0.6, 0.0],   # slight right
            [-0.8, 0.3, 0.0],  # hard left slow
            [0.8, 0.3, 0.0],   # hard right slow
        ]
        self.n_actions = len(self.actions)


        in_channels = 3 + 1 + 1
        state_dim = observation_space['state'].shape[0]

        self.q_net = QNetwork(in_channels, state_dim, self.n_actions).to(self.device)
        self.target_q_net = QNetwork(in_channels, state_dim, self.n_actions).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.buffer = ReplayBuffer(capacity=buffer_size)

        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.total_steps = 0

        self.loss_fn = nn.MSELoss()
        self.update_count = 0

    def action_from_index(self, idx: int) -> List[float]:
        return self.actions[int(idx)]

    def preprocess(self, obs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:

        rgb = obs['rgb'].astype(np.float32) / 255.0
        depth = obs['depth'].astype(np.float32)
        if depth.ndim == 2:
            depth = depth[..., np.newaxis]
        lidar = obs['lidar_bev'].astype(np.float32)
        if lidar.ndim == 2:
            lidar = lidar[..., np.newaxis]
        if lidar.max() > 1.0:
            lidar = lidar / 255.0

        stacked = np.concatenate([rgb, depth, lidar], axis=2)
        img = np.transpose(stacked, (2, 0, 1)).astype(np.float32)

        state = np.array(obs['state'], dtype=np.float32)
        return img, state


    def store_transition(self, obs, action_idx: int, reward: float, next_obs, done: bool):
        self.buffer.push(obs, int(action_idx), float(reward), next_obs, bool(done))
        self.total_steps += 1

    def current_epsilon(self) -> float:
        # Linear decay
        fraction = min(float(self.total_steps) / max(1, self.epsilon_decay), 1.0)
        return self.epsilon_start + fraction * (self.epsilon_final - self.epsilon_start)

    def select_action(self, obs: Dict[str, Any], evaluate: bool = False) -> int:
        eps = 0.0 if evaluate else self.current_epsilon()
        if random.random() < eps:
            return random.randrange(self.n_actions)

        img, state = self.preprocess(obs)
        img_t = torch.from_numpy(img).unsqueeze(0).to(self.device)
        state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.q_net(img_t, state_t)
            action = int(torch.argmax(q, dim=1).item())
        return action

    def sample_batch(self):
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = self.buffer.sample(self.batch_size)
        imgs = []
        states = []
        next_imgs = []
        next_states = []
        for ob, nob in zip(obs_batch, next_obs_batch):
            img, state = self.preprocess(ob)
            nimg, nstate = self.preprocess(nob)
            imgs.append(img)
            states.append(state)
            next_imgs.append(nimg)
            next_states.append(nstate)

        imgs = torch.from_numpy(np.stack(imgs)).to(self.device)
        states = torch.from_numpy(np.stack(states)).to(self.device)
        actions = torch.from_numpy(action_batch).to(self.device)
        rewards = torch.from_numpy(reward_batch).to(self.device)
        next_imgs = torch.from_numpy(np.stack(next_imgs)).to(self.device)
        next_states = torch.from_numpy(np.stack(next_states)).to(self.device)
        dones = torch.from_numpy(done_batch.astype(np.uint8)).to(self.device)

        return imgs, states, actions, rewards, next_imgs, next_states, dones

    def train_step(self) -> float:
        if len(self.buffer) < max(self.min_replay_size, self.batch_size):
            return 0.0

        imgs, states, actions, rewards, next_imgs, next_states, dones = self.sample_batch()

        q_values = self.q_net(imgs, states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_online = self.q_net(next_imgs, next_states)
            next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
            next_q_target = self.target_q_net(next_imgs, next_states)
            next_q = next_q_target.gather(1, next_actions).squeeze(1)
            target_q = rewards + (1.0 - dones.float()) * self.gamma * next_q

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    def save(self, path: str):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_q_net': self.target_q_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
        }, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(data['q_net'])
        self.target_q_net.load_state_dict(data.get('target_q_net', data['q_net']))
        # self.total_steps = data.get('total_steps', 0)

        if 'optimizer' in data:
            self.optimizer.load_state_dict(data['optimizer'])

    def get_epsilon(self) -> float:
        return self.current_epsilon()


