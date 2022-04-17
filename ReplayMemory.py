import collections
import random

import numpy as np

max_size = 100


class ReplayMemory(object):
    def __init__(self):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_, done = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_)
            done_batch.append(done)

        return np.array(obs_batch).astype(np.float32), \
            np.array(action_batch).astype(np.float32), \
            np.array(reward_batch).astype(np.float32), \
            np.array(next_obs_batch).astype(np.float32), \
            np.array(done_batch).astype(np.float32)

    def __len__(self):
        return len(self.buffer)

