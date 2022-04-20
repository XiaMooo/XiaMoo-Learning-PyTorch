from collections import deque, namedtuple
import random

Transition = namedtuple('Transition',
                        (
                            'state', 'action', 'reward', 'next_state', 'done'
                        ))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)

    def push(self, *args):  # state, action, next_state, reward
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        return mini_batch

    def __len__(self):
        return len(self.buffer)
