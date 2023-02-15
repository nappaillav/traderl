from collections import namedtuple, deque
import random

"""
Replay buffer
"""
transition = namedtuple('transition', 
                       ('state', 'action', 'next_state', 'reward', 'info'))

class Replaybuffer:
    def __init__(self, window, capacity):
        self.window = window
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """
        save transition
        """
        self.memory.append(transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
