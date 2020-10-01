import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable

from gym_minigrid.wrappers import *
from collections import deque

from gym_minigrid.window import Window
from torch.utils.tensorboard import SummaryWriter

envs_to_run = [
    'MiniGrid-Empty-8x8-v0',
    'MiniGrid-Unlock-v0',
    'MiniGrid-FourRooms-v0',
    'MiniGrid-SimpleCrossingS9N1-v0', 'MiniGrid-MultiRoom-N2-S4-v0',
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TransitionConcepts = namedtuple('TransitionConcepts', ('state', 'action', 'reward', 'next_state', 'done','concepts','proposals'))


class ReplayBufferConcepts(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = TransitionConcepts(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = TransitionConcepts(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32)
        return state_batch, action_batch, reward_batch, non_final_next_states, torch.tensor(batch.done,
                                                                                            dtype=torch.float32), non_final_mask

    def __len__(self):
        return len(self.memory)

