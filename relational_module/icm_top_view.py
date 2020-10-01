import torch
import random
import torch.nn as nn
from collections import namedtuple
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class FWModel(nn.Module):

    def __init__(self, input_size):
        super(FWModel, self).__init__()
        self.lin_1 = nn.Linear(input_size+7, out_features=100)
        self.lin_2 = nn.Linear(in_features=100, out_features=input_size)
        return

    def forward(self, state, action):
        action_ = torch.zeros(action.shape[0], 7).to(device)
        indices = torch.stack(list((torch.arange(action_.shape[0]).to(device), action.squeeze())), dim=0)
        action_[indices[0, :], indices[1, :]] = 1.
        x = torch.cat((state, action_), dim=1)
        y = F.relu(self.lin_1(x))
        y = self.lin_2(y)
        return y


class InverseModel(nn.Module):
    def __init__(self, input_dim, output_size):
        super(InverseModel, self).__init__()
        self.linear1 = nn.Linear(input_dim+input_dim, 100)
        self.linear2 = nn.Linear(100, output_size)

    def forward(self, state, next_state):
        x = torch.cat((state, next_state), dim=1)
        y = F.relu(self.linear1(x))
        y = self.linear2(y)
        y = F.softmax(y, dim=1)
        return y


class EncoderCNN(nn.Module):

    def __init__(self, h, w, outputs):
        super(EncoderCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 16, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=2, stride=2)
        return

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)  # size N, 288
        return x


class ConvDQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(ConvDQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 16, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=2, stride=2)

        def conv2d_size_out(size, kernel_size=2, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 16
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        return self.head(x.view(x.size(0), -1))


class ReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32)
        return state_batch,action_batch, reward_batch, non_final_next_states, torch.tensor(batch.done, dtype=torch.float32), non_final_mask

    def __len__(self):
        return len(self.memory)

