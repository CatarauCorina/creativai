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
'MiniGrid-DoorKey-16x16-v0',
    'MiniGrid-KeyCorridorS3R3-v0',
    'MiniGrid-BlockedUnlockPickup-v0',
    'MiniGrid-Unlock-v0',
    'MiniGrid-Fetch-8x8-N3-v0',
    'MiniGrid-Fetch-5x5-N2-v0',
    'MiniGrid-Empty-8x8-v0',
    'MiniGrid-FourRooms-v0',
    'MiniGrid-SimpleCrossingS9N1-v0',
    'MiniGrid-MultiRoom-N2-S4-v0',
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))



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
        return state_batch,action_batch, reward_batch, non_final_next_states, torch.tensor(batch.done, dtype=torch.int64), non_final_mask

    def __len__(self):
        return len(self.memory)


class ConvDQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(ConvDQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


def init_env(env_name):
    env = gym.make(env_name)
    env.max_steps = 256
    window = Window('gym_minigrid - ' + env_name)
    return env, window


def process_frames(env):

    resize = T.Compose([T.ToPILImage(),
                        T.Resize((64,64), interpolation=Image.CUBIC),
                        T.Grayscale(),
                        T.ToTensor()])
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    screen = torch.tensor(screen)
    return resize(screen).unsqueeze(0).to(device)


def select_action(state, params, policy_net, n_actions, steps_done):
    sample = random.random()
    eps_threshold = params['eps_end'] + (params['eps_start'] - params['eps_end']) * \
        math.exp(-1. * steps_done / params['eps_decay'])
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1), steps_done
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), steps_done


def compute_td_loss(model, replay_buffer,params ,batch_size=32):

    state, action, reward, next_state, done, non_final_mask = replay_buffer.sample(batch_size)

    state = state.to(device)
    next_state = next_state.to(device)
    action = action.to(device)
    reward = reward.to(device)
    done = done.to(device)

    q_values = model(state)
    next_q_values = torch.zeros(batch_size, device=device)
    next_q_values[non_final_mask] = model(next_state).max(1)[0]

    q_value = q_values.gather(1, action).squeeze(1)
    expected_q_value = reward + params['gamma'] * next_q_values * (1 - done)

    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

    params['optimizer'].zero_grad()
    loss.backward()
    params['optimizer'].step()

    return loss


def train(env, model, replay_buffer, params, writer, num_episodes=10000):
    episode_durations = []

    steps_done = 0
    counter = 0
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
        state = process_frames(env)
        rew_ep = 0
        loss_ep = 0
        losses = []
        for t in count():
            # Select and perform an action
            action, steps_done = select_action(state, params, model, env.action_space.n, steps_done)

            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            rew_ep += reward.item()
            current_screen = process_frames(env)
            if not done:
                next_state = current_screen
            else:
                next_state = None

            # Store the transition in memory
            replay_buffer.push(state, action, reward, next_state, done)

            # Move to the next state
            prev_state = state
            state = next_state

            # Perform one step of the optimization (on the target network)
            if len(replay_buffer) > 32:
                loss_ep = compute_td_loss(model, replay_buffer, params)
                losses.append(loss_ep.item())
                writer.add_scalar('Loss/iter',loss_ep, counter)
                counter += 1

            if done:
                episode_durations.append(t + 1)
                writer.add_scalar('Reward/episode', rew_ep, i_episode)
                print(rew_ep)

                # fig, ax = plt.subplots()
                # plt.imshow(prev_state.squeeze(0).cpu().permute(1, 2, 0).squeeze(2).numpy(), interpolation='none')
                # plt.title('Last state')
                # writer.add_figure(tag='simple_dqn_last', figure=fig)
                #
                # plt.show()
                # writer.add_scalar('Ep duration', t+1, i_episode)
                loss = np.sum(losses)
                writer.add_scalar('Loss/episode', loss/(t+1), i_episode)

                break

    return


def main():
    params = {
        'batch_size': 128,
        'gamma': 0.999,
        'eps_start': 0.9,
        'eps_end':0.05,
        'eps_decay': 200,
        'target_update': 100
    }
    writer = SummaryWriter(f'deep_rl/dqn_conv_network_832_{device}_{params["gamma"]}_{params["eps_start"]}_{params["eps_end"]}_{params["eps_decay"]}')

    env, window = init_env(envs_to_run[0])
    env.see_through_walls = False
    env.agent_view_size = 3
    init_screen = process_frames(env)
    env.reset()

    _, _, screen_height, screen_width = init_screen.shape
    n_actions = env.action_space.n

    model = ConvDQN(screen_height, screen_width, n_actions).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    memory = ReplayBuffer(1000)
    params['optimizer'] = optimizer

    train(env, model, memory, params, writer)
    torch.save(model.state_dict(), 'dqn_conv_8x8.pt')
    return



if __name__ == '__main__':
    main()
