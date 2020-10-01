import torch
import random
from collections import namedtuple
from itertools import count
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image

import matplotlib.pyplot as plt
from torch.autograd import Variable

from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
from relational_module.icm_top_view import FWModel, InverseModel, EncoderCNN, ConvDQN, ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

envs_to_run = [
    'MiniGrid-DoorKey-16x16-v0',
    'MiniGrid-KeyCorridorS3R3-v0',
    'MiniGrid-BlockedUnlockPickup-v0',
    'MiniGrid-Unlock-v0',
    'MiniGrid-Fetch-8x8-N3-v0',
    'MiniGrid-Fetch-5x5-N2-v0',
    'MiniGrid-Empty-8x8-v0',
    'MiniGrid-Unlock-v0',
     'MiniGrid-FourRooms-v0',
    'MiniGrid-SimpleCrossingS9N1-v0', 'MiniGrid-MultiRoom-N2-S4-v0',
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


def init_env(env_name):
    env = gym.make(env_name)
    env.max_steps = 256
    window = Window('gym_minigrid - ' + env_name)
    return env, window

def process_frames_env(env):

    resize = T.Compose([T.ToPILImage(),
                        T.Resize(32, interpolation=Image.CUBIC),
                        T.Grayscale(),
                        T.ToTensor()])
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    screen = torch.tensor(screen)
    return resize(screen).unsqueeze(0).to(device)


def process_frames(state):

    resize = T.Compose([T.ToPILImage(),
                        T.Grayscale(),
                        T.ToTensor()])
    screen = torch.tensor(state).T
    return resize(screen).to(device).unsqueeze(0)


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


def comput_icm_loss(dqn_loss, fw_loss, inverse_loss, params):
    loss = (1 - params['beta']) * inverse_loss
    loss = loss + params['beta']*fw_loss
    loss = loss.sum() / loss.flatten().shape[0]
    loss = loss + params['lambda'] * dqn_loss
    return loss


def icm_pass(state, action, next_state, non_final_mask, params):
    state_hat = params['encoder'](state)
    next_state_hat = params['encoder'](next_state)
    next_hat_pred = params['forward_model'](state_hat[non_final_mask, :].detach().to(device), action[non_final_mask,: ].detach().to(device))
    forward_pred_err =params['forward_loss'](next_hat_pred, next_state_hat.detach()).sum(dim=1).unsqueeze(dim=1)

    next_state_hat_all = torch.zeros(state_hat.shape, device=device)
    next_state_hat_all[non_final_mask, :] = next_state_hat
    pred_action = params['inverse_model'](state_hat[non_final_mask, :].to(device), next_state_hat_all[non_final_mask, :].to(device))
    inverse_pred_err =params['inverse_loss'](pred_action, action[non_final_mask, :].detach().flatten()).unsqueeze(dim=1)

    fw_pred_err = torch.zeros((params['batch_size'],1), device=device)
    fw_pred_err[non_final_mask, :] = forward_pred_err

    inv_pred_err = torch.zeros((params['batch_size'],1), device=device)
    inv_pred_err[non_final_mask, :] = inverse_pred_err
    return fw_pred_err, inv_pred_err


def optimize(params, use_extrinsic):
    state, action, reward, next_state, done, non_final_mask = params['replay_buffer'].sample(params['batch_size'])
    state = state.to(device)
    next_state = next_state.to(device)
    action = action.to(device)
    reward_batch = reward.unsqueeze(1).to(device)
    done = done.to(device)
    forward_pred_err, inverse_pred_err = icm_pass(state, action, next_state, non_final_mask, params)
    i_reward = (1. / params['eta']) * forward_pred_err
    i_reward = torch.tensor(i_reward, dtype=torch.float32).to(device)
    reward = i_reward.detach()
    if use_extrinsic:
        reward += reward_batch

    q_values = params['dqn_model'](state)
    reward_target = q_values.clone()

    next_q_values = torch.zeros(params['batch_size'], device=device)
    next_q_values[non_final_mask] = params['dqn_model'](next_state).max(1)[0]
    reward = reward + params['gamma'] * next_q_values.unsqueeze(1)

    indices = torch.stack(list((torch.arange(action.shape[0]).to(device), action.squeeze())), dim=0)
    reward_target[indices[0, :], indices[1, :]] = reward.squeeze()

    q_loss = 1e5 * params['dqn_loss'](F.normalize(q_values), F.normalize(reward_target.detach()))
    return forward_pred_err, inverse_pred_err, q_loss


def train(env, params, writer, num_episodes=10000, use_extrinsic=True):
    episode_durations = []
    running_reward = 0
    avg_length = 0
    steps_done = 0
    counter = 0
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        state_env = env.reset()
        state = process_frames_env(env)
        rew_ep = 0
        loss_ep = 0
        losses = []
        for t in count():
            action, steps_done = select_action(
                state, params, params['dqn_model'], env.action_space.n, steps_done
            )
            new_state_env, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device, dtype=torch.float32)
            running_reward += reward
            rew_ep += reward.item()
            prev_state = state
            current_screen = process_frames_env(env)
            if not done:
                next_state = current_screen
            else:
                next_state = None

            # Store the transition in memory
            params['replay_buffer'].push(state, action, reward, next_state, done)
            state = next_state

            # Perform one step of the optimization (on the target network)
            if len(params['replay_buffer']) > 32:
                forward_pred_err, inverse_pred_err, dqn_loss = optimize(params, use_extrinsic)
                loss = comput_icm_loss(dqn_loss, forward_pred_err, inverse_pred_err, params)
                loss_list = (dqn_loss.mean(), forward_pred_err.flatten().mean(), inverse_pred_err.flatten().mean())
                losses.append(loss)
                writer.add_scalar('Loss/iter', loss.mean().item(), counter)
                # writer.add_scalar('Loss/dqn', dqn_loss.mean().item(), counter)
                # writer.add_scalar('Loss/fw_pred', loss_list[1].item(), counter)
                # writer.add_scalar('Loss/inv_pred', loss_list[2].item(), counter)
                # print("Epoch {}, Loss: {}".format(i_episode, loss))
                # print("Forward loss: {} \n Inverse loss: {} \n Qloss: {}".format( \
                #     forward_pred_err.mean(), inverse_pred_err.mean(), dqn_loss.mean()))
                params['optimizer'].zero_grad()
                loss.backward()
                params['optimizer'].step()
                counter += 1

            if done:
                print(done)
                episode_durations.append(t + 1)
                writer.add_scalar('Reward/episode', rew_ep, i_episode)
                # fig, ax = plt.subplots()
                # plt.imshow(prev_state.squeeze(0).cpu().permute(1, 2, 0).squeeze(2).numpy(), interpolation='none')
                # plt.title('Last state')
                # writer.add_figure(tag='simple_dqn_last', figure=fig)
                #
                # plt.show()
                writer.add_scalar('Ep duration', t+1, i_episode)

                loss = np.sum(losses)
                writer.add_scalar('Loss/episode', loss.mean().item()/(t+1), i_episode)

                break
        avg_length += t
        if i_episode % params['log_interval'] == 0:
            avg_length = int(avg_length / params['log_interval'])
            running_reward = int((running_reward / params['log_interval']))

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            writer.add_scalar('Int/avg_length_ep', avg_length, i_episode)
            writer.add_scalar('Int/running_reward', running_reward, i_episode)
            running_reward = 0
            avg_length = 0

    return


def run_icm_model():
    env, window = init_env(envs_to_run[0])
    env.see_through_walls = False
    env.agent_view_size = 3
    init_screen = process_frames_env(env)
    env.reset()

    _, _, screen_height, screen_width = init_screen.shape
    n_actions = env.action_space.n
    memory = ReplayBuffer(2000)

    conv_dqn = ConvDQN(screen_height, screen_width, n_actions).to(device)
    encoder = EncoderCNN(screen_height, screen_width, n_actions).to(device)
    forward_model = FWModel(256).to(device)
    inverse_model = InverseModel(256, n_actions).to(device)
    forward_loss = nn.MSELoss(reduction='none')
    inverse_loss = nn.CrossEntropyLoss(reduction='none')
    dqn_loss = nn.MSELoss()
    all_model_params = list(conv_dqn.parameters()) + list(encoder.parameters()) + list(inverse_model.parameters())
    optimizer = optim.Adam(lr=0.0001, params=all_model_params)
    params = {
        'batch_size': 32,
        'gamma': 0.999,
        'eps_start': 0.9,
        'eta':1.0,
        'lambda': 0.1,
        'log_interval':100,
        'beta': 0.2,
        'eps_end': 0.05,
        'eps_decay': 200,
        'target_update': 100,
        'dqn_model': conv_dqn,
        'forward_model': forward_model,
        'inverse_model': inverse_model,
        'encoder': encoder,
        'forward_loss': forward_loss,
        'inverse_loss': inverse_loss,
        'dqn_loss': dqn_loss,
        'replay_buffer': memory,
        'optimizer': optimizer
    }
    writer = SummaryWriter(
        f'deep_rl/dqn_icm_top_view_no_scale3kbuff_{params["gamma"]}_{params["eps_start"]}_{params["eps_end"]}_{params["eps_decay"]}')

    train(env, params, writer)
    torch.save(
        {
            'forward_model': forward_model.state_dict(),
            'inverse_model': inverse_model.state_dict(),
            'encoder': encoder.state_dict(),
            'dqn_model': conv_dqn.state_dict(),

        }, 'dqn_conv_icm_top_no_scale3k_buff.pt')
    return


def main():
    run_icm_model()
    return

if __name__ == '__main__':
    main()
