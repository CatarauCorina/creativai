import gym
import sys
import os
import math
import torch
import random
import numpy as np
import torch.nn as nn
from einops import rearrange


import torchvision.transforms as T
from PIL import Image
from gym_minigrid.window import Window
from relational_module.double_dqn_utils import ConvDQN
from concept_gathering.concept_buffer import ReplayBufferConcepts
from concept_learning_matching.PCA.model import Net
from relational_module.graph_data_utils import GraphDataUtils
from torch_geometric.data import Data, InMemoryDataset, extract_zip


import torch.optim as optim
from itertools import count
from torch.utils.tensorboard import SummaryWriter

from torch.autograd import Variable
from concept_gathering.init_and_run_env import MiniGridEnv

envs_to_run = [
    'MiniGrid-DoorKey-16x16-v0',
    'MiniGrid-Unlock-v0',
    'MiniGrid-Fetch-8x8-N3-v0',
    'MiniGrid-Fetch-5x5-N2-v0',
    'MiniGrid-Empty-8x8-v0',
    'MiniGrid-FourRooms-v0',
    'MiniGrid-SimpleCrossingS9N1-v0',
    'MiniGrid-MultiRoom-N2-S4-v0',
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReducerNet(nn.Module):

    def __init__(self, params):
        super(ReducerNet, self).__init__()
        self.params = params
        self.concept_dim = params['concept_dim']
        self.embed_reduce = nn.Linear(self.concept_dim,500)
        return

    def forward(self, x):
        x = self.embed_reduce(x)
        return x


class RelationalConceptGraphLearning(nn.Module):

    def __init__(self, params):
        super(RelationalConceptGraphLearning, self).__init__()
        self.params = params
        self.concept_embeder = MiniGridEnv()
        self.concept_dim = params['concept_dim']
        self.embed_reduce = params['reducer']
        self.graph_model = params['graph_model']
        self.proj_shape = (500, params['node_size']*params['nr_heads'])
        self.k_proj = nn.Linear(*self.proj_shape)
        self.q_proj = nn.Linear(*self.proj_shape)
        self.v_proj = nn.Linear(*self.proj_shape)
        self.concept_nr = params['N']

        self.norm_shape = (params['N'], params['node_size'])
        self.k_norm = nn.LayerNorm(self.norm_shape, elementwise_affine=True)
        self.q_norm = nn.LayerNorm(self.norm_shape, elementwise_affine=True)
        self.v_norm = nn.LayerNorm(self.norm_shape, elementwise_affine=True)

        self.linear1 = nn.Linear(params['node_size']*params['nr_heads'],params['node_size']*params['nr_heads'])
        self.norm1 = nn.LayerNorm([params['N'],  params['node_size']*params['nr_heads']], elementwise_affine=False)
        self.linear2 = nn.Linear(params['node_size']*params['nr_heads'], params['out_dim'])
        self.att_map = None

        self.additive_k_lin = nn.Linear(params['node_size'], params['N'])
        self.additive_q_lin = nn.Linear(params['node_size'], params['N'])
        self.additive_attention_lin = nn.Linear(params['N'], params['N'])

    def forward(self, x=None, already_extracted=False, all_concepts=None):
        if not already_extracted:
            img, proposals, concepts, concept_embeddings, masks_roi = self.concept_embeder.roi_extractor.get_concept_proposals_batch(x)
            temp_concepts, template_concept_embeddings, coords, labels, masks_templates = self.concept_embeder.template_extractor.find_objects_in_frame_run(x)

            in_concepts = torch.stack(concept_embeddings).unsqueeze(0).squeeze(2).to(device)
            if len(template_concept_embeddings) > 0:
                if len(template_concept_embeddings) + len(concept_embeddings) > self.concept_nr:
                    allowed_concepts = self.concept_nr - len(concept_embeddings)
                    template_concept_embeddings = template_concept_embeddings[:allowed_concepts]
                    concepts = concepts[:allowed_concepts]
                    coords = coords[:allowed_concepts]
                in_concepts_template = torch.stack(template_concept_embeddings).unsqueeze(0).squeeze(2).to(device)
                all_concepts = torch.cat([in_concepts, in_concepts_template], dim=1)
            else:
                all_concepts = in_concepts

            x = all_concepts
            if x.shape[1] < self.concept_nr:
                zeros = torch.zeros(self.concept_nr-x.shape[1],x.shape[2]).unsqueeze(0).to(device)
                x = torch.cat([x, zeros], dim=1).to(device)
            x = self.embed_reduce(x)
        else:
            x = all_concepts
        if x.shape[1] != self.concept_nr:
            print(x.shape)

        K = rearrange(self.k_proj(x), "b n (head d) -> b head n d", head=self.params['nr_heads'])
        K = self.k_norm(K)

        Q = rearrange(self.q_proj(x), "b n (head d) -> b head n d", head=self.params['nr_heads'])
        Q = self.q_norm(Q)

        V = rearrange(self.v_proj(x), "b n (head d) -> b head n d", head=self.params['nr_heads'])
        V = self.v_norm(V)

        if self.params['attention'] == 'dot':
            A = torch.einsum('bhfe,bhge->bhfg', Q, K)
            A = A / np.sqrt(self.params['node_size'])
            A = torch.nn.functional.softmax(A, dim=3)
        else:
            A = torch.nn.functional.elu(self.additive_q_lin(Q) + self.additive_k_lin(K))
            A = self.additive_attention_lin(A)
            A = torch.nn.functional.softmax(A, dim=3)

        with torch.no_grad():
            self.att_map = A.clone()
        # import matplotlib.pyplot as plt
        # plt.imshow(A.clone().detach().cpu()[0][0])
        # plt.show()
        E = torch.einsum('bhfc,bhcd->bhfd', A, V)
        E = rearrange(E, 'b head n d -> b n (head d)')
        E = self.linear1(E)
        E = torch.relu(E)
        E = self.norm1(E)
        E = E.max(dim=1)[0]
        y = self.linear2(E)
        y = torch.nn.functional.elu(y)
        return y, A.clone().detach().cpu()


def init_env(env_name):
    env = gym.make(env_name)
    env.max_steps = 256
    window = Window('gym_minigrid - ' + env_name)
    return env, window


def process_frames(env):

    resize = T.Compose([T.ToPILImage(),
                        # T.Resize((16,16), interpolation=Image.CUBIC),
                        T.Grayscale()])
                        # T.ToTensor()])
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    screen = torch.tensor(screen)
    return resize(screen)


def apply_pil_changes(frame):
    crt_frame = frame.convert('RGB')
    crt_frame = np.array(crt_frame)
    return crt_frame


def select_action(state, params, policy_net, n_actions, steps_done, ep):
    sample = random.random()
    eps_threshold = params['eps_end'] + (params['eps_start'] - params['eps_end']) * \
        math.exp(-1. * steps_done / params['eps_decay'])
    steps_done += 1
    state = apply_pil_changes(state)
    if sample > eps_threshold:
        with torch.no_grad():
            res, attention =  policy_net(state)
            # if ep % 200 == 0 and ep!=0:
            #     import matplotlib.pyplot as plt
            #     plt.imshow(attention[0][0])
            #     plt.show()
            return res.max(1)[1].view(1, 1), steps_done
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), steps_done


def select_action_50(state, params, policy_net, n_actions, steps_done):
    with torch.no_grad():
        state = apply_pil_changes(state)
        pred = policy_net(state)[0].max(1)[1].view(1, 1), steps_done
    if np.random.rand() < 0.5:  # F
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), steps_done
    return pred, steps_done


def compute_td_loss(model, replay_buffer, params, target_model=None):
    state, action, reward, next_state, done, non_final_mask, nr_templates, nr_proposals, labels, masks_roi, masks_template = replay_buffer.sample(params['batch_size'])

    state = state.to(device)
    next_state = next_state.to(device)
    action = action.to(device)
    reward = reward.to(device)
    done = done.to(device)

    q_values = model(x=None, already_extracted=True, all_concepts=state)[0]
    next_q_values = torch.zeros(params['batch_size'], device=device)

    next_q_values[non_final_mask] = target_model(x=None, already_extracted=True, all_concepts=next_state)[0].max(1)[0]
    q_value = q_values.gather(1,action)
    expected_q_value = reward + params['gamma'] * next_q_values * (1 - done)

    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

    params['optimizer'].zero_grad()
    loss.backward()
    params['optimizer'].step()
    # print(f'Replay buffer dim {len(replay_buffer)}')
    # print(f'Replay buffer size {get_size(replay_buffer)}')
    # print(f'Model size {get_size(model)}')
    # print(f'Target model size {get_size(target_model)}')



    return loss


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def run_model_pca_gm(inputs, model_gr):

    emb_src = inputs.x1.to(device)
    emb_tgt = inputs.x2.to(device)
    perm_mat = inputs.train_y.unsqueeze(0).to(device)
    a_src = inputs.edge_attr1.to(device)
    a_tgt = inputs.edge_attr2.to(device)
    pos_matches = inputs.all_pos
    n1_gt = inputs.n1_gt.to(device)
    n2_gt = inputs.n2_gt.to(device)
    s_pred, new_emb_src, new_emb_tgt = model_gr(emb_src, emb_tgt, a_src, a_tgt)


    return new_emb_src, new_emb_tgt, s_pred


def create_graph_datapoint(
        nodes_g1, action_g1, reward_g1, nr_templates_g1, nr_proposals_g1, template_labels_g1,masks_temp_g1,masks_roi_g1,
        nodes_g2, action_g2, reward_g2, nr_templates_g2, nr_proposals_g2, template_labels_g2, masks_roi_g2
):
    ds = GraphDataUtils()

    x1 = nodes_g1
    x2 = nodes_g2
    adj_1 = ds.init_adj_matrix(x1)
    adj_2 = ds.init_adj_matrix(x2)
    matches_temp_roi = ds.find_coords_matches(masks_temp_g1, nr_proposals_g1, masks_roi_g1)

    train_y, all_pos = ds.process_permutation_matrix(template_labels_g1, template_labels_g2, nodes_g1, nodes_g2,
                                                       nr_proposals_g1, nr_proposals_g2)
    if len(matches_temp_roi) > 0:
        for (i, j) in matches_temp_roi:
            train_y[i, j] = 1
            all_pos.append([i, j])
            if i != j:
                train_y[j, i] = 1
                all_pos.append([j, i])

    edge_index1, edge_index2 = ds.process_graph(nodes_g1, nodes_g2)
    if len(all_pos) != 0:
        data = Data(x1=x1, edge_index1=edge_index1, x2=x2,
                    edge_index2=edge_index2,
                    edge_attr1=adj_1, edge_attr2=adj_2, train_y=train_y,
                    all_pos=all_pos,
                    n1_gt=torch.tensor([nr_proposals_g1 + nr_templates_g1]),
                    n2_gt=torch.tensor([nr_proposals_g2 + nr_templates_g2]))
        return data


def save_embeddings_only(state, model, reducer_model):
    img, proposals, concepts, concept_embeddings, masks_roi = model.concept_embeder.roi_extractor.get_concept_proposals_batch(state)
    temp_concepts, template_concept_embeddings, coords, labels, masks_templates = model.concept_embeder.template_extractor.find_objects_in_frame_run(state)

    # plt.imshow(img)

    in_concepts = torch.stack(concept_embeddings).unsqueeze(0).squeeze(2).to(device)
    if len(template_concept_embeddings) > 0:
        if len(template_concept_embeddings) + len(concept_embeddings) > model.concept_nr:
            allowed_concepts = model.concept_nr - len(concept_embeddings)
            template_concept_embeddings = template_concept_embeddings[:allowed_concepts]
            concepts = concepts[:allowed_concepts]
            labels = labels[:allowed_concepts]
            masks_templates = masks_templates[:allowed_concepts]
            coords = coords[:allowed_concepts]
        in_concepts_template = torch.stack(template_concept_embeddings).unsqueeze(0).squeeze(2).to(device)
        all_concepts = torch.cat([in_concepts, in_concepts_template], dim=1)
    else:
        all_concepts = in_concepts



    # img, proposals, concepts, concept_embeddings, masks_roi = model.concept_embeder.roi_extractor.get_concept_proposals_batch(state)
    # temp_concepts, template_concept_embeddings, coords, labels, masks_templates = model.concept_embeder.template_extractor.find_objects_in_frame_run(state)
    #
    # # plt.imshow(img)
    #
    # in_concepts = torch.stack(concept_embeddings).unsqueeze(0).squeeze(2).to(device)
    # if len(template_concept_embeddings) > 0:
    #     if len(template_concept_embeddings) + len(concept_embeddings) > model.concept_nr:
    #         allowed_concepts = model.concept_nr - len(concept_embeddings)
    #         template_concept_embeddings = template_concept_embeddings[:allowed_concepts]
    #         concepts = concepts[:allowed_concepts]
    #         labels = labels[:allowed_concepts]
    #         masks_templates = masks_templates[:allowed_concepts]
    #         coords = coords[:allowed_concepts]
    #     in_concepts_template = torch.stack(template_concept_embeddings).unsqueeze(0).squeeze(2).to(device)
    #     all_concepts = torch.cat([in_concepts, in_concepts_template], dim=1)
    # else:
    #     all_concepts = in_concepts

    x = all_concepts
    if x.shape[1] < model.concept_nr:
        zeros = torch.zeros(model.concept_nr - x.shape[1], x.shape[2]).unsqueeze(0).to(device)
        x = torch.cat([x, zeros], dim=1).to(device)
    # with torch.no_grad():
    #     x = reducer_model(x)
    nr_proposals = len(concept_embeddings)
    nr_templates = len(template_concept_embeddings)
    # print(nr_proposals, nr_templates)

    return x, temp_concepts, proposals, coords, nr_proposals, nr_templates, labels, masks_roi, masks_templates


def train(env, model, replay_buffer, params, writer, num_episodes=50000, target_model=None, reducer_model=None, graph_model=None):

    episode_durations = []

    steps_done = 0
    counter = 0
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        state_env = env.reset()
        state = process_frames(env)
        rew_ep = 0
        loss_ep = 0
        losses = []
        for t in count():
            # Select and perform an action
            if params['select_action'] == "50":
                action, steps_done = select_action_50(state, params, model, env.action_space.n, steps_done)
            else:
                action, steps_done = select_action(state, params, model, env.action_space.n, steps_done, t)

            new_state_env, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            rew_ep += reward.item()

            current_screen = process_frames(env)
            if not done:
                next_state = current_screen
            else:
                next_state = None
            import matplotlib.pyplot as plt
            plt.imshow(state)
            plt.show()

            plt.imshow(next_state)
            plt.show()

            # Store the transition in memory
            state_emb = apply_pil_changes(state)
            state_emb_res, concepts_state, proposals_state, coords_state, \
            nr_proposals_state, nr_templates_state, labels_state, masks_roi, masks_templates = save_embeddings_only(
                state_emb, model,reducer_model
            )
            # print(f'{nr_templates_state}-{nr_proposals_state}')
            if next_state is not None:
                next_state_emb = apply_pil_changes(next_state)
                next_state_emb_res, concepts, proposals, coords, nr_proposals, nr_templates, labels, masks_roi_next, masks_templates_next = save_embeddings_only(
                    next_state_emb, model, reducer_model
                )

            else:
                next_state_emb_res = next_state
            if next_state is not None:

                graph_data = create_graph_datapoint(
                    state_emb_res,action, reward, nr_templates_state, nr_proposals_state, labels_state, masks_templates, masks_roi,
                    next_state_emb_res, action, reward, nr_templates, nr_proposals, labels, masks_roi_next
                )
                import matplotlib.pyplot as plt
                plt.imshow(graph_data.train_y)
                plt.show()
                if graph_data is not None:
                    with torch.no_grad():
                        graph_result_state, graph_result_next_state, s = run_model_pca_gm(graph_data, graph_model)
                        import matplotlib.pyplot as plt
                        plt.imshow(s.clone().detach().cpu()[0])
                        plt.show()
                else:
                    graph_result_state = state_emb_res
                    graph_result_next_state = next_state_emb_res
            else:
                with torch.no_grad():
                    state_emb_res = reducer_model(state_emb_res)
                    next_state_emb_res = next_state_emb_res
                graph_result_state = state_emb_res
                graph_result_next_state = next_state_emb_res
            replay_buffer.push(graph_result_state, action, reward,
                               graph_result_next_state, done, concepts_state,
                               proposals_state, coords_state,
                               nr_templates_state, nr_proposals_state, labels_state, masks_roi, masks_templates)

            # Move to the next state
            prev_state = state
            state = next_state

            # Perform one step of the optimization (on the target network)
            if len(replay_buffer) > params['batch_size']:

                if target_model is not None:
                    loss_ep = compute_td_loss(model, replay_buffer, params, target_model=target_model)
                else:
                    loss_ep = compute_td_loss(model, replay_buffer, params)
                losses.append(loss_ep.item())

                writer.add_scalar('Loss/iter',loss_ep, counter)
                counter += 1
                #print(counter)

            if done:
                episode_durations.append(t + 1)
                writer.add_scalar('Reward/episode', rew_ep, i_episode)
                print(rew_ep)

                # fig, ax = plt.subplots()
                # plt.imshow(prev_state.squeeze(0).cpu().permute(1, 2, 0).squeeze(2).numpy(), cmap='gray',
                #            interpolation='none')
                # plt.title('Last state')
                # writer.add_figure(tag='att_dqn_top_view', figure=fig)

                #plt.show()
                # writer.add_scalar('Ep duration', t+1, i_episode)
                loss = np.sum(losses)
                writer.add_scalar('Loss/episode', loss/(t+1), i_episode)

                break
        if i_episode % params['target_update'] == 0:
            target_model.load_state_dict(model.state_dict())
        if i_episode % 50 == 0:
            torch.save(model.state_dict(), f't2{envs_to_run[3]}_non_z_8key_double_att_3h_pol_50_top_{i_episode}.pt')
            torch.save(reducer_model.state_dict(), f't2{envs_to_run[3]}_non_z_8key_double_att_3h_pol_50_top_reducer_{i_episode}.pt')


    return model


def main():
    params = {
        'batch_size': 64,
        'gamma': 0.999,
        'eps_start': 0.9,
        'eps_end': 0.05,
        'eps_decay': 200,
        'target_update': 1000,
        'select_action': 50,
    }
    params_att = {
        'ch_in': 1,
        'conv_1_ch': 16,
        'conv_2_ch': 20,
        'conv_3_ch': 24,
        'conv_4_ch': 30,
        'height': 32,
        'width': 32,
        'nr_heads':3,
        'node_size': 128,
        'lin_hidden': 500,
        'out_dim': 7,
        'coord_dim': 2,
        'N': 15,
        'attention':'add',
        'graph_model': None

    }
    writer = SummaryWriter(
        f'deep_rl/t2{envs_to_run[0]}_non_z_smaller_dqn_double_attention_concept_relations_{params["gamma"]}_{params["eps_start"]}_{params["eps_end"]}_{params["eps_decay"]}')

    env, window = init_env(envs_to_run[0])
    env.see_through_walls = False
    env.agent_view_size = 3
    concept_embeder = MiniGridEnv()
    init_screen = apply_pil_changes(process_frames(env))
    img, proposals, concepts, concept_embedding, masks_roi = concept_embeder.roi_extractor.get_concept_proposals_batch(
        init_screen)
    concept_dim = concept_embedding[0].shape[1]
    params_att['concept_dim'] = concept_dim

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    reducer = ReducerNet(params_att).to(device)
    # reducer.load_state_dict(torch.load('7key_double_att_3h_pol_50_top_reducer_600.pt', map_location=device))
    params_att['reducer'] = reducer
    dq_att = RelationalConceptGraphLearning(params_att).to(device)
    # dq_att.load_state_dict(torch.load('7key_double_att_3h_pol_50_top_600.pt', map_location=device))
    dq_att_target = RelationalConceptGraphLearning(params_att).to(device)
    # dq_att_target.load_state_dict(torch.load('7key_double_att_3h_pol_50_top_600.pt', map_location=device))

    model_gr = Net()
    model_gr.load_state_dict(
        torch.load('pre_temp_100temproi_pca_gm_frames_mixt_5gnn_200ep_fix_bug_adam_1.0e-3_500_60.pth', map_location=device)
    )
    model_gr.to(device)
    model_gr.eval()
    dq_att.graph_model = model_gr
    dq_att_target.graph_model = model_gr



    # from torchsummary import summary
    # print(summary(dq_att, (15, 4096)))

    maxsteps = 100  # D
    env.max_steps = maxsteps

    state_env = env.reset()


    screen_height, screen_width, _ = init_screen.shape
    n_actions = env.action_space.n

    dq_att_target.load_state_dict(dq_att.state_dict())
    dq_att_target.eval()
    optimizer = optim.Adam(dq_att.parameters(), lr=0.0001)
    memory = ReplayBufferConcepts(2000)
    params['optimizer'] = optimizer

    dq_att = train(env, dq_att, memory, params, writer, target_model=dq_att_target, reducer_model=reducer, graph_model=model_gr)
    torch.save(dq_att.state_dict(), 'double_att_att_concepts.pt')
    return


if __name__ == '__main__':
    #top_view()
    main()
