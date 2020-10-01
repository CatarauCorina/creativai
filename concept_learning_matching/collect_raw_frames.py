import cv2 as cv
import os
import numpy as np
import torch
import uuid
import pickle
from itertools import count
import random
from concept_gathering.init_and_run_env import MiniGridEnv
from relational_module.relational_attention_dqn import RelationalConceptGraphLearning, ReducerNet
from concept_gathering.concept_buffer import ReplayBufferConcepts
from concept_learning_matching.concept_ds_mixmatch import PerceptPairsDatasetMix
from matplotlib import pyplot as plt
from gym_minigrid.wrappers import *
from concept_learning_matching.PCA.model import Net

from PIL import Image
from gym_minigrid.window import Window
from torch_geometric.data import Data, InMemoryDataset, extract_zip

from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T

envs_to_run = [
'MiniGrid-DoorKey-16x16-v0',

    'MiniGrid-Unlock-v0',
    'MiniGrid-Fetch-8x8-N3-v0',
    'MiniGrid-Fetch-5x5-N2-v0',
    'MiniGrid-Empty-8x8-v0',
     'MiniGrid-FourRooms-v0',
    'MiniGrid-SimpleCrossingS9N1-v0', 'MiniGrid-MultiRoom-N2-S4-v0',
]


all_templates = ['temp_ad.png', 'temp_al.png', 'temp_ar.png','temp_au.png',
                 'temp_circle.png', 'temp_key.png','temp_d.png','goal.png']

# all_templates = ['circle1.png']
# T.ToTensor()])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_env(env_name):
    env = gym.make(env_name)
    env.max_steps = 256
    window = Window('gym_minigrid - ' + env_name)
    return env, window


def process_frames(env):
    resize = T.Compose([T.ToPILImage(),
                        # T.Resize((16,16), interpolation=Image.CUBIC),
                        T.Grayscale()])
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    screen = torch.tensor(screen)

    return resize(screen)


def run_test(template, template_name):
    env, window = init_env(envs_to_run[0])
    env.see_through_walls = False
    env.agent_view_size = 3
    init_screen = process_frames(env)
    env.reset()

    # _, _, screen_height, screen_width = init_screen.shape
    n_actions = env.action_space.n
    actions = [1, 2]
    for i in range(1):
        plt.figure()
        nr = int(i % 2)
        next_state, reward, done, _ = env.step(actions[nr])
        crt_frame = process_frames(env).convert('RGB')
        crt_frame = np.array(crt_frame)
        match_template(crt_frame, template, template_name, 0.6, show=True)

    return


def match_template(image, template, template_name, threshold, show=True):
    """
    Match the image with one single template. return the matched rectangular areas
    :param image:
    :param template: template file name
    :param threshold:
    :return: [(left,right,top,bottom), (...)]
    """
    object_locs = []
    img_gray = image
    w, h = template.shape[:-1]
    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)

    return loc


def process_env(env_id):
    env, window = init_env(envs_to_run[env_id])
    env.see_through_walls = False
    env.agent_view_size = 3
    init_screen = process_frames(env)
    env.reset()
    actions = [1, 2]
    for i in range(1):
        plt.figure()
        nr = int(i % 2)
        next_state, reward, done, _ = env.step(actions[nr])
        crt_frame = process_frames(env).convert('RGB')
        crt_frame = np.array(crt_frame)
        find_objects_in_frame(crt_frame, all_templates, env_id)
    return


def apply_pil_changes(frame):
    crt_frame = frame.convert('RGB')
    crt_frame = np.array(crt_frame)
    return crt_frame


def save_embeddings_only(state, model, reducer_model):
    img, proposals, concepts, concept_embeddings, masks_roi = model.concept_embeder.roi_extractor.get_concept_proposals_batch(state)
    temp_concepts, template_concept_embeddings, coords, labels, masks_templates = model.concept_embeder.template_extractor.find_objects_in_frame_run(state)
    nr_proposals = len(concept_embeddings)
    nr_templates = len(template_concept_embeddings)
    plt.imshow(img)

    in_concepts = torch.stack(concept_embeddings).unsqueeze(0).squeeze(2).to(device)
    if len(template_concept_embeddings) > 0:
        in_concepts_template = torch.stack(template_concept_embeddings).unsqueeze(0).squeeze(2).to(device)
        all_concepts = torch.cat([in_concepts, in_concepts_template], dim=1)
    else:
        all_concepts = in_concepts

    x = all_concepts
    if x.shape[1] < model.concept_nr:
        zeros = torch.zeros(model.concept_nr - x.shape[1], x.shape[2]).unsqueeze(0).to(device)
        x = torch.cat([x, zeros], dim=1).to(device)
    # with torch.no_grad():
    #     x = reducer_model(x)
    return x, temp_concepts, proposals, coords, nr_proposals, nr_templates, labels, masks_roi, masks_templates


def process_graph_data_mix(
        ds,
        nodes_g1, nr_templates_g1, nr_proposals_g1, template_labels_g1, masks_temp_g1,
        nodes_g2, nr_templates_g2, nr_proposals_g2, template_labels_g2, masks_temp_g2
):
    matches_temp_roi = ds.find_coords_matches(masks_temp_g1, nr_proposals_g1.item(), masks_roi_g2)
    full_nodes = nodes_g1[:, :(nr_proposals_g1 + nr_templates_g1), :]
    only_template_nodes = nodes_g1[:, nr_proposals_g1:(nr_proposals_g1 + nr_templates_g1), :]
    x1 = nodes_g1
    x2 = nodes_g2
    adj_1 = ds.init_adj_matrix(x1)
    adj_2 = ds.init_adj_matrix(x2)
    train_y, all_pos = ds.process_permutation_matrix(template_labels_g1, template_labels_g2, nodes_g1,
                                                       nodes_g2, nr_proposals_g1, nr_proposals_g2)
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


def run_model(inputs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Net()
    model.to(device)
    model.eval()
    emb_src = inputs.x1.to(device)
    emb_tgt = inputs.x2.to(device)
    perm_mat = inputs.train_y.unsqueeze(0).to(device)
    a_src = inputs.edge_attr1.to(device)
    a_tgt = inputs.edge_attr2.to(device)
    pos_matches = inputs.all_pos
    n1_gt = inputs.n1_gt.to(device)
    n2_gt = inputs.n2_gt.to(device)
    s_pred, new_emb_src, new_emb_tgt = model(emb_src, emb_tgt, a_src, a_tgt)
    print(s_pred)

    return s_pred, emb_src, emb_tgt


def test_frames(env, model, replay_buffer, params, num_episodes=30, target_model=None, reducer_model=None, env_id=1):

    episode_durations = []

    steps_done = 0
    counter = 0
    ds = PerceptPairsDatasetMix(os.path.join(os.getcwd(), 'caches_ds_mixt'), 'percepts_mixt')

    for i_episode in range(num_episodes):
        replay_buffer = ReplayBufferConcepts(256)
        # Initialize the environment and state
        state_env = env.reset()
        state = process_frames(env)
        rew_ep = 0
        loss_ep = 0
        losses = []
        for t in count():
            # Select and perform an action
            action = torch.tensor([[random.randrange(env.action_space.n)]], device=device, dtype=torch.long)

            new_state_env, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            rew_ep += reward.item()

            current_screen = process_frames(env)
            if not done:
                next_state = current_screen
            else:
                next_state = None

            # Store the transition in memory
            state_emb = apply_pil_changes(state)
            state_emb_res, concepts, proposals, coords, nr_templates, nr_proposals, labels, masks_roi, masks_templates = save_embeddings_only(
                state_emb, model,reducer_model
            )

            if next_state is not None:
                next_state_emb = apply_pil_changes(next_state)
                next_state_emb_res, concepts, proposals, coords, nr_templates, nr_proposals, labels, masks_roi_next, masks_templates_next = save_embeddings_only(
                    next_state_emb, model, reducer_model
                )
            else:
                next_state_emb_res = next_state
            data_points = process_graph_data_mix(
                ds,
                state_emb_res, concepts, proposals, coords, nr_templates, nr_proposals, labels, masks_roi,
                masks_templates,
                next_state_emb_res, concepts, proposals, coords, nr_templates, nr_proposals, labels, masks_roi_next,
                masks_templates_next

            )
            adj_matrix, new_state_emb, new_next_state_emb = run_model(data_points)
            print(adj_matrix)


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
        'lin_hidden': 100,
        'out_dim': 7,
        'coord_dim': 2,
        'N': 15,
        'attention':'add',

    }
    env_ids = [0,1]
    for env_id in env_ids:
        env, window = init_env(envs_to_run[env_id])
        env.see_through_walls = False
        env.agent_view_size = 3
        concept_embeder = MiniGridEnv()
        init_screen = apply_pil_changes(process_frames(env))
        img, proposals, concepts, concept_embedding, masks_roi = concept_embeder.roi_extractor.get_concept_proposals_batch(
            init_screen)
        concept_dim = concept_embedding[0].shape[1]
        params_att['concept_dim'] = concept_dim
        reducer = ReducerNet(params_att).to(device)
        params_att['reducer'] = reducer
        dq_att = RelationalConceptGraphLearning(params_att).to(device)

        # from torchsummary import summary
        # print(summary(dq_att, (15, 4096)))

        maxsteps = 256  # D
        env.max_steps = maxsteps

        state_env = env.reset()
        screen_height, screen_width, _ = init_screen.shape
        n_actions = env.action_space.n


        dq_att = test_frames(env, dq_att, None, params, reducer_model=reducer,env_id=env_id)
        torch.save(dq_att.state_dict(), 'double_att_att_concepts.pt')

    return



def find_objects_in_frame(frame, objects, env_id):
    to_gray = T.Compose([T.Grayscale()])
    all_locs = []
    centers = []
    new_frame = frame
    back = np.array(to_gray(Image.open(f'{os.getcwd()}/back.png')).convert('RGB'))

    for template_name in objects:
        template = np.array(to_gray(Image.open(f'{os.getcwd()}/templates/{template_name}')).convert('RGB'))
        newX, newY = 30, 30
        newimg = cv.resize(template, (int(newX), int(newY)))
        loc = match_template(frame, newimg,template_name, 0.8)
        all_locs.append(loc)
    for loc in all_locs:
        for pt in zip(*loc):
            roi = frame[pt[0]:pt[0] + newY, pt[1]:pt[1] + newX]
            back_res = cv.resize(back,(int(newX), int(newY)))
            new_frame[pt[0]:pt[0] + newY, pt[1]:pt[1] + newX] = back_res
            # cv.imwrite(f'{os.getcwd()}/results_template_matching/obj_{uuid.uuid4()}.png',roi)
            cv.rectangle(frame, (pt[1], pt[0]), (pt[1] + newX, pt[0] + newY), (0, 0, 255), 2)
    cv.imwrite(f'{os.getcwd()}/results_template_matching/output_frame_{env_id}.jpg', frame)
    cv.imwrite(f'{os.getcwd()}/results_template_matching/output_frame_masked_{env_id}.jpg', new_frame)


    return


# for i, env in enumerate(envs_to_run):
if __name__ == '__main__':
    main()



