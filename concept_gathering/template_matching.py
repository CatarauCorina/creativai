import cv2 as cv
import os
import numpy as np
import torch
import uuid
import tensorflow as tf
from matplotlib import pyplot as plt
from gym_minigrid.wrappers import *
from PIL import Image
from gym_minigrid.window import Window
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


def find_objects_in_frame(frame, objects, env_id):
    to_gray = T.Compose([T.Grayscale()])
    all_locs = []
    centers = []
    for template_name in objects:
        template = np.array(to_gray(Image.open(f'{os.getcwd()}/concept_gathering/templates/{template_name}')).convert('RGB'))
        newX, newY = 30, 30
        newimg = cv.resize(template, (int(newX), int(newY)))
        loc = match_template(frame, newimg,template_name, 0.8)
        all_locs.append(loc)
    for loc in all_locs:
        for pt in zip(*loc):
            roi = frame[pt[0]:pt[0] + newY, pt[1]:pt[1] + newX]
            cv.imwrite(f'{os.getcwd()}/concept_gathering/results_template_matching/obj_{uuid.uuid4()}.png',roi)
            cv.rectangle(frame, (pt[1], pt[0]), (pt[1] + newX, pt[0] + newY), (0, 0, 255), 2)
    cv.imwrite(f'{os.getcwd()}/concept_gathering/results_template_matching/output_frame_{env_id}.jpg', frame)
    return


# for i, env in enumerate(envs_to_run):
process_env(1)




