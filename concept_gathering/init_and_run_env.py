import cv2 as cv
import numpy as np
import torch
import uuid

from matplotlib import pyplot as plt
from gym_minigrid.wrappers import *
from PIL import Image
from gym_minigrid.window import Window
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
from concept_gathering.template_extractor import TemplateConceptExtractor
from concept_gathering.concept_extraction_roip import ROIConceptExtractor


class MiniGridEnv:
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

    def __init__(self):
        self.roi_extractor = ROIConceptExtractor(use_checkpoint=True, checkpoint_file_name='faster_more.pt')
        # self.roi_extractor_no_check = ROIConceptExtractor()
        self.template_extractor = TemplateConceptExtractor()
        return

    def init_env(self, env_name):
        env = gym.make(env_name)
        env.max_steps = 256
        window = Window('gym_minigrid - ' + env_name)
        return env, window

    def process_frames(self, env):
        resize = T.Compose([T.ToPILImage(),
                            # T.Resize((16,16), interpolation=Image.CUBIC),
                            T.Grayscale()])
        screen = env.render(mode='rgb_array').transpose((2, 0, 1))
        _, screen_height, screen_width = screen.shape
        screen = torch.tensor(screen)
        return resize(screen)

    def process_env_template_matching(self, env_id):
        env, window = self.init_env(self.envs_to_run[env_id])
        env.see_through_walls = False
        env.agent_view_size = 3
        init_screen = self.process_frames(env)
        env.reset()
        actions = [0, 1, 2]
        for i in range(30):
            plt.figure()
            nr = np.random.randint(0, 3)
            next_state, reward, done, _ = env.step(actions[nr])
            crt_frame = self.process_frames(env).convert('RGB')
            crt_frame = np.array(crt_frame)
            self.template_extractor.find_objects_in_frame(crt_frame)
        return

    def process_env_roi_extraction(self, env_id):
        env, window = self.init_env(self.envs_to_run[env_id])
        env.see_through_walls = False
        env.agent_view_size = 3
        init_screen = self.process_frames(env)
        env.reset()
        actions = [0, 1, 2]
        for i in range(30):
            plt.figure()
            nr = np.random.randint(0, 3)
            next_state, reward, done, _ = env.step(actions[nr])
            crt_frame = self.process_frames(env).convert('RGB')
            crt_frame = np.array(crt_frame)
            self.roi_extractor.get_concept_proposals_display(crt_frame)
        return



# def main():
#     env = MiniGridEnv()
#     #env.process_env_template_matching(0)
#     env.process_env_roi_extraction(0)
#     return
#
#
# if __name__ == '__main__':
#     main()


