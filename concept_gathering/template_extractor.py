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


class TemplateConceptExtractor:
    all_templates = ['temp_ad.png', 'temp_al.png', 'temp_ar.png', 'temp_au.png',
                     'temp_circle.png', 'temp_key.png', 'temp_d.png', 'goal.png']

    def __init__(self):

        self.dir = f'{os.getcwd()}'
        self.objects = self.all_templates
        self.to_gray = T.Compose([T.Grayscale()])

    def match_template(self, image, template, threshold):
        img_gray = image
        res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        return loc

    def find_objects_in_frame(self, frame, env_id):
        all_locs = []
        centers = []
        for template_name in self.objects:
            template = np.array(self.to_gray(Image.open(f'{self.dir}/templates/{template_name}')).convert('RGB'))
            newX, newY = 30, 30
            newimg = cv.resize(template, (int(newX), int(newY)))
            loc = self.match_template(frame, newimg, 0.8)
            all_locs.append(loc)

        for loc in all_locs:
            for pt in zip(*loc):
                roi = frame[pt[0]:pt[0] + newY, pt[1]:pt[1] + newX]
                cv.imwrite(f'{self.dir}/concepts/obj_{uuid.uuid4()}.png', roi)
        #         cv.rectangle(frame, (pt[1], pt[0]), (pt[1] + newX, pt[0] + newY), (0, 0, 255), 2)
        # cv.imwrite(f'{self.dir}/results_template_matching/output_frame_{env_id}.jpg', frame)
        return

def main():
    #object_detection_api('mn2.jpg', threshold=0.5)
    dir = f'{os.getcwd()}/concept_gathering/imgs/mn_circle2.png'
    template_extractor = TemplateConceptExtractor()


    return

if __name__ == '__main__':
    main()