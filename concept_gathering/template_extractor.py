import cv2 as cv
import os
import numpy as np
import torch
import uuid

from matplotlib import pyplot as plt
from gym_minigrid.wrappers import *
from PIL import Image
from gym_minigrid.window import Window
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
from concept_gathering.concept_embeder import ConceptDataSet


class TemplateConceptExtractor:
    all_templates = ['temp_ad.png', 'temp_al.png', 'temp_ar.png', 'temp_au.png',
                     'temp_circle.png', 'temp_key.png', 'temp_d.png', 'goal.png']

    def __init__(self):

        self.dir = f'{os.getcwd()}'
        self.objects = self.all_templates
        self.to_gray = T.Compose([T.Grayscale()])
        self.concept_ds = ConceptDataSet()

    def match_template(self, image, template, threshold):
        img_gray = image
        res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        return loc

    def find_objects_in_frame(self, frame):
        all_locs = []
        centers = []
        concepts = []
        concept_embeddings = []
        for template_name in self.objects:
            top_dir = os.path.dirname(self.dir)
            cwd = f'{top_dir}/concept_gathering/templates/{template_name}'
            template = np.array(self.to_gray(Image.open(cwd)).convert('RGB'))
            newX, newY = 30, 30
            newimg = cv.resize(template, (int(newX), int(newY)))
            loc = self.match_template(frame, newimg, 0.8)
            all_locs.append(loc)

        for loc in all_locs:
            for pt in zip(*loc):
                roi = frame[pt[0]:pt[0] + newY, pt[1]:pt[1] + newX]
                tensor_image, embeded_concept = self.concept_ds.get_embedding(roi)
                concepts.append(roi)
                concept_embeddings.append(embeded_concept)
                cv.imwrite(f'{self.dir}/concepts/obj_{uuid.uuid4()}.png', roi)
        #         cv.rectangle(frame, (pt[1], pt[0]), (pt[1] + newX, pt[0] + newY), (0, 0, 255), 2)
        # cv.imwrite(f'{self.dir}/results_template_matching/output_frame_{env_id}.jpg', frame)
        return concepts, concept_embeddings, all_locs

    def find_objects_in_frame_run(self, frame):
        all_locs = []
        centers = []
        concepts = []
        thrs = 0.8
        concept_embeddings = []
        top_dir = f'{os.path.dirname(self.dir)}/concept_gathering/templates/'
        all_types = os.listdir(top_dir)
        for obj_type in all_types:
            obj_dir = f'{top_dir}/{obj_type}/'
            if obj_type == 'door' or obj_type == 'arrow':
                thrs = 0.6


            # print(obj_type)
            for obj in os.listdir(obj_dir):
                obj_img = f'{obj_dir}/{obj}'
                template = np.array(self.to_gray(Image.open(obj_img)).convert('RGB'))
                newX, newY = 30, 30
                newimg = cv.resize(template, (int(newX), int(newY)))

                loc = self.match_template(frame, newimg, thrs)
                all_locs.append(loc)

        prev_pt_x = 0
        prev_pt_y = 0
        for loc in all_locs:
            for pt in zip(*loc):
                if pt[0] - prev_pt_y > 5 or pt[1] - prev_pt_x > 5:
                    roi = frame[pt[0]:pt[0] + newY, pt[1]:pt[1] + newX]
                    embeded_concept,_ = self.concept_ds.get_embedding(roi)
                    concepts.append(roi)
                    # plt.imshow(roi)
                    # plt.show()
                    concept_embeddings.append(embeded_concept)
                    prev_pt_y = pt[0]
                    prev_pt_x = pt[1]

        return concepts, concept_embeddings, all_locs


def main():
    #object_detection_api('mn2.jpg', threshold=0.5)
    dir = f'{os.getcwd()}/concept_gathering/imgs/mn_circle2.png'
    template_extractor = TemplateConceptExtractor()


    return

if __name__ == '__main__':
    main()