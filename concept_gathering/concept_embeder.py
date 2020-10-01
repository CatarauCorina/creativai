import os
import torch
from einops import rearrange
import numpy as np
import cv2 as cv
import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F

import matplotlib.pyplot as plt


class ConceptDataSet(Dataset):
    def __init__(self):
        self.dir = f'{os.getcwd()}'
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.main_dir = f'{self.dir}/concepts/'

        transform = T.Compose([T.ToTensor()])
        self.to_tensor_transf = transform
        vgg16 = torchvision.models.vgg16(pretrained=True).to(self.device)
        vgg16.eval()
        self.model = vgg16
        self.embeded_concept = []
        def hook(module, x, y):
            self.embeded_concept.append(y.detach())
            del y
            del x

        self.model.classifier[0].register_forward_hook(hook)

    # def __len__(self):
    #     return len(self.total_imgs)

    # def __getitem__(self, idx):
    #     embeded_concept = []
    #     img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
    #     image = np.array(Image.open(img_loc).convert("RGB"))
    #     image_resize = cv.resize(image, (256, 256))
    #
    #     tensor_image = self.to_tensor_transf(image_resize)
    #
    #     def hook(module, x, y):
    #         embeded_concept.append(y.to('cpu'))
    #
    #     self.model.classifier[0].register_forward_hook(hook)
    #     with torch.no_grad():
    #         batch_img = tensor_image.unsqueeze(0)
    #         self.model(batch_img.to(self.device))
    #
    #     return tensor_image, embeded_concept[0], image_resize

    def get_embedding(self, img):
        self.embeded_concept = []
        # image = np.array(img.convert("RGB"))
        image_resize = cv.resize(img, (256, 256))
        tensor_image = self.to_tensor_transf(image_resize).to(self.device)

        batch_img = tensor_image.unsqueeze(0)
        with torch.no_grad():
            batch_img = Variable(batch_img, volatile=True)
            self.model(batch_img.to(self.device))
        del tensor_image
        return self.embeded_concept[0], image_resize

    def get_embeddigs_matrix(self):
        all_embeddings = []
        for idx, img in enumerate(self.total_imgs):
            img, embedding, _ = self.__getitem__(idx)
            all_embeddings.append(np.array(embedding))
        return torch.tensor(all_embeddings).squeeze(1)


class ConceptEmbedder:

    def __init__(self):
        self.ds = ConceptDataSet()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        return

    def load_vgg_processed_inputs(self, layers_to_use={'4_2': 20, '5_1': 25}):
        vgg16_outputs = []

        def hook(module, x, y):
            vgg16_outputs.append(y.to('cpu'))

        vgg16 = torchvision.models.vgg16(pretrained=True).to(self.device)
        vgg16.eval()
        vgg16.classifier[0].register_forward_hook(hook)
        for i, batch_img in enumerate(self.ds):

            with torch.no_grad():
                batch_img = batch_img.unsqueeze(0)
                vgg16(batch_img.to(self.device))


        return vgg16_outputs



# ds = ConceptDataSet()
# print(len(ds))
# # for i in range(len(ds)):
# #     el = ds[i]
# to_compare = 32
# plt.imshow(ds[to_compare][2])
# plt.show()
# matrix = ds.get_embeddigs_matrix()
# from scipy.spatial.distance import cdist
# all_except_index = np.delete(matrix, to_compare, axis=0)
# dist = cdist(matrix[to_compare].unsqueeze(0), all_except_index, metric='cosine')
# res = np.argmin(dist)
# print(dist)
# print(res)
# plt.imshow(ds[res][2])
# plt.show()
