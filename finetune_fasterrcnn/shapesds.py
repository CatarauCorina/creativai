from torch.utils.data import Dataset
import pickle
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class ShapesDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()

    ])
        # load all image files, sorting them to
        # ensure that they are aligned
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        with open('data_shapes10k_3obj.pickle', 'rb') as handle:
            data = pickle.load(handle)
        self.imgs = data['imgs']
        self.bboxes = data['bboxes']
        self.boxes = data['boxes']
        self.masks = data['masks']

        self.labels = data['labels']

    def __getitem__(self, idx):


        img = self.imgs[idx]
        mask = self.masks[idx]
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)


        # convert everything into a torch.Tensor
        boxes = self.bboxes[idx]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        bboxes = torch.as_tensor(self.bboxes[idx], dtype=torch.float32)
        # there is only one class
        # labels = torch.ones((num_objs,), dtype=torch.int64)
        labels = torch.tensor(self.labels[idx], dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        # target["boxes"] = boxes.to(self.device)
        target['boxes'] = bboxes.to(self.device)
        target["labels"] = labels.to(self.device)
        target["masks"] = masks.to(self.device)
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img =  Image.fromarray(img)
            img, target = self.transforms(img) ,target

        return img.to(self.device), target

    def __len__(self):
        return len(self.imgs)