import numpy as np
import cv2
import math
import random
import pickle
from torch.utils.data import Dataset


class ShapesDataset(Dataset):
    image_info = {}
    all_shapes = {
        'square': 1,
        'circle': 2,
        'triangle': 3,
        'key': 4,
        'door': 5,
    }

    def load_shapes(self, count, height, width):
        # self.add_class("shapes", 1, "square")
        # self.add_class("shapes", 2, "circle")
        # self.add_class("shapes", 3, "triangle")


        for i in range(count):
            bg_color, shapes = self.random_image(height, width)
            self.add_image("shapes", image_id=i, path=None,
                           width=width, height=height,
                           bg_color=bg_color, shapes=shapes)

    def add_image(self,val, image_id, path, width, height, bg_color, shapes):
        self.image_info[image_id] = {
            'bg_color': bg_color,
            'width': width,
            'height': height,
            'shapes': shapes,
        }
        return

    def load_image(self, image_id):
        info = self.image_info[image_id]
        bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        shapes = []
        boxes = []
        for shape, color, dims in info['shapes']:
            image = self.draw_shape(image, shape, dims, color)
            shapes.append(self.all_shapes[shape])
            x, y, s = dims
            boxes.append([y - s, x - s, y + s, x + s])
        # plt.imshow(image)
        # plt.show()
        bbox, mask = self.load_bbox(image_id)
        return image, boxes, bbox, mask, shapes

    def load_all_images(self):
        all_imgs = []
        all_bbox = []
        all_boxes = []
        all_masks = []
        all_shapes = []
        for img_id in self.image_info.keys():
            img, boxes, bbox, mask, shape = self.load_image(img_id)
            all_imgs.append(img)
            all_bbox.append(bbox)
            all_boxes.append(boxes)
            all_masks.append(mask)
            all_shapes.append(shape)
        return all_imgs, all_bbox, all_boxes, all_masks, all_shapes

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def get_keys(self, d, value):
        return [k for k, v in d.items() if v == value]

    def load_bbox(self, image_id):
        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)

        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info['shapes']):
            mask[:, :, i:i + 1] = self.draw_shape(mask[:, :, i:i + 1].copy(),
                                                  shape, dims, 1)


        bboxes = []
        for i in range(mask.shape[2]):
            cnts, _ = cv2.findContours(mask[:, :, i] * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

            x, y, w, h = cv2.boundingRect(cnt)
            # The right format is:
            # (y1, x1, y2, x2)
            bboxes.append([x, y, x + w, y + h])

        # class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        #
        # if len(class_ids) != len(bboxes):
        #     raise ValueError("Class ids are not equal with num of bboxes")

        return np.array(bboxes), mask

    def draw_shape(self, image, shape, dims, color):
        x, y, s = dims
        if shape == 'square':
            image = cv2.rectangle(image, (x - s, y - s),
                                  (x + s, y + s), color, -1)
        elif shape == "circle":
            image = cv2.circle(image, (x, y), s, color, -1)
        elif shape == "triangle":
            points = np.array([[(x, y - s),
                                (x - s / math.sin(math.radians(60)), y + s),
                                (x + s / math.sin(math.radians(60)), y + s),
                                ]], dtype=np.int32)
            image = cv2.fillPoly(image, points, color)
        elif shape == 'key':
            s_img = cv2.imread("templates/temp_key.png", -1)
            self.overlay_image_alpha(image,
                                     s_img[:, :, 0:3],
                                     (x, y),
                                     s_img[:, :, 3] / 255.0)
        elif shape == 'door':
            s_img = cv2.imread("templates/temp_door.png", -1)
            self.overlay_image_alpha(image,
                                     s_img[:, :, 0:3],
                                     (x, y),
                                     s_img[:, :, 3] / 255.0)

        return image

    def overlay_image_alpha(self, img, img_overlay, pos, alpha_mask):
        """Overlay img_overlay on top of img at the position specified by
        pos and blend using alpha_mask.

        Alpha mask must contain values within the range [0, 1] and be the
        same size as img_overlay.
        """

        scale_percent = random.randint(10, 70)
        width = int(img_overlay.shape[1] * scale_percent / 100)
        height = int(img_overlay.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        img_overlay = cv2.resize(img_overlay, dim, interpolation=cv2.INTER_AREA)

        x, y = pos

        # Image ranges
        y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
        x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

        # Overlay ranges
        y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
        x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

        # Exit if nothing to do
        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return

        channels = img.shape[2]

        alpha = alpha_mask[y1o:y2o, x1o:x2o]
        alpha_inv = 1.0 - alpha

        for c in range(channels):
            img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                    alpha_inv * img[y1:y2, x1:x2, c])

    def random_shape(self, height, width):
        shape = random.choice(["square", "circle", "triangle"])
        color = tuple([random.randint(0, 255) for _ in range(3)])
        buffer = 10
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        s = random.randint(buffer, height // 4)
        return shape, color, (x, y, s)

    def random_image(self, height, width):
        # bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        bg_color = np.array([0, 0, 0], dtype=np.uint8)
        shapes = []
        N = random.randint(1, 3)
        for _ in range(N):
            shape, color, dims = self.random_shape(height, width)
            shapes.append((shape, color, dims))
            x, y, s = dims

        return bg_color, shapes


ds = ShapesDataset()
ds.load_shapes(10000,128,128)
all_imgs, all_bbox, all_boxes, all_masks, all_shapes = ds.load_all_images()

data = {
    'imgs': all_imgs,
    'bboxes': all_bbox,
    'boxes': all_boxes,
    'masks': all_masks,
    'labels': all_shapes
}
with open('data_shapes10k_3obj.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(len(all_imgs))