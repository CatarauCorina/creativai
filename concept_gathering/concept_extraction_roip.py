import torchvision
from PIL import Image
import os
import uuid
import torch
import cv2
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class ROIConceptExtractor:
    INSTANCE_CATEGORY_NAMES = [
        '__background__', 'square', 'circle',
        'triangle', 'key', 'door'
    ]
    device = torch.device('cpu')

    def __init__(self, use_checkpoint=False, checkpoint_file_name=""):
        self.checkpoint_file = checkpoint_file_name
        faster = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        num_classes = 4
        in_features = faster.roi_heads.box_predictor.cls_score.in_features
        faster.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.dir = f'{os.getcwd()}'
        if use_checkpoint:
            cwd = f'{self.dir}/checkpoints/{checkpoint_file_name}'
            faster.load_state_dict(torch.load(cwd, map_location=self.device))
        faster.eval()
        self.model = faster
        transform = T.Compose([T.ToTensor()])
        self.to_tensor_transf = transform

    def get_concept_proposals_display(self, img, rect_th=1):
        proposals = self.get_roi(img)[0]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img_prop = img.copy()

        for i in range(proposals.shape[0]):
            roi = img_prop[int(proposals[i][1]):int(proposals[i][3]), int(proposals[i][0]):int(proposals[i][2])]
            cv2.imwrite(f'{self.dir}/concepts/obj_{uuid.uuid4()}.png', roi)

            cv2.rectangle(img, (proposals[i][0], proposals[i][1]), (proposals[i][2], proposals[i][3]),
                          color=(0, 255, 0), thickness=rect_th)  # Draw Rectangle with the coordinates

        plt.imshow(img)
        id = uuid.uuid4()

        plt.savefig(f'{self.dir}/outputs/roi_{id}_{self.checkpoint_file}.jpg')
        return


    def get_roi(self, img):
        device = torch.device('cpu')
        img = self.to_tensor_transf(img)
        outputs = []
        sizes = []

        def hook(module, x, y):
            scores = [idx for idx, score in enumerate(y[0][0]['scores'][:5])]
            boxes = y[0][0]['boxes'][scores]
            outputs.append(boxes)
            sizes.append(x[2][0])

        self.model.roi_heads.register_forward_hook(hook)
        pred = self.model([img])

        scale_ratio_x = sizes[0][0] / img.shape[1]
        scale_ratio_y = sizes[0][1] / img.shape[2]
        outputs[0][:, 0] = outputs[0][:, 0] / scale_ratio_x
        outputs[0][:, 2] = outputs[0][:, 2] / scale_ratio_x

        outputs[0][:, 1] = outputs[0][:, 1] / scale_ratio_y
        outputs[0][:, 3] = outputs[0][:, 3] / scale_ratio_y
        return outputs

    def get_rpn(self, img_path):
        img = Image.open(img_path)  # Load the image
        img = self.to_tensor_transf(img)  # Apply the transform to the image

        outputs = []
        def hook(module, x, y):
            outputs.append(y[0][0])

        layer_feature = self.model.rpn.register_forward_hook(hook)
        pred = self.model([img])
        return outputs

    def get_prediction(self, img_path, threshold):
        img = Image.open(img_path)  # Load the image
        img = self.to_tensor_transf(img)
        pred = self.model([img])  # Pass the image to the model
        pred_class = [self.INSTANCE_CATEGORY_NAMES[i] for i in
                      list(pred[0]['labels'].numpy())]  # Get the Prediction Score
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]  # Bounding boxes
        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][
            -1]  # Get list of index with score greater than threshold.
        pred_boxes = pred_boxes[:pred_t + 1]
        pred_class = pred_class[:pred_t + 1]
        return pred_boxes, pred_class

    def get_object_detection_display(self, img_path, threshold=0.5, rect_th=1, text_size=0.8, text_th=1):

        boxes, pred_cls = self.get_prediction(img_path, threshold)  # Get predictions
        img = cv2.imread(img_path)  # Read image with cv2
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        for i in range(len(boxes)):
            cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0),
                          thickness=rect_th)  # Draw Rectangle with the coordinates
            cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0),
                        thickness=text_th)  # Write the prediction class
        plt.figure(figsize=(20, 30))  # display the output image
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()
        return



def main():
    #object_detection_api('mn2.jpg', threshold=0.5)
    dir = f'{os.getcwd()}/concept_gathering/imgs/mn_circle2.png'
    image = cv2.imread(dir)
    roi_extractor_1 = ROIConceptExtractor()
    roi_extractor_2 = ROIConceptExtractor(use_checkpoint=True, checkpoint_file_name='faster_max_3obj.pt')
    roi_extractor_3 = ROIConceptExtractor(use_checkpoint=True, checkpoint_file_name='faster_more.pt')
    #roi_extractor_3.get_object_detection_display(dir,threshold=0.8)
    roi_extractor_1.get_concept_proposals_display(image)
    roi_extractor_2.get_concept_proposals_display(image)
    roi_extractor_3.get_concept_proposals_display(image)

    return

if __name__ == '__main__':
    main()



