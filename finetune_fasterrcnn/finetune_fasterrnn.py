import matplotlib
import torchvision
import torch
import torchvision.transforms as T
import torch.utils as utils
from torch.utils.tensorboard import SummaryWriter


import matplotlib.pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from finetune_fastercnn_shapes.shapesds import ShapesDataset




def test_display(i=0, img_size=128):
    dataset = ShapesDataset('shapes')
    color_labels = ['r', 'g', 'b']
    img = dataset[500][0]
    img = img.permute(1, 2, 0)
    bboxes = dataset[500][1]['boxes']

    plt.imshow(img, interpolation='none', origin='lower',extent=[0, img_size, 0, img_size])
    for bbox in bboxes:
        plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], edgecolor='white', fc='none'))
        # plt.annotate(shape_labels[shape], (bbox[0], bbox[1] + bbox[3] + 0.7), color=color_labels[color], clip_on=False)
    plt.show()
    return


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def evaluate(model, ds, device):
    with torch.no_grad():
        for idx, val in enumerate(ds):
            # images, targets = next(iter(data_loader))
            images = val[0].unsqueeze(0).to(device)
            targets = [val[1]]
            model.eval()
            pred = model(images)  # Returns losses and detections
            print(pred[0]['labels'] == targets)
            print(pred)


def finetune_fastercnn():
    faster = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 4
    in_features = faster.roi_heads.box_predictor.cls_score.in_features
    faster.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    dataset = ShapesDataset('shapes')
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True)
    # For Training
    for idx, val in enumerate(dataset):
    #images, targets = next(iter(data_loader))
        images = val[0].unsqueeze(0)
        targets = [val[1]]
        output = faster(images, targets)  # Returns losses and detections
        print(output)
        break
    # For inference
    faster.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = faster(x)
    print(predictions)

    return


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_one_epoch(model, optimizer, dataset, device, epoch, writer, counter,test_ds, print_freq):
    model.train()

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(dataset) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for idx, val in enumerate(dataset):
        # images, targets = next(iter(data_loader))
        images = val[0].unsqueeze(0).to(device)
        targets = [val[1]]
        loss_dict = model(images, targets)  # Returns losses and detections
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        writer.add_scalar('Loss/iter', losses.item(), counter)
        print(losses.item())
        counter+=1


    return counter


def fintune_shapes_ds():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    faster = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 4
    in_features = faster.roi_heads.box_predictor.cls_score.in_features
    faster.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    dataset = ShapesDataset('shapes')
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset, indices[-50:])
    faster.to(device)
    params = [p for p in faster.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    num_epochs = 5
    counter = 0

    writer = SummaryWriter('faster/shapes_3obj')


    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        counter = train_one_epoch(faster, optimizer, dataset, device, epoch, writer, counter,dataset_test, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
    torch.save(faster.state_dict(),'faster_max_3obj.pt')

    print("That's it!")



#fintune_shapes_ds()
test_display()

