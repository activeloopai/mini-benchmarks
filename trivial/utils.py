import deeplake
import numpy as np
import math
import time
import os
from tqdm import tqdm
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

ds = deeplake.load('./dataset/coco-val')

WIDTH = 128
HEIGHT = 128

# These are the classes we care about and they will be remapped to 0,1,2,3,4,6 in the model
CLASSES_OF_INTEREST = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'traffic light', 'stop sign']

# The classes of interest correspond to the following array values in the current dataset
INDS_OF_INTEREST = [ds.categories.info.class_names.index(item) for item in CLASSES_OF_INTEREST]


# Augmentation pipeline using Albumentations
tform_train = A.Compose([
    A.RandomSizedBBoxSafeCrop(width=WIDTH, height=HEIGHT, erosion_rate=0.2),
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels', 'bbox_ids'], min_area=16, min_visibility=0.6)) # 'label_fields' and 'box_ids' are all the fields that will be cut when a bounding box is cut.


# Transformation function for pre-processing the deeplake sample before sending it to the model
def transform_train(sample_in):

    # Convert any grayscale images to RGB
    image = sample_in['images']
    if not isinstance(image, np.ndarray):
      image = np.array(image)
    shape = image.shape

    if len(shape) == 2:
      image = image.reshape(shape[0], shape[1], 1)
      shape = image.shape

    if shape[2] == 1:
        image = np.repeat(image, int(3/shape[2]), axis = 2)

    # Convert boxes to Pascal VOC format
    boxes = coco_2_pascal(sample_in['boxes'], shape)
    
    # Filter only the labels that we care about for this training run
    labels_all = sample_in['categories']
    indices = [l for l, label in enumerate(labels_all) if label in INDS_OF_INTEREST]
    labels_filtered = labels_all[indices]
    labels_remapped = [INDS_OF_INTEREST.index(label) for label in labels_filtered]
    boxes_filtered = boxes[indices,:]
    
    # Make sure the number of labels and boxes is still the same after filtering
    assert(len(labels_remapped)) == boxes_filtered.shape[0]

    # Pass all data to the Albumentations transformation
    transformed = tform_train(image = image, 
                              bboxes = boxes_filtered, 
                              bbox_ids = np.arange(boxes_filtered.shape[0]),
                              class_labels = labels_remapped,
                              )

    # Convert boxes and labels from lists to torch tensors, because Albumentations does not do that automatically.
    # Be very careful with rounding and casting to integers, becuase that can create bounding boxes with invalid dimensions
    labels_torch = torch.tensor(transformed['class_labels'], dtype = torch.int64)
    
    boxes_torch = torch.zeros((len(transformed['bboxes']), 4), dtype = torch.int64)
    for b, box in enumerate(transformed['bboxes']):
        boxes_torch[b,:] = torch.tensor(np.round(box))


    # Put annotations in a separate object
    target = {'labels': labels_torch, 'boxes': boxes_torch}
    
    return transformed['image'], target


# Conversion script for bounding boxes from coco to Pascal VOC format
def coco_2_pascal(boxes, shape):
    # Convert bounding boxes to Pascal VOC format and clip bounding boxes to make sure they have non-negative width and height
    
    return np.stack((np.clip(boxes[:,0], 0, None), np.clip(boxes[:,1], 0, None), np.clip(boxes[:,0]+np.clip(boxes[:,2], 1, None), 0, shape[1]), np.clip(boxes[:,1]+np.clip(boxes[:,3], 1, None), 0, shape[0])), axis = 1)

def collate_fn(batch):
    return tuple(zip(*batch))

# Helper function for loading the model
def get_model_object_detection(num_classes):
    # Load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
    
# Helper function for training for 1 epoch
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()

    start_time = time.time()
    for i, data in enumerate(data_loader):
        images = list(image.to(device) for image in data[0])
        targets = [{k: v.to(device) for k, v in t.items()} for t in data[1]]
                
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        
        # Print performance statistics
        if i%10 ==0:
            batch_time = time.time()
            speed = (i+1)/(batch_time-start_time)
            print('[%5d] loss: %.3f, speed: %.2f' %
                  (i, loss_value, speed))

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict)
            break

        optimizer.zero_grad()

        losses.backward()
        optimizer.step()


def train(train_loader):     
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    model = get_model_object_detection(len(CLASSES_OF_INTEREST))
    model.to(device)

    # Specify the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # Train the model for 1 epoch
    num_epochs = 1

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        
    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        print("------------------ Training Epoch {} ------------------".format(epoch+1))
        train_one_epoch(model, optimizer, train_loader, device)
        lr_scheduler.step()
        
        # --- Insert Testing Code Here ---

def iterate(train_loader):
    for el in tqdm(train_loader):
        pass


