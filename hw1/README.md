# Homework 1
## Problem 1: Image Classification
### Task 
#### A. Train a CNN Classification model from scratch
Model architecture

![modeA](https://user-images.githubusercontent.com/112916328/221918607-c76e4af1-69f8-42f0-a31f-d19b06ece203.jpg)

#### B. Try Alternative models/methods
I fine-tuned the EfficientNetV2 which was pre-trained on ImageNet-1k dataset. 

### Dataset
25000 colored 32x32 images with 50 classes.

### Result
* #### t-SNE analysis on Task B (2500 images)
![](https://i.imgur.com/brFzhns.jpg)
* #### Accuracy
Task A: 0.762  
Task B: 0.88

## Problem 2: Semantic Segmentation
### Task
#### A. Implement VGG16 + FCN32s model from scratch
Model architecture

![](https://i.imgur.com/dn5CP7Y.jpg)

#### B. Try Alternative models/methods
I fine-tuned the DeepLabV3 model which was trained on subset of COCO dataset.

### Dataset
2300 512x512 images with 7 class labels (includuing background)

### Result
* #### Mean IOU (averaging over 7 classes)
Task A: 0.642  
Task B: 0.74
