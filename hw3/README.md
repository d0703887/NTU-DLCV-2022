# Homework 3
## Problem 1: Zero-Shot Image Classification with CLIP 
Evaluate the pretrained CLIP on Image Classification task. (Try different prompt texts and compare their corresponding result)

## Problem 2: Image Captioning with Image and Language Model
### A. Implement a Vision and Language(VL) model(lmitied to tranformer-based model) for Image Captioning
For encoder, I used the pretrained ViT-L. For decoder, I implemented the decoder block in the way as the paper ***CPTR: Full Transformer Network For Image Captioning*** proposed. The output of decoder is fed into a Linear Layer to predict the probability for each word token.

* Architecture of decoder block

![tmp](https://github.com/d0703887/NTU-DLCV-2022/assets/112916328/3421a0fc-6aa6-469b-a67d-fd6af84fc49e)

### B. Implement CLIPScore for evaluation.


### Dataset
Over 12000 images and each image is paired with ultiple captions.

## Problem 3: Visualizing Attention in Image Captioning Model
**1.**

