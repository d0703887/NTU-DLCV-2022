# Homework 2
## Problem 1: GAN
Implement Gans and train them on face dataset(from scratch).
### Model
#### A. DCGAN
I followed the model architecture and training setting in original DCGAN paper.
#### B. Alternative Method
The network architecture is pretty much the same as in A. except that I removed Batch Normalization and Sigmoid layer in discriminator following the recommendation in the paper “*Improved Training of Wasserstein GANs*”.
### Dataset
A subset of human face dataset CelebA with 38464 64x64 images.
### Result
#### A. 
![](https://hackmd.io/_uploads/rkDvssyF2.jpg)

#### B.
![](https://hackmd.io/_uploads/HJkujsJtn.jpg)

## Problem 2: Diffusion Model
Implement a Conditional Diffusion Model from scratch and train it on MNIST-M dataset. Given conditional label 0-9, generate corresponding digit images.
### Model
I modified the source code in:

https://github.com/lucidrains/denoising-diffusion-pytorch

By feeding given conditional label into U-net blocks, we can generated conditional output basef on given input label.

### Dataset
MNIST-M, 56000 digit images with 10 classes(0-9).

### Result
![](https://hackmd.io/_uploads/By86Aikt2.jpg)
- Reverse Process
![](https://hackmd.io/_uploads/HywBy2kF3.jpg)


## Problem 3: Domain Adaptation
Implement DANN for image classification on the digit datasets, and consider following 2 scenarios:

(a) MNIST-M → SVHN 
(b) MNIST-M → USPS
(srouce damain → target domain)

### Model
I followed the architecture and training setting in the original DANN paper.

### Dataset
- SVHN, 79431 digit images with 10 classes(0-9).
- USPS, 7438 grayscale digit images with 10 classes(0-9).

### Result


