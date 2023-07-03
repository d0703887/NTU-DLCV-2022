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
![method1](https://github.com/d0703887/NTU-DLCV-2022/assets/112916328/f95f4b80-cb5e-4d42-a8b2-6038ce6a21d2)


#### B.
![method2](https://github.com/d0703887/NTU-DLCV-2022/assets/112916328/cb2d8f56-7be6-44fc-8380-7c0f86774509)


## Problem 2: Diffusion Model
Implement a Conditional Diffusion Model from scratch and train it on MNIST-M dataset. Given conditional label 0-9, generate corresponding digit images.
### Model
I modified the source code in:

https://github.com/lucidrains/denoising-diffusion-pytorch

By feeding given conditional label into U-net blocks, we can generated conditional output basef on given input label.

### Dataset
MNIST-M, 56000 digit images with 10 classes(0-9).

### Result
![1688350379142](https://github.com/d0703887/NTU-DLCV-2022/assets/112916328/fd0fc2e7-8e7d-4d37-9cf1-de6e182c157f)

- Reverse Process

![1688350516701](https://github.com/d0703887/NTU-DLCV-2022/assets/112916328/6115eee6-41d4-4fb8-8d67-655a11060869)


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


