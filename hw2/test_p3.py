import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd
import numpy as np
############################
from torch.autograd import Function
import argparse


# Argument
parser = argparse.ArgumentParser()
parser.add_argument("-t", type=str)
parser.add_argument("-o", type=str)
args = parser.parse_args()
test_path = args.t
output_path = args.o
print('testing image path: ', test_path)
print('output file path: ', output_path)


# Dataset
class DigitDataset(Dataset):
    def __init__ (self, transform):
        self.transform = transform
        self.filenames = []
        for images in os.listdir(test_path):
            self.filenames.append(images)
        self.len = len(self.filenames)

    def __getitem__(self, index):
        image = Image.open(os.path.join(test_path, self.filenames[index]))
        if self.transform is not None:
            image = self.transform(image)
        return image, self.filenames[index]

    def __len__(self):
        return self.len


# Gradient Reversal layer
class GRL(Function):
    def forward(self, input):
        return input

    def backward(self, grad):
        grad = grad.neg()
        return grad, None


# DANN
class DANN(nn.Module):
  def __init__(self, in_channel):
    super(DANN, self).__init__()
    self.feature = nn.Sequential(
        # (32, 32, in_channel)
        nn.Conv2d(in_channel, 32, kernel_size=5, stride=1, padding=0),
        # (28, 28, 32)
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        # (14, 14, 32)
        nn.Conv2d(32, 48, kernel_size=5, stride=1, padding=0),
        nn.ReLU(),
        # (10, 10, 48)
        nn.MaxPool2d(kernel_size=2)
        # (5, 5, 48)
    )

    self.classifier = nn.Sequential(
        nn.Linear(1200, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )

    self.domain = nn.Sequential(
        nn.Linear(1200, 100),
        nn.ReLU(),
        nn.Linear(100, 1),
        nn.Sigmoid()
    )

  def forward(self, x):
    feature = self.feature(x)
    feature = feature.view(-1, 1200)
    reverse = GRL.apply(feature)
    label = self.classifier(feature)
    domain = self.domain(reverse)

    return label, domain


# Determine is USPS or SVHN
for image in os.listdir(test_path):
    img = np.array(Image.open(os.path.join(test_path, image)))
    if len(img.shape) == 2:
        mode = 'USPS'
    elif len(img.shape) == 3:
        mode = 'SVHN'
    break


# Establish Dataset
if mode == 'USPS':
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(0.456, 0.224),
    ])
elif mode == 'SVHN':
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
testDataset = DigitDataset(transform=test_transform)
test_loader = DataLoader(testDataset, batch_size=60, shuffle=False)


# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device: ", device)
if mode == 'USPS':
    model = DANN(1).to(device)
    model.load_state_dict(torch.load('./p3_USPS_DANN.ckpt', map_location=torch.device(device))['model_state_dict'])
elif mode == 'SVHN':
    model = DANN(3).to(device)
    model.load_state_dict(torch.load('./p3_SVHN_DANN.ckpt', map_location=torch.device(device))['model_state_dict'])


# Start inferencing
L = []
model.eval()
for image, filename in test_loader:
    image= image.to(device)
    with torch.no_grad():
        clas, domain = model(image)
        clas = clas.to('cpu')
    for i in range(len(clas)):
        L.append([filename[i], str(torch.argmax(clas[i]).item())])

write = pd.DataFrame(L, columns=('image_name', 'label'))
write.to_csv(output_path, index=False)



