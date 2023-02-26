# run test images for classification
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import argparse

# Argument
parser = argparse.ArgumentParser()
parser.add_argument("-t", type=str)
parser.add_argument("-o", type=str)
args = parser.parse_args()
test_path = args.t
output_path = args.o
print('testing images path: ', test_path)
print('output file path: ', output_path)


# dataset
class imageDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.filenames = []
        self.toTensor = transforms.ToTensor()
        # fetch file names
        for name in os.listdir(test_path):
           if name.endswith('.jpg'):
            self.filenames.append(name)
        self.len = len(self.filenames)

    def __getitem__(self, index):
        img = self.toTensor(Image.open(os.path.join(test_path, self.filenames[index])))
        if self.transform is not None:
            img = self.transform(img)
        return img, self.filenames[index]

    def __len__(self):
        return self.len


# establish dataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device: ", device)
transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_dataset = imageDataset(transform=transform)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)


model = torchvision.models.segmentation.deeplabv3_resnet50(aux_loss=True, weights_backbone=None)
model.classifier[4] = nn.Conv2d(256, 7, kernel_size=(1, 1), stride=(1, 1))
model = model.to(device)
model.load_state_dict(torch.load('./p2_improved_model.ckpt', map_location=torch.device(device)))
model.eval()
for image, filename in test_loader:
    image, filename = image.to(device), filename
    with torch.no_grad():
        pred = model(image)['out']
        for i in range(len(pred)):
            tmp = torch.argmax(pred[i], dim=0).cpu()
            result = np.zeros((512, 512, 3))
            result[tmp == 0] = np.array([0, 0, 0])
            result[tmp == 1] = np.array([255, 255, 255])
            result[tmp == 2] = np.array([0, 0, 255])
            result[tmp == 3] = np.array([0, 255, 0])
            result[tmp == 4] = np.array([255, 0, 255])
            result[tmp == 5] = np.array([255, 255, 0])
            result[tmp == 6] = np.array([0, 255, 255])
            img = Image.fromarray(np.uint8(result))
            img.save(os.path.join(output_path, filename[i].replace('.jpg', '.png')))
