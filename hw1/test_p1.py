# run test images for classification
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
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
        # fetch file names
        for file in os.listdir(test_path):
            self.filenames.append(file)
        self.len = len(self.filenames)

    def __getitem__(self, index):
        img = Image.open(os.path.join(test_path, self.filenames[index]))
        if self.transform is not None:
            img = self.transform(img)

        return img, self.filenames[index]

    def __len__(self):
        return self.len


# establish dataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device: ", device)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_dataset = imageDataset(transform=transform)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)


#model
model = torchvision.models.efficientnet_v2_s()
model.classifier = nn.Sequential(
    nn.Linear(in_features=1280, out_features=50, bias=True)
)
model = model.to(device)
model.load_state_dict(torch.load('./p1_improved_model.ckpt', map_location=torch.device(device)))
L = []
model.eval()
for image, filename in test_loader:
    image, filename = image.to(device), filename
    with torch.no_grad():
        pred = model(image).cpu()
    for i in range(len(pred)):
        L.append([filename[i], str(torch.argmax(pred[i]).item())])

write = pd.DataFrame(L, columns=('filename', 'label'))
write.to_csv(output_path, index=False)
