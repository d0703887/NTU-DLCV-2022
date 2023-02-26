import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import clip
import pandas as pd

import matplotlib.pyplot as plt


# Dataset
class ImageDataset (Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.filenames = []
        for file in os.listdir(self.root):
            self.filenames.append(file)
        self.len = len(self.filenames)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.filenames[index]))
        if self.transform is not None:
            img = self.transform(img)
        label = int(self.filenames[index].split('_')[0])
        return img, label, self.filenames[index]

    def __len__(self):
        return self.len


# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load('ViT-L/14')


# Prepare inputs
test_dataset = ImageDataset('hw3_data/p1_data/val', preprocess)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
labels = pd.read_json('hw3_data/p1_data/id2label.json', typ='dictionary').values.tolist()
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in labels]).to(device)
# text_inputs = torch.cat([clip.tokenize(f"No {c}, no score.") for c in labels]).to(device)


# Start
total_count = 0
correct_count = 0
for image, label, filename in test_loader:
    image, label = image.to(device), label.to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    for i in range(len(similarity)):
        total_count += 1
        values, indices = similarity[i].topk(5)
        if indices[0] == label[i]:
            correct_count += 1

        prob = []
        option = []
        print(filename[i], "\n\n")
        for j in range(5):
            option.append(f"a photo of a {labels[indices[j]]}")
            prob.append(values[j].item())

        fig, ax = plt.subplots()
        ax.barh(option, prob)
        for s in ['top', 'bottom', 'left', 'right']:
            ax.spines[s].set_visible(False)
        ax.xaxis.set_tick_params(pad=5)
        ax.yaxis.set_tick_params(pad=10)
        ax.invert_yaxis()
        for i in ax.patches:
            plt.text(i.get_width() + 0.2, i.get_y() + 0.5,
                     str(round((i.get_width()), 2)),
                     fontsize=10, fontweight='bold',
                     color='grey')
        ax.set_title(f'correct probability {prob[0]:.2f}',
                     loc='right', )

        plt.show()
    break

print(correct_count / total_count)




