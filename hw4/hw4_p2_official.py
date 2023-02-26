import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from byol_pytorch import BYOL
from torchvision import models
from torchvision import transforms

# Dataset
class ImageDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        # Fetch filenames
        self.filenames = []
        for image in os.listdir(root):
            self.filenames.append(image)

        self.len = len(self.filenames)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, self.filenames[index]))
        image = self.transform(image)
        return image

    def __len__(self):
        return self.len

device = 'cuda:2'
train_transform = transforms.Compose([
    transforms.ToTensor()
])
train_dataset = ImageDataset('./hw4_data/mini/train', train_transform)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
model = models.resnet50()
learner = BYOL(model, image_size=128, hidden_layer='avgpool').to(device)
optimizer = torch.optim.Adam(learner.parameters(), lr=3e-4)
for i in range(1000):
    for images in train_loader:
        images = images.to(device)
        loss = learner(images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        learner.update_moving_average()
    print(f'Epoch: {i}, loss: {loss.item()}')
    if i % 10 == 0:
        torch.save(model.state_dict(), './p2_' + str(i) + '.ckpt')

