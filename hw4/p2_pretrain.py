import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import random
from PIL import Image
from tqdm import tqdm


# Dataset
class ImageDataset(Dataset):
    def __init__(self, root, transform1, transform2):
        self.root = root
        self.transform1 = transform1
        self.transform2 = transform2

        # Fetch filenames
        self.filenames = []
        for image in os.listdir(root):
            self.filenames.append(image)

        self.len = len(self.filenames)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, self.filenames[index]))
        image1 = self.transform1(image)
        image2 = self.transform2(image)
        return image1, image2

    def __len__(self):
        return self.len


# Random Apply Transformation
class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


# Loss Function
def loss_fn(q, z):
    q = F.normalize(q, dim=-1, p=2)
    z = F.normalize(z, dim=-1, p=2)
    return 2 - 2 * (q * z).sum(dim=-1)


# EMA
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


# Update EMA
def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


# Model
class Resnet(nn.Module):
    def __init__(self, mode):
        super(Resnet, self).__init__()
        self.representation = torchvision.models.resnet50()
        self.representation.fc = nn.Identity()

        self.projection = nn.Sequential(
            nn.Linear(2048, 4096, bias=True),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 256)
        )

        if mode == 'online':
            self.prediction = nn.Sequential(
                nn.Linear(256, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(),
                nn.Linear(4096, 256)
            )
        elif mode == 'target':
            self.prediction = nn.Identity()

    def forward(self, x):
        repr = self.representation(x)
        proj = self.projection(repr)
        pred = self.prediction(proj)
        return repr, proj, pred


# Save Checkpoint
def checkpoint(online_model, target_model, optimizer, scheduler, epoch, path):
    torch.save({
        'online_model': online_model.state_dict(),
        'target_model': target_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch
    }, path)


# Train
def train(online_model, target_model, train_loader, config, device, resume, path, model_only):
    # Initialization
    n_epochs, epoch, warmup = config['n_epochs'], 0, config['warmup']
    optimizer = optim.Adam(online_model.parameters(), lr=0.2 * config['batch_size'] / 256, weight_decay=1.5e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000)
    target_EMA = EMA(0.99)

    # Warmup
    if not resume:
        warmup_optimizer = optim.Adam(online_model.parameters(), lr=config['learning_rate'], weight_decay=1.5e-6)
        online_model.train()
        target_model.train()
        while epoch < warmup:
            train_pbar = tqdm(train_loader, position=0, leave=True)
            for images1, images2 in train_pbar:
                warmup_optimizer.zero_grad()
                images1, images2 = images1.to(device), images2.to(device)

                _, _, pred1 = online_model(images1)
                _, _, pred2 = online_model(images2)

                with torch.no_grad():
                    _, proj1, _ = target_model(images2)
                    _, proj2, _ = target_model(images1)

                loss1 = loss_fn(pred1, proj1)
                loss2 = loss_fn(pred2, proj2)
                loss = (loss1 + loss2).mean()

                loss.backward()
                warmup_optimizer.step()
                update_moving_average(target_EMA, target_model, online_model)

                # Display current epoch number and loss on tqdm progress bar.
                train_pbar.set_description(f'Warmup Epoch [{epoch + 1}/{warmup}]')
                train_pbar.set_postfix({'loss': loss.detach().item()})

            epoch += 1

        epoch = 0
    else:
        ck = torch.load(path)
        online_model.load_state_dict(ck['online_model'])
        target_model.load_state_dict(ck['target_model'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        epoch = ck['epoch']

    # Training
    online_model.train()
    target_model.train()
    while epoch < n_epochs:
        train_pbar = tqdm(train_loader, position=0, leave=True)
        for images1, images2 in train_pbar:
            optimizer.zero_grad()
            images1, images2 = images1.to(device), images2.to(device)

            _, _, pred1 = online_model(images1)
            _, _, pred2 = online_model(images2)

            with torch.no_grad():
                _, proj1, _ = target_model(images2)
                _, proj2, _ = target_model(images1)

            loss1 = loss_fn(pred1, proj1)
            loss2 = loss_fn(pred2, proj2)
            loss = (loss1 + loss2).mean()

            loss.backward()
            optimizer.step()
            update_moving_average(target_EMA, target_model, online_model)

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        scheduler.step()
        epoch += 1
        checkpoint(online_model, target_model, optimizer, scheduler, epoch, path)


# Config
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
print("device: ", device)
config = {
    'n_epochs': 1000,
    'batch_size': 256,
    'learning_rate': 0.2,
    'warmup': 10
}

# Start Training
transform = transforms.Compose([
    transforms.ToTensor(),
    RandomApply(
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
        p=0.3
    ),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomHorizontalFlip(),
    RandomApply(
        transforms.GaussianBlur((3, 3), (1.0, 2.0)),
        p=0.2
    ),
    transforms.RandomResizedCrop((128, 128)),
    transforms.Normalize(
        mean=torch.tensor([0.485, 0.456, 0.406]),
        std=torch.tensor([0.229, 0.224, 0.225])
    )
])
train_dataset = ImageDataset('hw4_data/mini/train', transform, transform)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
online_model = Resnet('online').to(device)
target_model = Resnet('target').to(device)

train(online_model, target_model, train_loader, config, device, False, './p2.ckpt', False)


