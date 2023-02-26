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
from tqdm import tqdm
import pandas as pd
############################
from torch.autograd import Function


# Dataset
class MnistDataset(Dataset):
    def __init__(self, root, transform, mode):
        self.root = root
        self.transform = transform
        self.channel_norm = channel_norm
        if mode == 'train':
            self.label = pd.read_csv('/content/hw2_data/digits/mnistm/train.csv').values.tolist()
        elif mode == 'val':
            self.label = pd.read_csv('/content/hw2_data/digits/mnistm/val.csv').values.tolist()
        self.len = len(self.label)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, self.label[index][0]))
        ans = self.label[index][1]
        label = np.zeros((10))
        label[int(ans)] = 1
        label = torch.tensor(label)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return self.len


class USPSDataset(Dataset):
    def __init__(self, root, transform, mode):
        self.root = root
        self.transform = transform
        if mode == 'train':
            self.label = pd.read_csv('/content/hw2_data/digits/usps/train.csv').values.tolist()
        elif mode == 'val':
            self.label = pd.read_csv('/content/hw2_data/digits/usps/val.csv').values.tolist()
        self.len = len(self.label)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, self.label[index][0]))
        ans = self.label[index][1]
        label = np.zeros((10))
        label[int(ans)] = 1
        label = torch.tensor(label)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return self.len


class SVHNDataset(Dataset):
    def __init__(self, root, transform, mode):
        self.root = root
        self.transform = transform
        if mode == 'train':
            self.label = pd.read_csv('/content/hw2_data/digits/svhn/train.csv').values.tolist()
        elif mode == 'val':
            self.label = pd.read_csv('/content/hw2_data/digits/svhn/val.csv').values.tolist()
        self.len = len(self.label)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, self.label[index][0]))
        ans = self.label[index][1]
        label = np.zeros((10))
        label[int(ans)] = 1
        label = torch.tensor(label)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return self.len


# Classifier
class Classifier(nn.Module):
    def __init__(self, in_channel):
        super(Classifier, self).__init__()
        self.main = nn.Sequential(
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
        self.fc = nn.Sequential(
            nn.Linear(1200, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 1200)
        x = self.fc(x)
        return x


# Gradient reversal layer
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


# Save checkpoint
def save_checkpoint(path, epoch: int, module, optimizer):
    torch.save({
        'epoch': epoch,
        'model_state_dict': module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)


# Train
def train(model, source_train_loader, target_train_loader, target_val_loader, config, device, resume, path=None):
    class_criterion = torch.nn.CrossEntropyLoss()
    domain_criterion = torch.nn.BCELoss()
    optimizer = torch.optim.RAdam(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    epoch = 0
    target_train_count = 0
    target_train_batch = len(target_train_loader)
    target_val_len = len(target_val_loader.dataset)

    if resume is True:
        # load checkpoint
        checkpoint = torch.load(path)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    n_epochs, best_loss, step, = config['n_epochs'], 100, 0

    while epoch < n_epochs:
        target_val_count = 0
        model.train()  # Set your model to train mode.
        train_pbar = tqdm(source_train_loader, position=0, leave=True)

        for image, label in train_pbar:
            # train with source domain images
            optimizer.zero_grad()
            image, label = image.to(device), label.to(device)
            domain_label = torch.zeros((len(image), 1)).float().to(device)
            clas, domain = model(image)
            source_class_loss = class_criterion(clas, label)
            source_domain_loss = domain_criterion(domain, domain_label)

            # train with target domain images
            if target_train_count % target_train_batch == 0:
                target_iter = iter(target_train_loader)
            image, label = target_iter.next()
            target_train_count += 1
            image, label = image.to(device), label.to(device)
            domain_label = torch.ones((len(image), 1)).float().to(device)
            clas, domain = model(image)
            target_domain_loss = domain_criterion(domain, domain_label)

            loss = target_domain_loss + source_class_loss + source_domain_loss
            loss.backward()
            optimizer.step()

        # Validation set on target domain
        model.eval()  # Set your model to evaluation mode.
        for image, label in target_val_loader:
            image, label = image.to(device), label.to(device)
            with torch.no_grad():
                pred, domain = model(image)
                for i in range(len(pred)):
                    if (torch.argmax(pred[i]) == torch.argmax(label[i])):
                        target_val_count += 1

        print(
            f'Epoch [{epoch + 1}/{n_epochs}]: Target Valid accuracy: {target_val_count / target_val_len:.4f}({target_val_count}/{target_val_len})')

        save_checkpoint(path, epoch + 1, model, optimizer)
        epoch += 1


# Config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 234,
    'n_epochs': 3000,
    'batch_size': 60,
    'learning_rate': 0.00001,
    'early_stop': 1200,
}


# Establishing dataset
class channel_norm(nn.Module):
    def __init__(self):
        super(channel_norm, self).__init__()

    def forward(self, x):
        x = torch.mean(x, dim=0)
        x = x[None, :, :]
        return x


usps_train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    channel_norm(),
    transforms.Normalize(0.456, 0.224),
    # transforms.RandomResizedCrop((32, 32), scale=(0.6, 1)),
    # transforms.RandomErasing(),
    # transforms.RandomRotation(degrees=25),
])
usps_val_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(0.456, 0.224),
])
usps_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(0.456, 0.224),
    # transforms.RandomResizedCrop((32, 32), scale=(0.6, 1)),
    # transforms.RandomErasing(),
    # transforms.RandomRotation(degrees=25),
])
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomResizedCrop((32, 32), scale=(0.6, 1)),
    transforms.RandomErasing(),
    transforms.RandomRotation(degrees=25),
])
val_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Mnist dataset
# mnist = MnistDataset("/content/hw2_data/digits/mnistm/data", usps_train_transform, 'train')
# Mnist_train = DataLoader(mnist, batch_size=config['batch_size'], shuffle=True)
# mnist = MnistDataset('/content/hw2_data/digits/mnistm/data', val_transform, 'val')
# Mnist_val = DataLoader(mnist, batch_size=config['batch_size'], shuffle=True)
# SVHN dataset
# svhn = SVHNDataset("/content/hw2_data/digits/svhn/data", train_transform, 'train')
# SVHN_train = DataLoader(svhn, batch_size=config['batch_size'], shuffle=True)
# svhn = SVHNDataset("/content/hw2_data/digits/svhn/data", val_transform, 'val')
# SVHN_val = DataLoader(svhn, batch_size=config['batch_size'], shuffle=True)
# USPS dataset
# usps = USPSDataset('/content/hw2_data/digits/usps/data', usps_transform, 'train')
# USPS_train = DataLoader(usps, batch_size=config['batch_size'], shuffle=True)
usps = USPSDataset('/content/hw2_data/digits/usps/data', usps_val_transform, 'val')
USPS_val = DataLoader(usps, batch_size=config['batch_size'], shuffle=True)

# Testing
def test(model, dataloader, device, path=None):
    ck = torch.load(path)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    total_count = len(dataloader.dataset)
    correct_count = 0
    for image, label in dataloader:
        image, label = image.to(device), label.to(device)
        pred, domain = model(image)
        for i in range(len(pred)):
            if torch.argmax(pred[i]) == torch.argmax(label[i]):
                correct_count += 1
    print(f"Accuracy: {correct_count / total_count:.4f} ({correct_count}/{total_count})")


# Start training
model = DANN(1).to(device)
ck = torch.load('/content/drive/MyDrive/DLCV/hw2/DANN/Source2Target/0_76.ckpt')
model.load_state_dict(ck['model_state_dict'])
train(model, Mnist_train, USPS_train, USPS_val, config, device, False,
      '/content/drive/MyDrive/DLCV/hw2/DANN/Source2Target/cur_model.ckpt')