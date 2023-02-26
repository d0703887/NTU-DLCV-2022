# Run on Google colab
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
# Import for PCA, t-SNE
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import manifold


# Dataset
class imageDataset(Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        if mode == 'train':
            self.len = 22500
        else:
            self.len = 2500
        self.transform = transform
        self.filenames = [None] * self.len
        self.mode = mode
        # fetch file names
        if mode == 'train':
            for name in os.listdir(root):
                if name.endswith('.png'):
                    label = name.split('_')[0]
                    id = name.split('_')[1].split('.')[0]
                    self.filenames[int(id) + int(label) * 450] = name;
        elif mode == 'val':
            for name in os.listdir(root):
                if name.endswith('.png'):
                    label = name.split('_')[0]
                    id = name.split('_')[1].split('.')[0]
                    self.filenames[int(id) + int(label) * 50 - 450] = name;

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, self.filenames[index]))
        ans = self.filenames[index].split('_')[0]
        label = np.zeros((50))
        label[int(ans)] = 1
        label = torch.tensor(label)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.len


# Config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 234,
    'n_epochs': 3000,
    'batch_size': 60,
    'learning_rate': 0.0001,
    'early_stop': 1200,
    'save_path': '/content/drive/MyDrive/DLCV/hw1/classification/best_model.ckpt'
}


# Resblock
class resBlock(nn.Module):
    def __init__(self, in_channel, out_channel, downSample):
        super(resBlock, self).__init__()
        self.downSample = downSample
        self.down = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel)
        )
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(out_channel)
        self.batchNorm2 = nn.BatchNorm2d(out_channel)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        shortcut = x.clone()
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = self.activation(x)
        if self.downSample is True:
            shortcut = self.down(shortcut)
        x = x + shortcut
        x = self.activation(x)
        return x


# Nerual Network
class MyCNN(nn.Module):
    def __init__(self, resBlock):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Sequential(
            resBlock(3, 64, True),
            resBlock(64, 64, False),

            nn.MaxPool2d(2),  # 32x32

            resBlock(64, 128, True),
            resBlock(128, 128, False),
            resBlock(128, 128, False),

            nn.MaxPool2d(2),  # 16x16

            resBlock(128, 256, True),
            resBlock(256, 256, False),

            nn.MaxPool2d(2),  # 8x8

            resBlock(256, 512, True),
            resBlock(512, 512, False),

            nn.MaxPool2d(2),  # 4x4
        )

        self.GlobalMax = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(512, 50),
            nn.Dropout(0.2),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.GlobalMax(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x


# Save checkpoint
def save_checkpoint(path, epoch: int, module, optimizer, scheduler):
    torch.save({
        'epoch': epoch,
        'model_state_dict': module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)


# Train
def train(model, train_loader, val_loader, config, device, resume, path=None):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RAdam(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 40, gamma=0.2)
    epoch = 0
    # Check if we want to resume the training progress
    if resume is True:
        # load checkpoint
        checkpoint = torch.load(path)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], 100, 0, 0

    while epoch < n_epochs:
        val_count = 0
        train_count = 0
        model.train()  # Set your model to train mode.
        loss_record = []

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)
        for image, label in train_pbar:
            optimizer.zero_grad()  # Set gradient to zero.
            image, label = image.to(device), label.to(device)  # Move your data to device.
            pred = model(image)
            loss = criterion(pred, label)
            loss.backward()
            for i in range(len(pred)):
                if (torch.argmax(pred[i]) == torch.argmax(label[i])):
                    train_count += 1  # Compute gradient(backpropagation).
            optimizer.step()  # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        scheduler.step()  # decrease learning rate
        mean_train_loss = sum(loss_record) / len(loss_record)

        model.eval()  # Set your model to evaluation mode.
        loss_record = []
        for image, label in val_loader:
            image, label = image.to(device), label.to(device)
            with torch.no_grad():
                pred = model(image)
                loss = criterion(pred, label)
                for i in range(len(pred)):
                    if (torch.argmax(pred[i]) == torch.argmax(label[i])):
                        val_count += 1

        loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(
            f'Epoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}, Train accuracy: {train_count / 22500:.4f}, Valid accuracy: {val_count / 2500:.4f}')

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])  # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            print()
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return
        save_checkpoint(path, epoch + 1, model, optimizer, scheduler)
        epoch += 1


# Establishing dataset
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((72, 72)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomResizedCrop((72, 72), scale=(0.6, 1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomErasing(),
    transforms.RandomRotation(degrees=25),
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((72, 72)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
train_dataset = imageDataset(root='/content/hw1_data/hw1_data/p1_data/train_50/', transform=train_transform,
                             mode='train')
val_dataset = imageDataset(root='/content/hw1_data/hw1_data/p1_data/val_50/', transform=val_transform, mode='val')
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)
dim_loader = DataLoader(val_dataset, 50, shuffle=False)

# Hook
def get_feature(feature):
    def hook(model, input, output):
        feature.append(output.detach())

    return hook


# PCA / t - SNE
def dim(dim_loader, model, per, mode):
    model.eval()
    total = per * 50
    la = torch.zeros(total)
    count = 0
    feature = []
    model.GlobalMax.register_forward_hook(get_feature(feature))
    color = []
    for i in range(50):
        color.append(np.random.choice(range(256), size=3) / 256)

    for image, label in dim_loader:
        image, label = image.to(device), label.to(device)
        for i in range(per):
            la[count] = torch.argmax(label[i])
            count += 1
        with torch.no_grad():
            image = image[0:per]
            pred = model(image)
    tmp = torch.empty(total, 512)
    for i in range(len(feature)):
        for j in range(per):
            tmp[i * per + j] = torch.squeeze(torch.tensor(feature[i][j])).cpu()
    index = torch.randperm(total)
    feature = torch.empty(total, 512)
    for i in range(total):
        feature[i] = tmp[index[i]]

    la = torch.tensor(la)
    if mode == 'pca':
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(feature)
    else:
        tsne = manifold.TSNE(perplexity=30, n_components=2, init='pca', verbose=1, learning_rate='auto', n_iter=20000,
                             n_iter_without_progress=100000)
        reduced = tsne.fit_transform(feature)
    for i in range(len(reduced)):
        plt.scatter(x=reduced[i, 0], y=reduced[i, 1], color=color[(int)(la[i].item())])
    plt.show()


# Start training
model = MyCNN(resBlock).to(device)
model.load_state_dict(torch.load('/content/drive/MyDrive/DLCV/hw1/classification/0_762.ckpt'))
train(model, train_loader, val_loader, config, device, False,
      '/content/drive/MyDrive/DLCV/hw1/classification/cur_model.ckpt')