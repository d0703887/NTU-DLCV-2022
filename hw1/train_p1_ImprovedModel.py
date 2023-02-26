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


# Dataset
class imageDataset(Dataset):
    def __init__(self, root, mode, transform=None):
        self.images = None
        self.label = None
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
    'seed': 234234,
    'n_epochs': 30,
    'batch_size': 50,
    'learning_rate': 0.00001,
    'early_stop': 1200,
    'save_path': '/content/drive/MyDrive/DLCV/hw1/classification/best_model.ckpt'
}


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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.5)
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


# Estsblishing dataset
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomResizedCrop((224, 224), scale=(0.6, 1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomErasing(),
    transforms.RandomRotation(degrees=25),
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = imageDataset(root='/content/hw1_data/hw1_data/p1_data/train_50/', transform=train_transform,
                             mode='train')
val_dataset = imageDataset(root='/content/hw1_data/hw1_data/p1_data/val_50/', transform=val_transform, mode='val')
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)


# training
model = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT)
model.classifier = nn.Sequential(
    nn.Linear(in_features=1280, out_features=50, bias=True)
)
model = model.to(device)
model.load_state_dict(torch.load('/content/drive/MyDrive/DLCV/hw1/classification/backbone_efficientnet_9_0_88.ckpt'))
train(model, train_loader, val_loader, config, device, False, '/content/drive/MyDrive/DLCV/hw1/classification/cur_model.ckpt')
