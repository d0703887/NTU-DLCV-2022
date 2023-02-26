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
import random


# Dataset
def Hflip(image, label):
    num = random.uniform(0, 1)
    if num >= 0.5:
        image, label = transforms.functional.hflip(image), transforms.functional.hflip(label)
    return image, label


def Vflip(image, label):
    num = random.uniform(0, 1)
    if num >= 0.5:
        image, label = transforms.functional.vflip(image), transforms.functional.vflip(label)
    return image, label


def rotate(image, label):
    num = random.uniform(0, 1)
    angle = random.uniform(0, 45)
    if num >= 0.5:
        image, label = transforms.functional.rotate(image, angle), transforms.functional.rotate(label, angle)
    return image, label


class imageDataset(Dataset):
    def __init__(self, root, mode, transform=None):
        self.images = None
        self.label = None
        self.root = root
        if mode == 'train':
            self.len = 2000
        else:
            self.len = 257
        self.transform = transform
        self.filenames = [None] * self.len
        self.mode = mode
        self.toTensor = transforms.ToTensor()

        # fetch file names
        for name in os.listdir(root):
            if name.endswith('.jpg'):
                id = name.split('_')[0]
                self.filenames[int(id)] = id

    def __getitem__(self, index):
        image = self.toTensor(Image.open(os.path.join(self.root, self.filenames[index] + '_sat.jpg')))
        if self.mode == 'train':
            label = torch.tensor(np.load(os.path.join('/content/hw1/train', self.filenames[index] + '_mask.npy')))
        else:
            label = torch.tensor(np.load(os.path.join('/content/hw1/val', self.filenames[index] + '_mask.npy')))
        if self.transform is not None:
            if self.mode is 'train':
                image = self.transform(image)
                image, label = Hflip(image, label)
                image, label = Vflip(image, label)
                image, label = rotate(image, label)
            else:
                image = self.transform(image)
        return image, label

    def __len__(self):
        return self.len


# Config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 234234,
    'n_epochs': 3000,
    'batch_size': 10,
    'learning_rate': 0.00001,
    'early_stop': 1200,
    'save_path': '/content/drive/MyDrive/DLCV/hw1/segmentation/best_model.ckpt'
}


#Save checkpoint
def save_checkpoint(path, epoch: int, module, optimizer, scheduler):
    torch.save({
        'epoch': epoch,
        'model_state_dict': module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)


#Mean IOU
def mean_iou_score(pred, labels):
    mean_iou = 0
    for i in range(1, 7):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
        print('class #%d : %1.5f' % (i, iou))
    print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou


def IOU(model, val_loader, criterion, loss_record):
    prediction = np.empty((257, 512, 512))
    ground_truth = np.empty((257, 512, 512))
    count = 0
    for image, label in val_loader:
        image, label = image.to(device), label.to(device)
        with torch.no_grad():
            pred = model(image)['out']
            loss = criterion(pred, label)
            for i in range(len(pred)):
                prediction[count] = np.array(torch.argmax(pred[i], dim=0).cpu())
                ground_truth[count] = np.array(torch.argmax(label[i], dim=0).cpu())
                count += 1
        loss_record.append(loss.item())
    mean_iou_score(prediction, ground_truth)


# Train
def train(model, train_loader, val_loader, config, device, resume, path=None):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RAdam(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 4, gamma=0.5)
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
        model.train()  # Set your model to train mode.
        loss_record = []

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)
        for image, label in train_pbar:
            optimizer.zero_grad()  # Set gradient to zero.
            image, label = image.to(device), label.to(device)  # Move your data to device.
            pred = model(image)['out']
            loss = criterion(pred, label)
            loss.backward()  # Compute gradient(backpropagation).
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
        IOU(model, val_loader, criterion, loss_record)

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')

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
train_transform = nn.Sequential(
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #  torchvision.transforms.ColorJitter(),
    #  transforms.GaussianBlur(kernel_size=35),
    #  #new added
    #  transforms.RandomSolarize(0.3),
    #  transforms.RandomAdjustSharpness(0.3),
    #  transforms.RandomAutocontrast(0.4)

)
val_transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = imageDataset(root='/content/hw1_data/hw1_data/p2_data/train/', transform=train_transform, mode='train')
val_dataset = imageDataset(root='/content/hw1_data/hw1_data/p2_data/validation/', transform=val_transform, mode='val')
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)


# Model
model = torchvision.models.segmentation.deeplabv3_resnet50(
    weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT,
    weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT)
model.classifier[4] = nn.Conv2d(256, 7, kernel_size=(1, 1), stride=(1, 1))
model = model.to(device)


# training
model.load_state_dict(torch.load('/content/drive/MyDrive/DLCV/hw1/segmentation/res_0_74.ckpt'))
train(model, train_loader, val_loader, config, device, False,
      '/content/drive/MyDrive/DLCV/hw1/segmentation/cur_model.ckpt')
