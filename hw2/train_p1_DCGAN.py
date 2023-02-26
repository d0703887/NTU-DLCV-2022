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


#Dataset
class FaceDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.filenames = []
        # fetch images
        for file in os.listdir(root):
            self.filenames.append(file)
        self.len = len(self.filenames)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, self.filenames[index]))

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return self.len


# DCGAN: Generator
class Generator(nn.Module):
    def __init__(self, z):
        super(Generator, self).__init__()
        self.z = z
        self.generator = nn.Sequential(
            # input noise with dimension z
            nn.ConvTranspose2d(z, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # (4, 4, 512)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # (8, 8, 256)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # (16, 16, 128)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # (32, 32, 128)
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # (64, 64, 3)
        )

    def forward(self, input):
        return self.generator(input)


# DCGAN: Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            # (64, 64, 3)
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (32, 32, 64)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (16, 16, 128)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (8, 8, 256)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (4, 4, 512)
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # (1, 1, 1)
        )

    def forward(self, input):
        return self.discriminator(input)


# Weight initialization / Save checkpoint
def weight_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


def save_checkpoint(pathG, pathD, epoch: int, G, D, optimizerG, optimizerD):
    # save generator
    torch.save({
        'epoch': epoch,
        'model_state_dict': G.state_dict(),
        'optimizer_state_dict': optimizerG.state_dict()
    }, pathG)
    # save discriminator
    torch.save({
        'epoch': epoch,
        'model_state_dict': D.state_dict(),
        'optimizer_state_dict': optimizerD.state_dict()
    }, pathD)


# Train
def train(G, D, train_loader, val_loader, config, device, resume, pathG, pathD):
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(D.parameters(), lr=config['learning_rate'], betas=(0.5, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=config['learning_rate'], betas=(0.5, 0.999))
    epoch = 0

    n_epochs = config['n_epochs']

    if resume is True:
        checkpointG = torch.load(pathG)
        checkpointD = torch.load(pathD)
        epoch = checkpointG['epoch']
        G.load_state_dict(checkpointG['model_state_dict'])
        optimizerG.load_state_dict(checkpointG['optimizer_state_dict'])
        D.load_state_dict(checkpointD['model_state_dict'])
        optimizerD.load_state_dict(checkpointD['optimizer_state_dict'])

    while epoch < n_epochs:
        ##########################
        # Train Discriminator #
        ##########################

        train_pbar = tqdm(train_loader, position=0, leave=True)
        for image in train_pbar:
            batch = len(image)
            # Feed real-batch
            optimizerD.zero_grad()
            image = image.to(device)
            label = ((1.2 - 0.7) * torch.rand(batch) + 0.7).to(device)  # true label
            # label = torch.full((batch, ), 1.0, device=device)
            pred = D(image).view(-1)
            real_loss = criterion(pred, label)
            real_loss.backward()
            D_x = pred.mean().item()

            # feed fake-batch
            noise = torch.randn(batch, config['noise_dimension'], 1, 1, device=device)
            fake = G(noise)
            label = ((0.3 - 0.0) * torch.rand(batch)).to(device)
            # label.fill_(0)
            pred = D(fake.detach()).view(-1)
            fake_loss = criterion(pred, label)
            fake_loss.backward()
            D_G_z = pred.mean().item()
            total_D_loss = real_loss + fake_loss
            optimizerD.step()

            #####################
            # Train Genrator #
            #####################
            optimizerG.zero_grad()
            label.fill_(1)
            pred = D(fake).view(-1)
            G_loss = criterion(pred, label)
            G_loss.backward()
            D_G_z2 = pred.mean().item()
            optimizerG.step()

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'D loss': total_D_loss.detach().item()})

        save_checkpoint(pathG, pathD, epoch, G, D, optimizerG, optimizerD)
        epoch += 1

        if epoch % 10 == 0:
            print(
                f'Epoch [{epoch + 1}/{n_epochs}]: D loss: {total_D_loss:.4f}, G loss: {G_loss:.4f}, D(x): {D_x:.4f}, D(G(z1))/D(G(z2)): {D_G_z:.4f}/{D_G_z2:.4f}')
            # generate 1000 images
            pred = G(fixed_noise).detach().cpu()
            for i in range(1000):
                tmp = transforms.ToPILImage()(pred[i])
                tmp.save(os.path.join('/content/output', str(i) + '.png'))


# Config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 3407,
    'noise_dimension': 100,
    'n_epochs': 800,
    'batch_size': 128,
    'learning_rate': 0.0002,
    'save_path': '/content/drive/MyDrive/DLCV/hw2/gan/best_model.ckpt'
}


# Establishing dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = FaceDataset('/content/hw2_data/face/train/', transform)
val_dataset = FaceDataset('/content/hw2_data/face/val', transform)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)


# GANs
torch.manual_seed(config['seed'])
G = Generator(config['noise_dimension']).to(device)
D = Discriminator().to(device)
G.apply(weight_init)
D.apply(weight_init)
fixed_noise = torch.randn(1000, config['noise_dimension'], 1, 1, device=device)
print(G)
print(D)


def norm_ip(img):
    low = float(img.min())
    high = float(img.max())
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))


checkpointG = torch.load("/content/drive/MyDrive/DLCV/hw2/gan/cur_G.ckpt")
G.load_state_dict(checkpointG['model_state_dict'])
pred = G(fixed_noise).detach().cpu()
for i in range(1000):
    norm_ip(pred[i])
    tmp = transforms.ToPILImage()(pred[i])
    tmp.save(os.path.join('/content/output', str(i) + '.png'))

# Start training
train(G, D, train_loader, val_loader, config, device, True, '/content/drive/MyDrive/DLCV/hw2/gan/cur_G.ckpt', '/content/drive/MyDrive/DLCV/hw2/gan/cur_D.ckpt')
