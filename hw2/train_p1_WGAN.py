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
from torch.autograd import Variable
from torch.autograd import grad

# Dataset ######################################
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


# Resblock ######################################
class Resblock(nn.Module):
    def __init__(self, in_dim, out_dim, resample, h, w):
        super(Resblock, self).__init__()
        if resample == 'up':
            self.shortcut = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1)
            self.main = nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(),
                nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )
        elif resample == 'down':
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=4, stride=2, padding=1),
                nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
            )
            self.main = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                nn.AvgPool2d(kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )

    def forward(self, x):
        short = self.shortcut(x)
        x = self.main(x)
        return x + short


# WGAN: Generator ######################################
class Generator(nn.Module):
    def __init__(self, z, Resblock):
        super(Generator, self).__init__()
        self.z = z
        # input noise with dimension z
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

    def forward(self, x):
        x = self.generator(x)
        return x


# WGAN: Discriminator ######################################


class Discriminator(nn.Module):
    def __init__(self, Resblock):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            # (64, 64, 3)
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (32, 32, 64)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (16, 16, 128)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (8, 8, 256)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (4, 4, 512)
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # (1, 1, 1)
        )

    def forward(self, input):
        return self.discriminator(input)


# Save checkpoint / Normalize output image ######################################
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


def norm_ip(img):
    low = float(img.min())
    high = float(img.max())
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))


# Compute gradient penalty ######################################
def compute_gradient_penalty(D, real, fake, device):
    # interpolation
    shape = [fake.size(0)] + [1] * (fake.dim() - 1)
    alpha = torch.rand(shape, device=device).to(device)
    z = fake + alpha * (real - fake)

    # gradient penalty
    z = Variable(z, requires_grad=True).to(device)
    o = D(z)
    g_o = torch.ones(o.size(), device=device)
    g = grad(o, z, grad_outputs=g_o, create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()

    return gp


# Train ######################################
def train(G, D, train_loader, config, device, resume, pathG, pathD):
    optimizerD = optim.Adam(D.parameters(), lr=config['learning_rate'], betas=(0.5, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=config['learning_rate'], betas=(0.5, 0.999))
    epoch = 0
    n_epochs = config['n_epochs']
    best_fid = 100

    if resume is True:
        checkpointG = torch.load(pathG)
        checkpointD = torch.load(pathD)
        epoch = checkpointG['epoch']
        G.load_state_dict(checkpointG['model_state_dict'])
        optimizerG.load_state_dict(checkpointG['optimizer_state_dict'])
        D.load_state_dict(checkpointD['model_state_dict'])
        optimizerD.load_state_dict(checkpointD['optimizer_state_dict'])

    while epoch < n_epochs:
        train_pbar = tqdm(train_loader, position=0, leave=True)
        count = 1
        for image in train_pbar:
            ##########################
            # Train Discriminator #
            ##########################
            batch = len(image)
            image = image.to(device)
            noise = torch.randn(batch, config['noise_dimension'], 1, 1, device=device)
            fake = G(noise)

            # Feed real-batch
            real_pred = D(image)
            # feed fake-batch
            fake_pred = D(fake)
            # gradient penalty
            gradient_penalty = compute_gradient_penalty(D, image, fake, device)
            optimizerD.zero_grad()
            D_loss = -torch.mean(real_pred) + torch.mean(fake_pred) + config['gradient_penalty'] * gradient_penalty
            D_loss.backward()
            optimizerD.step()


            optimizerG.zero_grad()
            #####################
            # Train Genrator #
            #####################
            fake = G(noise)
            pred = D(fake)
            G_loss = -torch.mean(pred)
            G_loss.backward()
            optimizerG.step()

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'D_real': torch.mean(real_pred).item(), 'D_fake': torch.mean(fake_pred).item()})
            count += 1

        save_checkpoint(pathG, pathD, epoch, G, D, optimizerG, optimizerD)
        epoch += 1


        # generate 1000 images
        pred = G(fixed_noise).detach().cpu()
        for i in range(1000):
            norm_ip(pred[i])
            tmp = transforms.ToPILImage()(pred[i])
            tmp.save(os.path.join('./output', str(i) + '.png'))

        out = subprocess.Popen(['python', '-m', 'pytorch_fid', '--device', 'cuda:3', './output', './hw2_data/face/val'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        fid, _ = out.communicate()
        fid = fid.decode().split()[-1].replace('\n', '')
        if float(fid) < best_fid:
            best_fid = float(fid)
            save_checkpoint('./best_G.ckpt', './best_D.ckpt', epoch, G, D, optimizerG, optimizerD)
        print(best_fid, '/',float(fid))




# Config ######################################

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print(device)
config = {
    'seed': 234,
    'noise_dimension': 100,
    'gradient_penalty': 10,
    'critic_iter': 5,
    'n_epochs': 1000,
    'batch_size': 128,
    'learning_rate': 0.00001,
}

# Establishing dataset ######################################

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = FaceDataset('./hw2_data/face/train/', transform)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)


# GANs ######################################
G = Generator(config['noise_dimension'], Resblock).to(device)
D = Discriminator(Resblock).to(device)


# Start training 
train(G, D, train_loader, config, device, False, './cur_G.ckpt', './cur_D.ckpt')

