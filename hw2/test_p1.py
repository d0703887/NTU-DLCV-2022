import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import argparse


# Argument
parser = argparse.ArgumentParser()
parser.add_argument("-o", type=str)
args = parser.parse_args()
output_path = args.o
print('output file path: ', output_path)


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


# Model
def norm_ip(img):
    low = float(img.min())
    high = float(img.max())
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device: ", device)
torch.manual_seed('34073407')
G = Generator(100).to(device)
G.load_state_dict(torch.load('./p1_scratch_model.ckpt', map_location=torch.device(device))['model_state_dict'])
fixed_noise = torch.randn(1000, 100, 1, 1, device=device)
pred = G(fixed_noise).detach().cpu()
for i in range(1000):
    norm_ip(pred[i])
    tmp = transforms.ToPILImage()(pred[i])
    tmp.save(os.path.join(output_path, str(i)+'.png'))
