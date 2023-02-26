import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import argparse
import csv


# Argument
parser = argparse.ArgumentParser()
parser.add_argument("-t", type=str)
parser.add_argument("-i", type=str)
parser.add_argument("-p", type=str)
args = parser.parse_args()
test_csv = args.t
image_path = args.i
pred_path = args.p


# Dataset
class ImageDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.classes = {'Couch': 0, 'Helmet': 1, 'Refrigerator': 2, 'Alarm_Clock': 3, 'Bike': 4, 'Bottle': 5, 'Calculator': 6, 'Chair': 7, 'Mouse': 8, 'Monitor': 9, 'Table': 10, 'Pen': 11, 'Pencil': 12, 'Flowers': 13, 'Shelf': 14, 'Laptop': 15, 'Speaker': 16, 'Sneakers': 17, 'Printer': 18, 'Calendar': 19, 'Bed': 20, 'Knives': 21, 'Backpack': 22, 'Paper_Clip': 23, 'Candles': 24, 'Soda': 25, 'Clipboards': 26, 'Fork': 27, 'Exit_Sign': 28, 'Lamp_Shade': 29, 'Trash_Can': 30, 'Computer': 31, 'Scissors': 32, 'Webcam': 33, 'Sink': 34, 'Postit_Notes': 35, 'Glasses': 36, 'File_Cabinet': 37, 'Radio': 38, 'Bucket': 39, 'Drill': 40, 'Desk_Lamp': 41, 'Toys': 42, 'Keyboard': 43, 'Notebook': 44, 'Ruler': 45, 'ToothBrush': 46, 'Mop': 47, 'Flipflops': 48, 'Oven': 49, 'TV': 50, 'Eraser': 51, 'Telephone': 52, 'Kettle': 53, 'Curtains': 54, 'Mug': 55, 'Fan': 56, 'Push_Pin': 57, 'Batteries': 58, 'Pan': 59, 'Marker': 60, 'Spoon': 61, 'Screwdriver': 62, 'Hammer': 63, 'Folder': 64}

        # Fetch filenames
        self.filenames = []
        for image in os.listdir(root):
            self.filenames.append(image)

        self.len = len(self.filenames)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, self.filenames[index]))

        if transforms is not None:
            image = self.transform(image)

        return image, self.filenames[index]

    def __len__(self):
        return self.len


class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        self.model = torchvision.models.resnet50()
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 65)
        )


    def forward(self, x):
        x = self.model(x)
        return x


# Config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device: ", device)

# Start Training
model = Resnet50().to(device)
model.load_state_dict(torch.load('./setting_c.ckpt', map_location=torch.device(device)))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=torch.tensor([0.485, 0.456, 0.406]),
        std=torch.tensor([0.229, 0.224, 0.225])
    ),
    transforms.Resize((128, 128))
])
dataset = ImageDataset(image_path, transform)
loader = DataLoader(dataset, batch_size=16, shuffle=False)
classes = {0: 'Couch', 1: 'Helmet', 2: 'Refrigerator', 3: 'Alarm_Clock', 4: 'Bike', 5: 'Bottle', 6: 'Calculator', 7: 'Chair', 8: 'Mouse', 9: 'Monitor', 10: 'Table', 11: 'Pen', 12: 'Pencil', 13: 'Flowers', 14: 'Shelf', 15: 'Laptop', 16: 'Speaker', 17: 'Sneakers', 18: 'Printer', 19: 'Calendar', 20: 'Bed', 21: 'Knives', 22: 'Backpack', 23: 'Paper_Clip', 24: 'Candles', 25: 'Soda', 26: 'Clipboards', 27: 'Fork', 28: 'Exit_Sign', 29: 'Lamp_Shade', 30: 'Trash_Can', 31: 'Computer', 32: 'Scissors', 33: 'Webcam', 34: 'Sink', 35: 'Postit_Notes', 36: 'Glasses', 37: 'File_Cabinet', 38: 'Radio', 39: 'Bucket', 40: 'Drill', 41: 'Desk_Lamp', 42: 'Toys', 43: 'Keyboard', 44: 'Notebook', 45: 'Ruler', 46: 'ToothBrush', 47: 'Mop', 48: 'Flipflops', 49: 'Oven', 50: 'TV', 51: 'Eraser', 52: 'Telephone', 53: 'Kettle', 54: 'Curtains', 55: 'Mug', 56: 'Fan', 57: 'Push_Pin', 58: 'Batteries', 59: 'Pan', 60: 'Marker', 61: 'Spoon', 62: 'Screwdriver', 63: 'Hammer', 64: 'Folder'}
predict = {}
with torch.no_grad():
    for image, filename in loader:
        image = image.to(device)
        pred = model(image)
        for i in range(len(pred)):
            id = torch.argmax(pred[i], dim=0)
            ans = classes[id.item()]
            predict[filename[i]] = ans


test = csv.reader(open(test_csv, 'r'))
ans = []
for row in test:
    if row[0] != 'id':
        row[2] = predict[row[1]]
    ans.append(row)

with open(pred_path, 'w') as fp:
    writer = csv.writer(fp)
    writer.writerows(ans)






