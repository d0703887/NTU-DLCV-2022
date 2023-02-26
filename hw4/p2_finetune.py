import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from tqdm import tqdm
import csv


# Dataset
class ImageDataset(Dataset):
    def __init__(self, root, transform, mode):
        self.root = root
        self.transform = transform
        self.classes = {'Couch': 0, 'Helmet': 1, 'Refrigerator': 2, 'Alarm_Clock': 3, 'Bike': 4, 'Bottle': 5, 'Calculator': 6, 'Chair': 7, 'Mouse': 8, 'Monitor': 9, 'Table': 10, 'Pen': 11, 'Pencil': 12, 'Flowers': 13, 'Shelf': 14, 'Laptop': 15, 'Speaker': 16, 'Sneakers': 17, 'Printer': 18, 'Calendar': 19, 'Bed': 20, 'Knives': 21, 'Backpack': 22, 'Paper_Clip': 23, 'Candles': 24, 'Soda': 25, 'Clipboards': 26, 'Fork': 27, 'Exit_Sign': 28, 'Lamp_Shade': 29, 'Trash_Can': 30, 'Computer': 31, 'Scissors': 32, 'Webcam': 33, 'Sink': 34, 'Postit_Notes': 35, 'Glasses': 36, 'File_Cabinet': 37, 'Radio': 38, 'Bucket': 39, 'Drill': 40, 'Desk_Lamp': 41, 'Toys': 42, 'Keyboard': 43, 'Notebook': 44, 'Ruler': 45, 'ToothBrush': 46, 'Mop': 47, 'Flipflops': 48, 'Oven': 49, 'TV': 50, 'Eraser': 51, 'Telephone': 52, 'Kettle': 53, 'Curtains': 54, 'Mug': 55, 'Fan': 56, 'Push_Pin': 57, 'Batteries': 58, 'Pan': 59, 'Marker': 60, 'Spoon': 61, 'Screwdriver': 62, 'Hammer': 63, 'Folder': 64}
        if mode == 'train':
            tmp = csv.reader(open('./hw4_data/office/train.csv', 'r'))
        elif mode == 'val':
            tmp = csv.reader(open('./hw4_data/office/val.csv', 'r'))
        self.csv = {}
        for row in tmp:
            self.csv[row[1]] = row[2]

        # Fetch filenames
        self.filenames = []
        for image in os.listdir(root):
            self.filenames.append(image)

        self.len = len(self.filenames)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, self.filenames[index]))

        label = torch.zeros(65, dtype=torch.float)
        label[self.classes[self.csv[self.filenames[index]]]] = 1
        if transforms is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.len


class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        self.model = torchvision.models.resnet50()
        self.model.load_state_dict(torch.load('./p2_210.ckpt'))
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 65)
        )


    def forward(self, x):
        x = self.model(x)
        return x


# Save Checkpoint
def checkpoint(model, optimizer, scheduler, epoch, path):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch
    }, path)


# Train
def train(model, train_loader, val_loader, config, device, resume, path):
    n_epochs, epoch, best_accuracy = config['n_epochs'], 0, 0
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=0)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, 15, 0.1)
    train_length = len(train_loader)
    val_length = len(val_loader)

    if resume:
        ck = torch.load(path)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        epoch = ck['epoch']

    while epoch < n_epochs:
        train_loss = 0
        val_loss = 0
        train_pbar = tqdm(train_loader, position=0, leave=True)
        model.train()
        for images, labels in train_pbar:
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)

            pred = model(images)
            loss = criterion(pred, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        scheduler.step()

        # validation
        total_count = 0
        correct_count = 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                pred = model(images)
                loss = criterion(pred, labels)
                val_loss += loss.item()

                for i in range(len(pred)):
                    total_count += 1
                    if torch.argmax(pred[i]) == torch.argmax(labels[i]):
                        correct_count += 1

        train_loss = train_loss / train_length
        val_loss = val_loss / val_length
        accuracy = correct_count / total_count
        print(total_count)
        print(correct_count)
        print(f"Epoch: {epoch}, train_loss: {train_loss:.4}, val_loss: {val_loss:.4}, Accuracy: {accuracy:.4}")
        epoch += 1
        checkpoint(model, optimizer, scheduler, epoch, path)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), './office_best.ckpt')


# Config
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
print("device: ", device)
config = {
    'n_epochs': 1000,
    'batch_size': 16,
    'learning_rate': 0.001,
}


# Start Training
model = Resnet50().to(device)
for param in model.model.parameters():
    param.requires_grad = False
for param in model.model.fc.parameters():
    param.requires_grad = True

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=torch.tensor([0.485, 0.456, 0.406]),
        std=torch.tensor([0.229, 0.224, 0.225])
    ),
    # transforms.Resize((128, 128))
    transforms.RandomResizedCrop((128, 128), scale=(0.6, 1)),
    transforms.RandomRotation(degrees=30),
    transforms.RandomErasing()
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=torch.tensor([0.485, 0.456, 0.406]),
        std=torch.tensor([0.229, 0.224, 0.225])
    ),
    transforms.Resize((128, 128))
])
train_dataset = ImageDataset('./hw4_data/office/train', train_transform, 'train')
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_dataset = ImageDataset('./hw4_data/office/val', val_transform, 'val')
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

#train(model, train_loader, val_loader, config, device, False, 'p2_fine.ckpt')

model.load_state_dict(torch.load('./office_best.ckpt'))
model.eval()
total = 0
correct = 0
with torch.no_grad():
    for image, label in val_loader:
        image = image.to(device)
        pred = model(image)
        for j in range(len(pred)):
            total += 1
            if torch.argmax(pred[j], dim=0) == torch.argmax(label[j], dim=0):
                correct += 1
print(correct / total)




