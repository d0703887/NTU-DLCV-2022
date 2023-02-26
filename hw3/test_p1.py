import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import clip
import pandas as pd
import argparse


# Argument
parser = argparse.ArgumentParser()
parser.add_argument("-t", type=str)
parser.add_argument("-j", type=str)
parser.add_argument("-o", type=str)
args = parser.parse_args()
test_path = args.t
json_path = args.j
output_path = args.o
print('test images path: ', test_path)
print('output file path: ', output_path)


# Dataset
class ImageDataset (Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.filenames = []
        for file in os.listdir(self.root):
            self.filenames.append(file)
        self.len = len(self.filenames)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.filenames[index]))
        if self.transform is not None:
            img = self.transform(img)

        return img, self.filenames[index]

    def __len__(self):
        return self.len


# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load('ViT-L/14')
model.to(device)

# Prepare inputs
test_dataset = ImageDataset(test_path, preprocess)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
labels = pd.read_json(json_path, typ='dictionary').values.tolist()
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in labels]).to(device)


# Start inference
L = []

for image, filename in test_loader:
    image = image.to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    for i in range(len(similarity)):
        values, indices = similarity[i].topk(5)
        pred_label = indices[0]
        L.append([filename[i], str(pred_label.item())])

write = pd.DataFrame(L, columns=('filename', 'label'))
write.to_csv(output_path, index=False)




