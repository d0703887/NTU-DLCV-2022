from torch import Tensor
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms

# standard library
import os
import json
import numpy as np
from copy import deepcopy
import math

# others
from PIL import Image
from tokenizers import Tokenizer
from tqdm import tqdm
import timm


# dataset/ImageData
class ImageDataset(Dataset):
    def __init__(self, root, transform, mode):
        self.tokenizer = Tokenizer.from_file('./hw3_data/caption_tokenizer.json')
        self.tokenizer.enable_padding(length=54)
        self.root = root
        self.transform = transform
        self.filenames = []
        for file in os.listdir(self.root):
            self.filenames.append(file)
        self.len = len(self.filenames)

        # read caption
        if mode == 'train':
            f = open('./hw3_data/p2_data/train_caption.json')
        elif mode == 'val':
            f = open('./hw3_data/p2_data/val_caption.json')
        self.caption = json.load(f)
        f.close()

    def __getitem__(self, index):
        image = transforms.ToTensor()(Image.open(os.path.join(self.root, self.filenames[index])))
        captions = self.caption[self.filenames[index]]

        # Preprocess images with 1 channel
        if image.size(0) == 1:
            image = torch.cat((image, image, image), dim=0)

        if self.transform is not None:
            image = self.transform(image)

        # Random choose 1 caption and tokenize
        idx = np.random.randint(0, 5)
        caption = captions[idx]
        caption = self.tokenizer.encode(caption).ids
        caption = torch.tensor(caption)  # (number of captions, max caption length) = (5, 54)

        return image, caption

    def __len__(self):
        return self.len


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        param:
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        self.d_model = d_model

        # create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # not a parameter, but should be part of the modules state.
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.size(1)]
        return x


class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, feedforward_dim: int,
                 dropout: float):
        super(DecoderLayer, self).__init__()
        """
        param:
        d_model:    features size.
                    int
        num_heads:  number of heads in the multiheadattention model.
                    int
        dropout:    dropout value
                    float
        """

        self.dec_self_attn = nn.MultiheadAttention(d_model,
                                                num_heads,
                                                dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                 num_heads,
                                                 dropout=dropout)

        self.self_attn_norm = nn.LayerNorm(d_model)
        self.multihead_norm = nn.LayerNorm(d_model)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.multihead_dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(nn.Linear(d_model, feedforward_dim),
                                nn.ReLU(inplace=True), nn.Dropout(p=dropout),
                                nn.Linear(feedforward_dim, d_model))

        self.ff_norm = nn.LayerNorm(d_model)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, dec_inputs: Tensor, enc_outputs: Tensor,
                tgt_mask: Tensor,
                tgt_pad_mask: Tensor):
        """
        param:
        dec_inputs:     [max_len, batch_size, embed_dim]
        enc_outputs:    [encode_size^2=196, batch_size, embed_dim]
        """
        # self attention + resedual summation + norm
        output, _ = self.dec_self_attn(dec_inputs,
                                       dec_inputs,
                                       dec_inputs,
                                       attn_mask=tgt_mask,
                                       key_padding_mask=tgt_pad_mask)
        output = dec_inputs + self.self_attn_dropout(output)
        output = self.self_attn_norm(output)  # type: Tensor

        # # self attention + residual + norm + FF
        output2, attns = self.multihead_attn(output, enc_outputs, enc_outputs)
        output = output + self.multihead_dropout(output2)
        output = self.multihead_norm(output)

        output2 = self.ff(output)  # type: Tensor
        output = self.ff_norm(output + self.ff_dropout(output2))

        return output, attns


class Decoder(nn.Module):
    """
    param:
    layer:          an instance of the EecoderLayer() class
    vocab_size:     the number of vocabulary
                    int
    d_model:        size of features in the transformer inputs
                    int
    num_layers:     the number of decoder-layers
                    int
    max_len:        maximum len pf target captions
                    int
    dropout:        dropout value
                    float
    pad_id:         padding token id
                    float
    """

    def __init__(self,
                 layer: DecoderLayer,
                 vocab_size: int,
                 d_model: int,
                 num_layers: int,
                 max_len: int,
                 dropout: float,
                 pad_id: int):
        super().__init__()

        self.pad_id = pad_id

        # Embedding layer + pos encoding
        self.cptn_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = PositionalEncoding(d_model, max_len - 1)

        # Make copies of the decoder layer
        self.layers = nn.ModuleList(
            [deepcopy(layer) for _ in range(num_layers)])

        self.dropout = nn.Dropout(p=dropout)

    def get_attn_subsequent_mask(self, sz: int) -> Tensor:
        """
        Generates an upper-triangular matrix of -inf, with zeros on diag.
        """
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, tgt_cptn: Tensor,
                src_img: Tensor):
        """
        param:
        tgt_cptn:   Captions (Transformer target sequence)
                    Tensor
                    [batch_size, max_len-1]
        src_img:    Encoded images (Transformer source sequence)
                    Tensor
                    [encode_size^2, batch_size, image_embed_dim]
        outputs:
        output:     Decoder output
                    Tensor
                    [max_len, batch_size, model_embed_dim]
        attn_all:   Attension weights
                    Tensor
                    [layer_num, batch_size, head_num, max_len-1,
                    encode_size^2]
                    See comments in decoder_layers.DecoderLayer
        """

        # create masks, then pass to decoder
        tgt_pad_mask = (tgt_cptn == self.pad_id)
        tgt_mask = self.get_attn_subsequent_mask(tgt_cptn.size()[1])
        tgt_mask = tgt_mask.to(tgt_cptn.device)

        # encode captions + pos enc
        # (B, max_len) -> (B, max_len, d_model) -> (max_len, B, d_model)
        tgt_cptn = self.cptn_emb(tgt_cptn)  # type: Tensor
        tgt_cptn = self.dropout(self.pos_emb(tgt_cptn.permute(1, 0, 2)))

        attns_all = []
        for layer in self.layers:
            tgt_cptn, attns = layer(tgt_cptn, src_img, tgt_mask, tgt_pad_mask)
            attns_all.append(attns)
        # [layer_num, batch_size, head_num, max_len, encode_size**2]
        attns_all = torch.stack(attns_all)

        return tgt_cptn, attns_all

class Transformer(nn.Module):
    def __init__(self, embed_size, vocab_size, decoder_n, head, max_len, dropout, pad_id):
        super(Transformer, self).__init__()
        self.encoder = timm.create_model('vit_large_patch16_224_in21k', pretrained=True)
        self.linear = nn.Linear(1024, 384)
        decoder_layer = DecoderLayer(embed_size, head, 2048, dropout)
        self.decoder = Decoder(decoder_layer, vocab_size, embed_size, decoder_n, max_len, dropout, pad_id)
        self.ff = nn.Linear(embed_size, vocab_size)

    def forward(self, image, caption):
        image = self.encoder.forward_features(image)
        image = self.linear(image).permute(1, 0, 2)
        output, attn = self.decoder(caption, image)
        output = self.ff(output)
        return output


# checkpoint
def save_checkpoint(path, epoch: int, module, optimizer):
    torch.save({
        'epoch': epoch,
        'model_state_dict': module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


# config
class Config(object):
    def __init__(self):
        # Learning Rates
        self.lr = 1e-4

        # Epochs
        self.epochs = 100
        self.lr_drop = 20
        self.start_epoch = 0
        self.weight_decay = 1e-4

        self.batch_size = 32



# main
def train(model, train_loader, val_loader, config, device, path, resume, model_only):
     # Start training
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop)
    criterion = nn.CrossEntropyLoss()
    tokenizer = Tokenizer.from_file('./hw3_data/caption_tokenizer.json')
    epoch = 0

    if model_only is True:
        model.load_state_dict(torch.load(path)['model_state_dict'])

    if resume is True:
        checkpoint = torch.load(path)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    n_epochs, best_loss, train_loss, val_loss, train_total, val_total = config.epochs, 100, 0, 0, len(train_loader), len(val_loader)

    while epoch < n_epochs:
        train_loss = 0
        val_loss = 0
        model.train()
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for images, captions in train_pbar:
            images, captions = images.to(device), captions.to(device)
            output = model(images, captions[:, :-1])

            loss = criterion(output.permute(1, 2, 0), captions[:, 1:])
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        lr_scheduler.step()

        # Evaluation
        with torch.no_grad():
            for images, captions in val_loader:
                images, captions = images.to(device), captions.to(device)
                output = model(images, captions[:, :-1])
                loss = criterion(output.permute(1, 2, 0), captions[:, 1:])
                val_loss += loss.item()

        print(f'Epoch: {epoch}, train_Loss: {train_loss / train_total}, val_loss: {val_loss/val_total}\n')
        save_checkpoint(path, epoch + 1, model, optimizer)
        epoch += 1


# Model
device = 'cuda:2'
model = Transformer(384, 18022, 6, 8, 54, 0.1, 0).to(device)
for param in model.encoder.parameters():
    param.requires_grad = False
config = Config()

# Dataset
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
train_dataset = ImageDataset('./hw3_data/p2_data/images/train', train_transform, mode='train')
train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True)
val_dataset = ImageDataset('./hw3_data/p2_data/images/val', val_transform, mode='val')
val_loader = DataLoader(val_dataset, config.batch_size, shuffle=True)

train(model, train_loader, val_loader, config, device, './vit_L.ckpt', False, False)



