# torch
from torch import Tensor
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms

# standard library
import os
import json
from copy import deepcopy
import math
import numpy as np
import matplotlib.pyplot as plt

# others
from PIL import Image
from tokenizers import Tokenizer
import timm
import cv2

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
        return output, attn


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

tmp = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.ToPILImage()
])
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
tokenizer = Tokenizer.from_file('./hw3_data/caption_tokenizer.json')
model = Transformer(384, 18022, 6, 8, 54, 0.1, 0).to(device)
model = model.to(device)
model.load_state_dict(torch.load("./vit_L_0_635.ckpt")['model_state_dict'])
model.eval()

for image in os.listdir('./hw3_data/p3_data/images'):
    with torch.no_grad():
        image = Image.open(os.path.join('./hw3_data/p3_data/images', image))
        x = transform(image).to(device)
        images_features = model.linear(model.encoder.forward_features(x.unsqueeze(0))).permute(1, 0, 2)
        start = torch.zeros((1, 54), dtype=torch.long, device=device)
        start[0, 0] = 2

        image = tmp(image)
        fig = plt.figure(8)

        fig.add_subplot(2, 7, 1)
        plt.imshow(image)
        plt.axis('off')
        for i in range(53):
            output, att_mat = model.decoder(start[:, :-1], images_features)

            output = model.ff(output)
            output = output.permute(1, 0, 2)
            output = output[0, i, :]
            next = torch.argmax(output, dim=-1)
            start[:, i + 1] = next

            # visualize
            att_mat = att_mat.squeeze(1).cpu()  # [num_layers, 53, 197]
            # att_mat = att_mat.mean(dim=0)
            heatmap = att_mat[-1][i][1:].reshape((14, 14))
            heatmap = heatmap - heatmap.min() / (heatmap.max() - heatmap.min())
            heatmap = heatmap.unsqueeze(0).unsqueeze(0).detach()
            heatmap = nn.functional.interpolate(heatmap, size=(224, 224), mode='bilinear')
            heatmap = heatmap.squeeze()

            fig.add_subplot(2, 7, i + 2)
            plt.imshow(image)
            plt.imshow(heatmap, alpha=0.6, cmap='rainbow')
            plt.title(tokenizer.decode([next]))
            plt.axis('off')
            ############
            if next == 3:
                break
        plt.show()

