import torch
import torch.nn as nn
import torch.nn.functional as F
from .spiralconv import SpiralConv


def Pool(x, trans, dim=1):
    row = trans["row"]
    col = trans["col"]
    value = trans["data"].unsqueeze(-1)
    colIdx = torch.index_select(x, dim, col) * value
    
    temp = torch.zeros(x.size(0), trans['size'][0], x.size(2), dtype=x.dtype).to(x.device)           
    indices = torch.stack([row]*colIdx.size(0))
    indices = torch.stack([indices]*colIdx.size(2), dim=2)
    
    out = torch.scatter_add(temp, dim, indices, colIdx)

    return out

class SpiralEnblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralEnblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, down_transform):
        out = F.elu(self.conv(x))
        out = Pool(out, down_transform)
        return out




class SpiralDeblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralDeblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, up_transform):
        out = Pool(x, up_transform)
        out = F.elu(self.conv(out))
        return out

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, spiral_indices, down_transform):        
        super(Encoder, self).__init__()

        self.down_transform = down_transform
        self.num_vert = self.down_transform[-1]['size'][0]

        # encoder
        self.layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.layers.append(
                                SpiralEnblock(in_channels, out_channels[idx], spiral_indices[idx])
                                )
            else:
                self.layers.append(
                                SpiralEnblock(out_channels[idx - 1], out_channels[idx], spiral_indices[idx]))
        self.layers.append(
                nn.Linear(self.num_vert * out_channels[-1], latent_channels))


    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i != len(self.layers) - 1:
                x = layer(x, self.down_transform[i])
            else:
                x = x.view(-1, layer.weight.size(1))
                x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, spiral_indices, up_transform):
        super(Decoder, self).__init__()

        self.up_transform = up_transform
        self.num_vert = self.up_transform[-1]['size'][1]
        self.out_channels = out_channels

        # decoder
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Linear(latent_channels, self.num_vert * out_channels[-1]))
        for idx in range(len(out_channels)):
            if idx == 0:
                self.layers.append(
                    SpiralDeblock(out_channels[-idx - 1],out_channels[-idx - 1], spiral_indices[-idx - 1]))
            else:
                self.layers.append(
                    SpiralDeblock(out_channels[-idx], out_channels[-idx - 1],spiral_indices[-idx - 1]))
        self.layers.append(
            SpiralConv(out_channels[0], in_channels, spiral_indices[0]))
    
    def forward(self, x):
        num_layers = len(self.layers)
        num_features = num_layers - 2
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.num_vert, self.out_channels[-1])
            elif i != num_layers - 1:
                x = layer(x, self.up_transform[num_features - i])
            else:
                x = layer(x)
        return x

class AE(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, spiral_indices, down_transform, up_transform):
        super(AE, self).__init__()

        # Initialize encoder and decoder
        self.encoder = Encoder(in_channels, out_channels, latent_channels, spiral_indices, down_transform)
        self.decoder = Decoder(in_channels, out_channels, latent_channels, spiral_indices, up_transform)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


class DAE(AE):

    def forward(self, x):
        z = self.decoder(x)        
        out = self.encoder(z)
        return out
