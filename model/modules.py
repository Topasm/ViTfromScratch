import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderMLP(nn.Module):
    """mlp in vit


    mlp size is 3072 in 5p table 1"""

    def __init__(self, inputdim, hiddendim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(inputdim, hiddendim),
                                 nn.ReLU())

    def forward(self, x):
        x = self.mlp(x)


class ClassificationHeadMLP(nn.Module):
    """MLP Head for Classification
    4p 3.1 section 
    The classification head is implemented by a MLP with one hidden layer at pre-training
    time and by a single linear layer at fine-tuning time"""

    def __init__(self, inputdim, outdim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(inputdim, outdim),
                                 nn.ReLU())

    def forward(self, x):
        x = self.mlp(x)


class SA(nn.Module):
    "sa example"

    def __init__(self, input):
        super().__init__()
        self.U_qkv = nn.Linear(input, 3*input)

    def forward(self, x):

        B, N, D = x.shape

        qkv = self.U_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        Attention = F.softmax(q @ k.transpose(-2, -1) / (D**0.5))

        SAout = Attention@v

        return SAout


class MultiheadAttention(nn.Module):
    """ multihead Attention"""

    def __init__(self, input, num_head):
        super().__init__()

        self.num_head = num_head

        self.q_proj = nn.Linear(input, ouput)
        self.k_proj = nn.Linear(input, ouput)
        self.v_proj = nn.Linear(input, ouput)

    def forward(self, x):

        B, N, D = x.shape

        qkv = self.U_msa(x)
        q, k, v = qkv.chunk(self.num_head, dim=-1)

        Attention = F.softmax(q @ k.transpose(-2, -1) / (D**0.5))

        SAout = Attention@v

        return SAout


class TransformerEncoder(nn.Module):
    """ Transformer Encoder

     but replace the Batch Normalization lay-
    ers (Ioffe & Szegedy, 2015) with Group Normalization """

    def __init__(self, input):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=3)
