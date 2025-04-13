import torch
import torch.nn as nn


class EncoderMLP(nn.Module):
    """mlp in vit
    4p 3.1 section 
    The classification head is implemented by a MLP with one hidden layer at pre-training
    time and by a single linear layer at fine-tuning time

    mlp size is 3072 in 5p table 1"""

    def __init__(self, inputdim, hiddendim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(inputdim, hiddendim))


class ClassificationMLP(nn.Module):
    """MLP Head for Classification"""
