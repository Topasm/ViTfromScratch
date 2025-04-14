import torch
import torch.nn as nn
from modules import TransformerEncoder, ClassificationHeadMLP


class Vit(nn.module):

    """ViT-Base Layer 12  Hidden size 768 MLP 3072 Head 12 86M"""

    def __init__(self, input_dim, num_head, num_patch, num_class):
        self.cls = nn.Embedding(num_patch, num_class)

        self.pose_embedding = nn.Embedding(num_patch, latent_dim)

        self.TransformerEncoderLayer1 = TransformerEncoder(
            input_dim=input_dim, num_head=num_head)
        self.TransformerEncoderLayer2 = TransformerEncoder(
            input_dim=input_dim, num_head=num_head)
        self.TransformerEncoderLayer3 = TransformerEncoder(
            input_dim=input_dim, num_head=num_head)
        self.TransformerEncoderLayer4 = TransformerEncoder(
            input_dim=input_dim, num_head=num_head)
        self.TransformerEncoderLayer5 = TransformerEncoder(
            input_dim=input_dim, num_head=num_head)
        self.TransformerEncoderLayer6 = TransformerEncoder(
            input_dim=input_dim, num_head=num_head)
        self.TransformerEncoderLayer7 = TransformerEncoder(
            input_dim=input_dim, num_head=num_head)
        self.TransformerEncoderLayer8 = TransformerEncoder(
            input_dim=input_dim, num_head=num_head)
        self.TransformerEncoderLayer9 = TransformerEncoder(
            input_dim=input_dim, num_head=num_head)
        self.TransformerEncoderLayer10 = TransformerEncoder(
            input_dim=input_dim, num_head=num_head)
        self.TransformerEncoderLayer11 = TransformerEncoder(
            input_dim=input_dim, num_head=num_head)
        self.TransformerEncoderLayer12 = TransformerEncoder(
            input_dim=input_dim, num_head=num_head)
        self.MLP_Head = ClassificationHeadMLP(inputdim=input_dim, outdim=)

    def forward(self, x):

        x_cls = torch.concat([x, cls], dim=1)

        x_pose = x_cls + self.pose_embedding
