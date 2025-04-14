import torch
import torch.nn as nn
from .modules import TransformerEncoder, ClassificationHeadMLP


class Vit(nn.Module):

    """ViT-Base Layer 12  Hidden size 768 MLP 3072 Head 12 86M"""

    def __init__(self, input_dim=768, num_head=12, num_patch=256, num_class=10):
        super().__init__()
        self.cls = nn.Parameter(num_patch, num_class)

        self.pose_embedding = nn.Embedding(num_patch+1, latent_dim)

        self.Transformer = nn.Sequential(*[TransformerEncoder(
            input_dim=input_dim, num_head=num_head)
            for _ in range(12)])
        self.MLP_Head = ClassificationHeadMLP(
            inputdim=input_dim, outdim=num_class)

    def forward(self, x):

        x_cls = torch.concat([x, self.cls], dim=1)

        x_pose = x_cls + self.pose_embedding

        x_t_out = self.Transformer(x_pose)

        cls_ouput = x[:-1]
        out = self.MLP_Head(cls_ouput)
        return out
