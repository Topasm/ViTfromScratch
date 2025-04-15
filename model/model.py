import torch
import torch.nn as nn
from .modules import TransformerEncoder, ClassificationHeadMLP, PatchEmbedding


class Vit(nn.Module):

    """ViT-Base Layer 12  Hidden size 768 MLP 3072 Head 12 86M"""

    def __init__(self, input_dim=768, num_head=12, num_patch=196, num_class=10, patch_size=16):
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            3, embedded_dim=input_dim, patch_size=patch_size)
        # https://velog.io/@dusruddl2/torch.rand-torch.randn
        self.cls = nn.Parameter(torch.randn(1, 1, input_dim))

        self.pose_embedding = nn.Embedding(num_patch+1, input_dim)

        self.Transformer = nn.Sequential(*[TransformerEncoder(
            input_dim=input_dim, num_head=num_head)
            for _ in range(12)])
        self.MLP_Head = ClassificationHeadMLP(
            inputdim=input_dim, outdim=num_class)

    def forward(self, x):

        # torch.Size([1024, 3, 224, 224]) -Based on figure it should 224x224
        B, _, _, _ = x.shape

        x = self.patch_embedding(x)

        # torch.Size([1024, 196, 768])(patch 14x14=196)

        cls_token = self.cls.expand(B, -1, -1)

        # torch.Size([1024, 197, 768])

        x_cls = torch.concat([x, cls_token], dim=1)

        x_pose = x_cls + self.pose_embedding()

        x_t_out = self.Transformer(x_pose)

        cls_ouput = x[:-1]
        out = self.MLP_Head(cls_ouput)
        return out
