import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

"https://data-analysis-science.tistory.com/50 einops example"


class EncoderMLP(nn.Module):
    """mlp in vit

    The MLP contains two layers with a GELU non-linearity

    mlp size is 3072 in 5p table 1"""

    def __init__(self, inputdim, hiddendim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(inputdim, hiddendim),
                                 nn.GELU(),
                                 nn.Linear(hiddendim, hiddendim),
                                 nn.GELU)

    def forward(self, x):
        x = self.mlp(x)

        return x


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

    def __init__(self, embedded_dim, num_head):
        super().__init__()

        self.num_head = num_head

        self.q_proj = nn.Linear(embedded_dim, embedded_dim)
        self.k_proj = nn.Linear(embedded_dim, embedded_dim)
        self.v_proj = nn.Linear(embedded_dim, embedded_dim)

        self.output_proj = nn.Linear()

    def forward(self, x):

        B, N, D = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        Attention = F.softmax(q @ k.transpose(-2, -1) / (D**0.5))

        MHAout = Attention@v

        MHAout = self.output_proj(MHAout)

        return MHAout


class TransformerEncoder(nn.Module):
    """ Transformer Encoder

    Transformer encoder (Vaswani et al., 2017) consists of alternating layers of multiheaded self-
attention (MSA, see Appendix A) and MLP blocks (Eq. 2, 3). Layernorm (LN) is applied before
every block, and residual connections after every block """

    "embedded patchs 16x16 B, N, D, C"

    def __init__(self, input_dim, num_head):
        super().__init__()
        self.norm1 = nn.LayerNorm(num_groups=32, num_channels=3)
        self.mha = MultiheadAttention(input_dim, num_head=num_head)
        self.norm2 = nn.LayerNorm(num_groups=32, num_channels=3)
        self.mlp = EncoderMLP(input_dim, input_dim)

    def forward(self, x):
        x_norm = self.norm1(x)
        x_attn = self.mha(x_norm)
        x_update = x + x_attn
        x_update_norm = self.norm2(x_update)
        x_out = self.mlp(x_update_norm)
        x_out = x_out + x_update

        return x_out


class PatchEmbedding(nn.Module):

    """To handle 2D images, we reshape the image x ∈ RH×W ×C into a
sequence of flattened 2D patches xp ∈ RN×(P 2·C), where (H,W) is the resolution of the original
image, C is the number of channels, (P,P) is the resolution of each image patch, and N = HW/P2
is the resulting number of patches, which also serves as the effective input sequence length for the
Transformer. The Transformer uses constant latent vector size D through all of its layers, so we
flatten the patches and map to D dimensions with a trainable linear projection (Eq. 1). We refer to
the output of this projection as the patch embeddings"""

    def __init__(self, input_dim, num_head, patch_size):
        super().__init__()
        self.patchfy = nn.Conv2d(input_dim, outm,)

    def forward(self, x):

        B, C, H, W = x.shape

        patch = self.patchfy(x)  # n p^2xc

        x = rearrange(x, 'b c h w -> b ')

        return out
