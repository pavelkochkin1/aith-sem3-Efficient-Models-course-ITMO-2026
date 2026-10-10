"""Bias-free appendix MLP and an explicitly separate MNIST patch-transformer study."""

import math

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .initialization import initialize_linear_
except ImportError:
    from initialization import initialize_linear_


class MLP(nn.Module):
    def __init__(self, input_dim=784, width=2048, method="zero_paper"):
        super().__init__()
        self.first = nn.Linear(input_dim, width, bias=False)
        self.hidden = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, 10, bias=False)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                initialize_linear_(layer, method)

    def features(self, images):
        x = images.flatten(1)
        return F.relu(self.hidden(F.relu(self.first(x))))

    def forward(self, images):
        return self.output(self.features(images))


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        if width % heads:
            raise ValueError("Transformer width must be divisible by head count")
        self.heads = heads
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def forward(self, x, return_attention=False):
        batch, length, width = x.shape
        def split(layer):
            return layer(x).reshape(batch, length, self.heads, width // self.heads).transpose(1, 2)
        q, k, v = split(self.q), split(self.k), split(self.v)
        attention = (q @ k.transpose(-1, -2) / math.sqrt(width // self.heads)).softmax(dim=-1)
        output = (attention @ v).transpose(1, 2).reshape(batch, length, width)
        output = self.out(output)
        return (output, attention) if return_attention else output


class TransformerBlock(nn.Module):
    def __init__(self, width, heads, ff_width):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = Attention(width, heads)
        self.norm2 = nn.LayerNorm(width)
        self.ff1 = nn.Linear(width, ff_width, bias=False)
        self.ff2 = nn.Linear(ff_width, width, bias=False)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.ff2(F.relu(self.ff1(self.norm2(x))))


def sinusoidal_positions(length, width):
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000) / width))
    result = torch.zeros(length, width)
    result[:, 0::2] = torch.sin(position * frequency)
    result[:, 1::2] = torch.cos(position * frequency[:width // 2])
    return result


class PatchTransformer(nn.Module):
    def __init__(self, image_size=28, patch_size=7, width=64, heads=4, depth=2,
                 ff_width=256, method="standard"):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("Image size must be divisible by patch size")
        if method not in {"standard", "identity", "zero_inspired", "zero_inspired_author"}:
            raise ValueError("Transformer method must be standard, identity, or zero_inspired[_author]")
        self.patch_size = patch_size
        self.patch = nn.Linear(patch_size * patch_size, width, bias=False)
        self.register_buffer("position", sinusoidal_positions((image_size // patch_size) ** 2, width))
        self.blocks = nn.ModuleList(TransformerBlock(width, heads, ff_width) for _ in range(depth))
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 10, bias=False)
        linear_method = {
            "standard": "xavier", "identity": "identity",
            "zero_inspired": "zero_paper", "zero_inspired_author": "zero_author",
        }[method]
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                initialize_linear_(layer, linear_method)
        if method.startswith("zero_inspired"):
            for block in self.blocks:
                nn.init.zeros_(block.attention.k.weight)
                nn.init.zeros_(block.attention.v.weight)

    def tokens(self, images):
        patches = F.unfold(images, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        return self.patch(patches) + self.position.unsqueeze(0)

    def features(self, images):
        x = self.tokens(images)
        for block in self.blocks:
            x = block(x)
        return self.norm(x).mean(dim=1)

    def forward(self, images):
        return self.output(self.features(images))

    @torch.no_grad()
    def attention_diagnostics(self, images):
        x = self.tokens(images)
        rows = []
        for index, block in enumerate(self.blocks):
            _, attention = block.attention(block.norm1(x), return_attention=True)
            entropy = -(attention.clamp_min(1e-12).log() * attention).sum(dim=-1).mean()
            head_difference = (attention - attention[:, :1]).abs().mean()
            rows.append({"block": index, "attention_entropy": entropy.item(),
                         "head_attention_difference": head_difference.item()})
            x = block(x)
        return rows
