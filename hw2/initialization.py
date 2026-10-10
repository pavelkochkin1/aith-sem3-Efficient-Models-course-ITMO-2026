"""ZerO Algorithm 1 (arXiv:2110.12661v3) and explicit control methods."""

import math

import torch
from torch import nn


def hadamard(order, *, dtype=torch.float32, device=None):
    """Sylvester matrix in the same row/column order as scipy.linalg.hadamard."""
    if order < 1 or order & (order - 1):
        raise ValueError("Hadamard order must be a positive power of two")
    matrix = torch.ones((1, 1), dtype=dtype, device=device)
    while matrix.shape[0] < order:
        matrix = torch.cat(
            (torch.cat((matrix, matrix), dim=1), torch.cat((matrix, -matrix), dim=1)),
            dim=0,
        )
    return matrix


@torch.no_grad()
def zero_init_(weight, normalization="paper"):
    """Initialize a P×Q matrix: eye if P<=Q, truncated Hadamard otherwise.

    paper: Algorithm 1, c=2**(-(m-1)/2)=sqrt(2/M).
    author: example_mnist.ipynb, c=2**(-m/2)=1/sqrt(M).
    M is the smallest power of two >= P. These scales differ by sqrt(2).
    """
    if weight.ndim != 2:
        raise ValueError("Algorithm 1 expects a matrix")
    if normalization not in {"paper", "author"}:
        raise ValueError("normalization must be 'paper' or 'author'")
    rows, columns = weight.shape
    if rows <= columns:
        return nn.init.eye_(weight)
    order = 1 << (rows - 1).bit_length()
    scale = math.sqrt((2 if normalization == "paper" else 1) / order)
    weight.copy_(hadamard(order, dtype=weight.dtype, device=weight.device)[:rows, :columns])
    weight.mul_(scale)
    return weight


@torch.no_grad()
def initialize_linear_(layer, method):
    if method == "zero_all":
        nn.init.zeros_(layer.weight)
    elif method == "identity":
        nn.init.eye_(layer.weight)
    elif method in {"zero_paper", "zero_author"}:
        zero_init_(layer.weight, normalization=method.removeprefix("zero_"))
    elif method == "kaiming":
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    elif method == "xavier":
        nn.init.xavier_normal_(layer.weight)
    else:
        raise ValueError(f"Unknown initialization: {method}")
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
