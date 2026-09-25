"""Последовательная CNN из условия ДЗ №1, без BatchNorm."""

from torch import nn


def build_model():
    """Вход (B, 3, S, S), выход (B, 100); CUDA и eval() задаёт вызывающий код.

    BatchNorm не добавляем по рабочему допущению из README.
    У Linear есть bias, как в наших аналитических расчётах.
    """
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(start_dim=1),
        nn.Linear(512, 256, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(256, 100, bias=True),
    )
