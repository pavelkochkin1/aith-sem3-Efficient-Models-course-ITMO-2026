"""Перенеси сюда собственные рукописные выводы.

Все функции должны работать с числами и NumPy-массивами с broadcasting.
Здесь не должно быть запуска сети или измерений на GPU.
"""

import numpy as np


def flops(image_size, batch):
    """Число FLOPs одного forward 
    1 MAC = 2 FLOPs"""
    return batch * (17751 * image_size**2 + 313956)


def memory(image_size, batch):
    """оценка в байтах для S >= 32, кратного 16, и B >= 1

    FP32, без BatchNorm, Linear с bias, исходный вход сохраняется.
    ReLU inplace; старые активации освобождаются; flatten без копирования.
    Временные индексы, рабочие буферы и округления аллокатора не учитываются.
    Вывод и полный список допущений — в derivations.md.
    """
    return 4_161_296 + 52.0 * batch * image_size * image_size


def bytes_moved(image_size, batch):
    """Базовый объём чтений и записей за forward в байтах.

    Вход и параметры слоя читаются один раз, выход записывается один раз.
    Кэширование между слоями и внутренние временные буферы не учитываются.
    """
    return 364.0 * batch * image_size * image_size + 8592.0 * batch + 4_161_296


def latency(image_size, batch, theta):
    """T = t0 + sum(max(F_i / C, Q_i / beta)), результат в секундах.

    theta — словарь: t0 >= 0 (с), C > 0 (FLOP/с), beta > 0 (байт/с).
    Вычисления и передачи внутри слоя перекрываются; слои последовательны.
    C и beta считаются постоянными и общими для всех слоёв.
    """
    image_size = np.asarray(image_size, dtype=float)
    batch = np.asarray(batch, dtype=float)
    bs2 = batch * image_size**2

    # (F_i, Q_i) по нашей таблице: FLOPs и байты за один forward.
    layer_costs = (
        (2352 * bs2, 44 * bs2 + 18_816),          # Conv7x7, 3->32
        (8 * bs2, 64 * bs2),                     # ReLU
        (16 * bs2, 40 * bs2),                    # MaxPool
        (6400 * bs2, 24 * bs2 + 204_800),         # Conv5x5, 32->64
        (4 * bs2, 32 * bs2),                     # ReLU
        (2304 * bs2, 24 * bs2 + 294_912),         # Conv3x3, 64->128
        (2 * bs2, 16 * bs2),                     # ReLU
        (1024 * bs2, 24 * bs2 + 131_072),         # Conv1x1, 128->256
        (4 * bs2, 32 * bs2),                     # ReLU
        (4608 * bs2, 20 * bs2 + 2_359_296),       # Conv3x3, 256->256
        (bs2, 8 * bs2),                         # ReLU
        (1024 * bs2, 12 * bs2 + 524_288),         # Conv1x1, 256->512
        (2 * bs2, 16 * bs2),                     # ReLU
        (2 * bs2, 8 * bs2 + 2048 * batch),       # GlobalAvgPool
        (0, 0),                                 # Flatten без копирования
        (262400 * batch, 3072 * batch + 525_312), # Linear512->256
        (256 * batch, 2048 * batch),             # ReLU
        (51300 * batch, 1424 * batch + 102_800),  # Linear256->100
    )
    return theta["t0"] + sum(
        np.maximum(layer_flops / theta["C"], layer_bytes / theta["beta"])
        for layer_flops, layer_bytes in layer_costs
    )


def energy(image_size, batch, theta_energy):
    """E = P0*T + eps_F*F + eps_Q*Q, результат в джоулях.

    theta_energy — словарь с ключами:
    latency: словарь theta для latency();
    P0: базовая составляющая мощности, Вт;
    eps_F: дополнительная энергия на операцию, Дж/FLOP;
    eps_Q: дополнительная энергия на перемещение данных, Дж/байт.
    Энергетические коэффициенты постоянны и неотрицательны.
    """
    image_size = np.asarray(image_size, dtype=float)
    batch = np.asarray(batch, dtype=float)
    return (
        theta_energy["P0"] * latency(image_size, batch, theta_energy["latency"])
        + theta_energy["eps_F"] * flops(image_size, batch)
        + theta_energy["eps_Q"] * bytes_moved(image_size, batch)
    )
