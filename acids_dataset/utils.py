import importlib
import os
from pathlib import Path
import random
import math
import torch
import torchaudio
import gin


_CACHED_MODULES = {}
_VALID_BACKENDS = ['numpy', 'torch', 'jax']

def load_backend(backend: str):
    try:
        return importlib.import_module(backend)
    except ModuleNotFoundError:
        return None

def get_backend(backend: str):
    if backend not in _VALID_BACKENDS: 
        raise ValueError('backend %s not available.'%backend)
    if backend not in _CACHED_MODULES:
        _CACHED_MODULES[backend] = load_backend(backend)
    return _CACHED_MODULES[backend]

def load_file(file_path):
    return torchaudio.load(file_path)


def loudness(waveform: torch.Tensor, sample_rate: int):
    r"""
    Custom extension of torchaudio loudness, allowing loudness computation for small chunks
    """

    if not torch.is_tensor(waveform):
        waveform = torch.from_numpy(waveform)

    if waveform.size(-2) > 5:
        raise ValueError("Only up to 5 channels are supported.")

    gate_duration = min(0.4, waveform.size(-1) / sample_rate)
    overlap = 0.75
    gamma_abs = -70.0
    kweight_bias = -0.691
    gate_samples = int(round(gate_duration * sample_rate))
    step = int(round(gate_samples * (1 - overlap)))

    # Apply K-weighting
    waveform = torchaudio.functional.treble_biquad(waveform, sample_rate, 4.0, 1500.0, 1 / math.sqrt(2))
    waveform = torchaudio.functional.highpass_biquad(waveform, sample_rate, 38.0, 0.5)

    # Compute the energy for each block
    energy = torch.square(waveform).unfold(-1, gate_samples, step)
    energy = torch.mean(energy, dim=-1)

    # Compute channel-weighted summation
    g = torch.tensor([1.0, 1.0, 1.0, 1.41, 1.41], dtype=waveform.dtype, device=waveform.device)
    g = g[: energy.size(-2)]

    energy_weighted = torch.sum(g.unsqueeze(-1) * energy, dim=-2)
    loudness = -0.691 + 10 * torch.log10(energy_weighted)

    # Apply absolute gating of the blocks
    gated_blocks = loudness > gamma_abs
    gated_blocks = gated_blocks.unsqueeze(-2)

    energy_filtered = torch.sum(gated_blocks * energy, dim=-1) / torch.count_nonzero(gated_blocks, dim=-1)
    energy_weighted = torch.sum(g * energy_filtered, dim=-1)
    gamma_rel = kweight_bias + 10 * torch.log10(energy_weighted) - 10

    # Apply relative gating of the blocks
    gated_blocks = torch.logical_and(gated_blocks.squeeze(-2), loudness > gamma_rel.unsqueeze(-1))
    gated_blocks = gated_blocks.unsqueeze(-2)

    energy_filtered = torch.sum(gated_blocks * energy, dim=-1) / torch.count_nonzero(gated_blocks, dim=-1)
    energy_weighted = torch.sum(g * energy_filtered, dim=-1)
    LKFS = kweight_bias + 10 * torch.log10(energy_weighted)
    return LKFS


def checklist(item, n=1, copy=False):
    """Repeat list elemnts
    """
    if not isinstance(item, (list, )):
        if copy:
            item = [copy.deepcopy(item) for _ in range(n)]
        elif isinstance(item, torch.Size):
            item = [i for i in item]
        else:
            item = [item]*n
    return item

def get_random_hash(n=8):
    return "".join([chr(random.randrange(97,122)) for i in range(n)])
