import pytest
import os
import random
import torchaudio
import numpy as np
from pathlib import Path
import torch
import pkgutil
import importlib
import inspect
from acids_dataset import transforms
from acids_dataset.utils import get_subclasses_from_package



@pytest.mark.parametrize("transform", get_subclasses_from_package(transforms, transforms.Transform))
@pytest.mark.parametrize("sr", [22050, 44100])
@pytest.mark.parametrize("in_type", ["torch", "numpy"])
def test_transform(transform: transforms.Transform, sr: int, in_type: str, t: float = 10.0):
    if in_type == "torch":
        x = torch.randn(4, 1, int(t * sr))
    elif in_type == "numpy":
        x = np.random.random((1, 1, int(t * sr)))
    # test forward
    t = transform(sr=sr)
    out = t(x)
    if torch.is_tensor(out) or isinstance(torch, np.ndarray):
        assert type(out) == type(x)

    if transform.allow_random:
        t = transform(sr=sr, p=0.4)
        out = t(x)
        if torch.is_tensor(out) or isinstance(torch, np.ndarray):
            assert type(out) == type(x)
    # specialized tests
    if hasattr(transform, "test"):
        for obj, test_fn in transform.test():
            assert test_fn(x, obj(x))


def import_audio(path, sr):
    x, orig_sr = torchaudio.load(str(Path(__file__).parent / "audio" / path))
    if orig_sr != sr: 
        x = torchaudio.functional.resample(x, orig_sr, sr)
    return x


test_signals = {
    "sin": lambda t, sr: torch.sin(2 * torch.pi * random.randrange(80, 600) * torch.arange(int(t * sr)) / sr)[None], 
    "noise": lambda t, sr: (torch.rand(1, int(t*sr)) * 2 - 1),
    "bass": lambda t, sr: import_audio("bass.wav", sr)
}


@pytest.mark.parametrize("transform", inspect_module_objects(transforms))
@pytest.mark.parametrize("sr", [22050, 44100])
def test_transform_with_audio(transform, sr, t=2.0, n_samples=8):
    out_dir = Path(__file__).parent / "outs" / "transforms" / f"{transform.__name__}" / str(sr)
    os.makedirs(out_dir, exist_ok=True)
    for k, v in test_signals.items():
        outs = []
        for _ in range(n_samples):
            x = v(t, sr)
            tr = transform(sr=sr)
            out = tr(x)
            if tr.type_hash(out) not in [tr.input_types.torch, tr.input_types.numpy]:
                pytest.skip("output is not audio")
            outs.append(out)
        out = torch.concatenate(outs, -1)
        torchaudio.save(str(out_dir / f"{k}.wav"), out, sample_rate=sr)