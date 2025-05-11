import torch, torch.nn as nn
import dill
import numpy as np
import torch
from torchaudio.functional import resample
import os
from pathlib import Path
from .base import AcidsDatasetFeature
from typing import List, Dict
from .. import transforms as adt
import gin.torch

@gin.configurable(module="features")
class ModuleEmbedding(AcidsDatasetFeature):
    def __init__(self, 
                 module: nn.Module | None = None, 
                 module_path: str | Path | None = None, 
                 module_sr: int | None = None,
                 method: str = "forward", 
                 method_args = tuple(), 
                 method_kwargs = dict(), 
                 transforms: List[adt.Transform] | None = None,
                 batch_transforms: bool = True,
                 sr: int | None = None,
                 retain_module: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.sr = sr
        self.retain_module = retain_module
        self._init_module(module, module_path, module_sr, method, method_args, method_kwargs, self.device)
        self._init_transforms(transforms, batch_transforms)

    def __repr__(self):
        if hasattr(self, "_module"):
            module_repr = type(self._module).__name__
        else:
            module_repr = "<NoRecordedModule>"
        transform_repr = [type(t).__name__ for t in self._transforms]
        return "ModuleEmbedding(module=%s, method=%s, transforms=%s)"%(module_repr, self._method, transform_repr)

    @property
    def default_feature_name(self):
        return "embedding"

    @property
    def n_augmentations(self): 
        return len(self._transforms)

    @property
    def transforms(self): 
        return self._transforms

    def __getstate__(self):
        statedict = super().__getstate__()
        if not self.retain_module and "_module" in statedict: 
            del statedict['_module']
        return statedict

    def _init_module(self, module=None, module_path=None, module_sr=None, method="forward", method_args=tuple(), method_kwargs=dict(), device='cpu'):
        if isinstance(device, str): device = torch.device(device)
        if module is not None: 
            assert module_path is None, "either module or module_path must be given"
            self._module = module
        elif module_path is not None: 
            assert module is None, "either module or module_path must be given"
            self._load_module_from_path(module_path)
        else:
            raise ValueError("either module or module_path must be given")
        assert hasattr(self._module, method), "method %s not in %s"%(method, module)
        assert callable(getattr(self._module, method)), "method %s is not callable"%(method)
        self._method = method
        self._method_args = tuple()
        self._method_kwargs = dict()
        self._module = self._module.to(device)
        assert hasattr(self._module, self._method), "module does not have callback %s"%("forward")
        assert callable(getattr(self._module, self._method)), "attribtue %s is not a method"%self._method
        self._module.eval()
        self._module_sr = module_sr

    def _load_module_from_path(self, module_path): 
        if os.path.splitext(module_path)[1] == ".ts":
            assert not self.retain_module, "retain_module must be False if given a .ts file (TorchScript modules are not pickable)"
            self._module = torch.jit.load(module_path, map_location=torch.device('cpu'))
        else:
            raise RuntimeError("cannot load model %s"%module_path)

    def _init_transforms(self, transforms, batch_transforms):
        if transforms is None: transforms = []
        for i, t in enumerate(transforms): 
            if isinstance(t, list): 
                for j, t_tmp in enumerate(t):
                    if isinstance(t_tmp, type):
                        t[j] = t_tmp(sr=self.sr) 
                transforms[i] = adt.Compose(*t)
            else:
                if isinstance(t, type): 
                    transforms[i] = t(sr=self.sr)
                assert isinstance(transforms[i], adt.Transform), "wrong type for transform #%d: %s"%(i, type(t))
        self._transforms = transforms
        self._batch_transforms = batch_transforms

    def _audio_from_fragment(self, fragment):
        sr = fragment.get_buffer('waveform').sampling_rate
        waveform = fragment.raw_audio
        if self._module_sr is not None and sr != self._module_sr:
            waveform = resample(waveform, sr, self._module_sr)
        return waveform

    def _get_transforms(self, audio):
        audios = [audio]
        for transform in self._transforms: 
            audios.append(transform(audio, _force_transform=True))
        return np.stack(audios, 0)

    def _process_audio(self, audio):
        with torch.set_grad_enabled(False):
            audio = torch.from_numpy(self._get_transforms(audio)).to(torch.get_default_dtype())
            audio = audio.to(self.device)
            if audio.ndim == 1: 
                audio = audio[None, None]
            elif audio.ndim == 2: 
                audio = audio[None]
            out = getattr(self._module, self._method)(audio, *self._method_args, **self._method_kwargs)
        return out.cpu().numpy()

    def from_fragment(self, fragment, write: bool = True):
        if self._module is None: 
            raise RuntimeError('ModuleEmbedding has no _module. If coming from pickling, set retain_module kwargs to True when saving feature in the database.')
        audio = self._audio_from_fragment(fragment)
        out = self._process_audio(audio)
        if write: 
            fragment.put_array(self.feature_name, out)
        return out
        
