from typing import Optional, Callable, Any
import torch
import torchaudio
from collections import UserDict
from absl import logging
from ..utils import load_file

class FeatureException(Exception):
    pass


class FileHash(UserDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._idx = 0

    @property
    def current_id(self):
        return int(self._idx)

    def __setitem__(self, key: Any, item: Any) -> None:
        self._idx += 1
        return super().__setitem__(key, item)


class AcidsDatasetFeature(object):
    # denylist = ['audio_path', 'audio_data', 'sr', 'start', 'end', 'duration']
    denylist = []
    has_hash = False
    def __init__(
            self, 
            name: Optional[str] = None,
            hash_from_feature: Optional[Callable] = None, 
        ):
        self.feature_name = name or self.default_feature_name
        self.hash_from_feature = hash_from_feature
        if self.hash_from_feature is not None: 
            self.has_hash = True

    @property
    def default_feature_name(self):
        return type(self).__name__.lower()

    def extract(self, fragment, **kwargs):
        raise NotImplementedError()

    def close(self):
        """if some features has side effects like buffering, empty buffers and delete files"""
        pass

    def _sample(self, pos: int | float | None, sr: int):
        if isinstance(pos, float):
            return int(pos * sr)
        return pos

    def get(self, audio_fragment):
        raise NotImplementedError()

    def load_file(self, path = None, start = None, end = None, duration = None, target_sr = None, channels = None):
        path = path or self.path
        out, sr = load_file(path)

        if out.shape[0] > channels:
            out = out[:channels]
        elif out.shape[0] < channels: 
            out = out[torch.arange(channels)%out.shape[0]]

        # get start
        start = self._sample(start or self.start, sr)
        if end is None and duration is not None: 
            end = duration if start is None else start + duration
        # get end
        end = self._sample(end, sr)

        out = out[..., start:end]
        if target_sr != sr and target_sr is not None:
            out = torchaudio.functional.resample(out, sr, target_sr)
        return out    
        