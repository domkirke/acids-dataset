from typing import Optional, Any
import torchaudio
from absl import logging
from ..utils import load_file

class FeatureException(Exception):
    pass

class AcidsDatasetFeature(object):
    # denylist = ['audio_path', 'audio_data', 'sr', 'start', 'end', 'duration']
    denylist = []
    def __init__(
            self, 
            # audio_path: Optional[str] = None, 
            # audio_data: Optional[Any] = None,
            # start: Optional[float | int] = None,
            # end: Optional[float | int] = None,
            # duration: Optional[float | int] = None,
            # sr: Optional[int] = None
        ):
        pass
        # self.audio_path = audio_path
        # self.audio_data = audio_data
        # self.start = start
        # self.end = end
        # self.duration = duration
        # self.sr = sr

    def extract(self, fragment, **kwargs):
        raise NotImplementedError()

    def _sample(self, pos: int | float | None, sr: int):
        if isinstance(pos, float):
            return int(pos * sr)
        return pos

    def get(self, audio_fragment):
        raise NotImplementedError()

    def load_file(self, path = None, start = None, end = None, duration = None, target_sr = None):
        path = path or self.path
        out, sr = load_file(path)

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
        