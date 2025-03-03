import gin
import torch

from .base import AcidsDatasetFeature
from ..utils import loudness


@gin.configurable(module="features")
class Loudness(AcidsDatasetFeature):
    def __init__(
            self, 
            sr: int = 44100, 
            **kwargs
    ):
        super().__init__(**kwargs)
        self.sr = sr
        self.kwargs = kwargs

    def __repr__(self):
        return "Loudness(sr=%d)"%(self.sr)

    @property
    def has_hash(self):
        return False

    @property
    def default_feature_name(self):
        return "loudness"

    def extract(self, fragment, current_key, feature_hash):
        data = torch.from_numpy(fragment.raw_audio).float()
        try: 
            data_loudness = loudness(data, self.sr)
        except RuntimeError:
            return  
        fragment.put_array(self.feature_name, data_loudness)
        




