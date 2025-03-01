import gin
import torch
from torchaudio.transforms import MelSpectrogram
from .base import AcidsDatasetFeature


@gin.configurable(module="features")
class Mel(AcidsDatasetFeature):
    def __init__(
            self, 
            sr: int = 44100, 
            **kwargs
    ):
        super().__init__(**kwargs)
        self.sr = sr
        self.kwargs = kwargs
        self.mel_spectrogram = MelSpectrogram(**kwargs, sample_rate=sr)

    def __repr__(self):
        return "Mel(%s, sr=%d)"%(self.kwargs, self.sr)

    @property
    def has_hash(self):
        return False

    @property
    def feature_name(self):
        return f"mel_{self.mel_spectrogram.n_mels}"

    def extract(self, fragment, current_key, feature_hash):
        data = torch.from_numpy(fragment.raw_audio).float()
        try: 
            mels = self.mel_spectrogram(data)
        except RuntimeError:
            return  
        fragment.put_array(self.feature_name, mels)
        




