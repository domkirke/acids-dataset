import gin
import torch
from torchaudio.transforms import MelSpectrogram
from .base import AcidsDatasetFeature


@gin.configurable(module="features")
class Mel(AcidsDatasetFeature):
    def __init__(
            self, 
            sr: int = 44100, 
            name: str = None,
            **kwargs
    ):
        self.sr = sr
        self.mel_spectrogram = MelSpectrogram(**kwargs, sample_rate=sr)
        super().__init__()

    def __repr__(self):
        return f"Mel(n_mels = {self.mel_spectrogram.n_mels}, n_fft = {self.mel_spectrogram.n_fft}, sr={self.sr})"

    @property
    def has_hash(self):
        return False

    @property
    def default_feature_name(self):
        return f"mel_{self.mel_spectrogram.n_mels}"

    def from_fragment(self, fragment, write: bool = True):
        data = torch.from_numpy(fragment.raw_audio).float()
        try: 
            mels = self.mel_spectrogram(data)
        except RuntimeError:
            return  
        if write:
            fragment.put_array(self.feature_name, mels)
        return mels
        




