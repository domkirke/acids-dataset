import gin
from torchaudio.transforms import MelSpectrogram
from .base import AcidsDatasetFeature


@gin.configurable(module="features")
class Mel(AcidsDatasetFeature):
    def __init__(
            self, 
            sr: int = 44100, 
            **kwargs
    ):
        super().__init__()
        self.sr = sr
        self.kwargs = kwargs
        self.mel_spectrogram = MelSpectrogram(**kwargs, sample_rate=sr)

    def __repr__(self):
        return "Mel(%s, sr=%d)"%(self.kwargs, self.sr)

    def extract(self, fragment):
        data = fragment.raw_audio



