import torch, gin.torch
import numpy as np
import re
import librosa
from .base import AcidsDatasetFeature
from ..utils import apply_nested
from typing import Optional, Callable
from collections import Counter

_AD_F0_METHODS = ["yin", "pyin"] 

yin_config = """
{{NAME}}:
\tmode="yin"
\tfmin = "C2"
\tfmax = "C7"
\tframe_length = 2048
\ttrough_threshold=0.1
\tcenter=True
\tpad_mode="constant"
"""

pyin_config = """
{{NAME}}:
\tmode="pyin"
\tfmin = "C2"
\tfmax = "C7"
\tframe_length = 2048
\tn_thresholds=100
\tbeta_parameters=(2, 18)
\tboltzmann_parameter=2
\tresolution=0.1
\tmax_transition_rate=35.92
\tswitch_prob=0.01
\tno_trough_prob=0.01
\tcenter=True
\tpad_mode="constant"
"""

_post_config = """
features.parse_features:
    features = @features.{{NAME}}
"""

_configs = {
    "yin": f"{yin_config}\n\n{_post_config}", 
    "pyin": f"{pyin_config}\n\n{_post_config}", 
}

try:
    import torchcrepe 
    _AD_F0_METHODS.append("crepe")
    crepe_config = "\n\t".join(['{{NAME}}:', 'mode="crepe"', "fmin=50", "fmax=2006", 'decoder="viterbi"']) 
    _configs['crepe'] = f"{crepe_config}\n\n{_post_config}" 
except ModuleNotFoundError: 
    pass

try:
    import pesto 
    _AD_F0_METHODS.append("pesto")
    pesto_config = "\n\t".join(['{{NAME}}:', 'mode="pesto"']) 
    _configs['pesto'] = f"{pesto_config}\n\n{_post_config}" 
except ModuleNotFoundError: 
    pass

try: 
    import pyworld
    _AD_F0_METHODS.append("world")
    pw_config = "\n\t".join(['{{NAME}}:', 'mode="world"']) 
    _configs['world'] = f"{pw_config}\n\n{_post_config}" 
except ModuleNotFoundError:
    pass

try: 
    import parselmouth
    _AD_F0_METHODS.append("praat")
    praat_config = "\n\t".join(['{{NAME}}:', 'mode="praat"'])
    _configs['praat'] = f"{praat_config}\n\n{_post_config}" 
except ModuleNotFoundError:
    pass


@gin.configurable(module="features")
class F0(AcidsDatasetFeature):
    _valid_modes_ = _AD_F0_METHODS
    gin_configs = _configs
    def __init__(
            self, 
            mode="pyin",
            sr: int = 44100, 
            name: Optional[str] = None,
            hash_from_feature: Optional[Callable] = None, 
            device: torch.device = None, 
            metadata = {},
            **kwargs
    ):
        super().__init__(name=name, hash_from_feature=hash_from_feature, device=device, metadata=metadata)
        self.sr = sr
        assert self.sr is not None, "F0 needs sr keyword"
        self.mode = mode
        self.kwargs = kwargs

    def __repr__(self):
        return "F0(mode=%s, sr=%d)"%(self.mode, self.sr)

    @property
    def default_feature_name(self):
        return "f0"

    @staticmethod
    def predict(audio: np.ndarray, mode: str, sr: int, device=None, **add_kwargs):
        # ["yin", "pyin", "crepe", "pesto"]
        #TODO if channels > 2?  
        assert mode in _AD_F0_METHODS, "mode %s not available. Available modes : %s"%_AD_F0_METHODS
        if mode in ["yin", "pyin"]:
            f_min = add_kwargs.get('fmin')
            if isinstance(f_min, str): add_kwargs['fmin'] = librosa.note_to_hz(f_min)
            f_max = add_kwargs.get('fmax')
            if isinstance(f_max, str): add_kwargs['fmax'] = librosa.note_to_hz(f_max)
            if mode == "yin":
                f0 = librosa.yin(audio, sr=sr, **add_kwargs)
            elif mode == "pyin":
                f0, voiced_flag, voiced_prob = librosa.pyin(audio, sr=sr, **add_kwargs)
                f0 = np.stack([f0, voiced_prob], axis=0)
        elif mode == "crepe": 
            if add_kwargs.get("decoder") is None: 
                add_kwargs["decoder"] = "viterbi"
            if isinstance(add_kwargs.get("decoder"), str):
                add_kwargs["decoder"] = getattr(torchcrepe.decode, add_kwargs["decoder"])
            f0 = torchcrepe.predict(torch.from_numpy(audio).float(), sr, return_harmonicity=False, return_periodicity=False, device=device, **add_kwargs).numpy()
        elif mode == "pesto": 
            out = pesto.predict(torch.from_numpy(audio).float(), sr)
            f0 = torch.cat([out[0][None], out[1], out[2]], dim=0)
        elif mode == "world": 
            f0 = []
            if audio.ndim == 1: audio = audio[None]
            for a in audio:
                _f, t = pyworld.dio(a, sr)
                f = pyworld.stonemask(a, _f, t, sr) 
                f0.append(f)
            f0 = np.stack(f0)
        elif mode == "praat":
            sound = parselmouth.Sound(audio, sampling_frequency=sr)
            f0 = sound.to_pitch(**add_kwargs).to_array()
            f0 = np.stack([f0[0]['frequency'], f0[0]['strength']], 0)
        f0 = np.where(np.isnan(f0), 0.0, f0)
        f0 = np.where(np.isinf(f0), 0.0, f0)
        return f0

    def from_fragment(self, fragment, write: bool = True):
        data = fragment.raw_audio
        try: 
            f0 = self.predict(data, self.mode, self.sr, device=self.device, **self.kwargs)
        except RuntimeError:
            return  
        if write:
            fragment.put_array(self.feature_name, f0)
        return f0


def f0_to_pitch(x):
    if np.allclose(x, 0.):
        x = -1
    else:
        note = librosa.hz_to_note(x)
        try: 
            root = re.match(r"^([A-G]+[#b♯♭]?)(\-?\d+)$", note).groups()[0]
            x = Pitch.idx_hash()[root]
        except Exception: 
            pass
    return x
    

@gin.configurable(module="features")
class Pitch(F0): 
    
    def __repr__(self):
        return "Pitch(mode=%s, sr=%d)"%(self.mode, self.sr)

    @property
    def default_feature_name(self):
        return "pitch"

    @classmethod
    def note_hash(cls):
        return {-1: "X", 0: "A", 1: "A♯", 2:"B", 3:"C", 4:"C♯", 5:"D", 6:"D♯", 7:"E", 8: "F", 9:"F♯", 10:"G", 11:"G♯"}
    @classmethod
    def idx_hash(cls):
        return {v: k for k, v in cls.note_hash().items()}

    @property
    def has_hash(self):
        return True 
    
    def hash_from_feature(self, meta): 
        if self.mode in ["pyin", "pesto"]:
            pitch = Counter(meta[0].flatten().tolist()).most_common(1)[0][0]
        else:
            pitch = Counter(meta.flatten().tolist()).most_common(1)[0][0]
        return Pitch.note_hash()[pitch]

    @staticmethod
    def predict(audio: np.ndarray, mode: str, sr: int, **add_kwargs):
        f0 = F0.predict(audio, mode, sr, **add_kwargs)
        note = apply_nested(f0_to_pitch, f0.tolist())
        return np.array(note)


