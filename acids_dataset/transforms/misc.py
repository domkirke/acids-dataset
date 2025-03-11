import torch
import gin
import bisect
from random import random

import torchaudio
from torchaudio.functional import lfilter

from .base import Transform, RandomApply
from typing import Tuple
from .resample_poly import resample_poly
from .utils import random_phase_mangle, normalize_signal, get_derivator, get_integrator



@gin.configurable(module="transforms")
class AudioResample(Transform):
    """
    Resample target signal to target sample rate.
    """
    def __init__(self, orig_sr: int, target_sr: int):
        self.orig_sr = orig_sr
        self.target_sr = target_sr

    def __call__(self, x):
        return torchaudio.functional.resample(torch.from_numpy(x).float(), self.orig_sr, self.target_sr).numpy()


@gin.configurable(module="transforms")
class RandomPitch(Transform):
    def __init__(self, n_signal, pitch_range = [0.7, 1.3], max_factor: int = 20, prob: float = 0.5):
        self.n_signal = n_signal
        self.pitch_range = pitch_range
        self.factor_list, self.ratio_list = self._get_factors(max_factor, pitch_range)
        self.prob = prob

    def _get_factors(self, factor_limit, pitch_range):
        factor_list = []
        ratio_list = []
        for x in range(1, factor_limit):
            for y in range(1, factor_limit):
                if (x==y):
                    continue
                factor = x / y
                if factor <= pitch_range[1] and factor >= pitch_range[0]:
                    i = bisect.bisect_left(factor_list, factor)
                    factor_list.insert(i, factor)
                    ratio_list.insert(i, (x, y))
        return factor_list, ratio_list

    def __call__(self, x):
        perform_pitch = bool(torch.bernoulli(torch.tensor(self.prob)))
        if not perform_pitch:
            return x
        random_range = list(self.pitch_range)
        random_range[1] = min(random_range[1], x.shape[-1] / self.n_signal)
        random_pitch = random() * (random_range[1] - random_range[0]) + random_range[0]
        ratio_idx = bisect.bisect_left(self.factor_list, random_pitch)
        if ratio_idx == len(self.factor_list):
            ratio_idx -= 1
        up, down = self.ratio_list[ratio_idx]
        x_pitched = resample_poly(x, up, down, padtype='mean', axis=-1)
        return x_pitched


@gin.configurable(module="transforms")
class RandomCrop(Transform):
    """
    Randomly crops signal to fit n_signal samples
    """
    def __init__(self, n_signal):
        self.n_signal = n_signal

    def __call__(self, x):
        in_point = torch.randint(0, x.shape[-1] - self.n_signal)
        x = x[..., in_point:in_point + self.n_signal]
        return x

@gin.configurable(module="transforms")
class Dequantize(Transform):
    def __init__(self, bit_depth):
        self.bit_depth = bit_depth

    def __call__(self, x):
        x += torch.rand(*x.shape) / 2**self.bit_depth
        return x


@gin.configurable(module="transforms")
class Compress(Transform):
    def __init__(self, time="0.1,0.1", lookup="6:-70,-60,-20 ", gain="0", sr=44100):
        self.sox_args = ['compand', time, lookup, gain]
        self.sr = sr

    def __call__(self, x: torch.Tensor):
        x = torchaudio.sox_effects.apply_effects_tensor(torch.from_numpy(x).float(), self.sr, [self.sox_args])[0].numpy()
        return x

@gin.configurable(module="transforms")
class RandomCompress(Transform):
    def __init__(self, threshold = -40, amp_range = [-60, 0], attack=0.1, release=0.1, prob=0.8, limit=True, sr=44100):
        assert prob >= 0. and prob <= 1., "prob must be between 0. and 1."
        self.amp_range = amp_range
        self.threshold = threshold
        self.attack = attack
        self.release = release
        self.prob = prob
        self.limit = limit
        self.sr = sr

    def __call__(self, x: torch.Tensor):
        perform = bool(torch.bernoulli(torch.full((1,), self.prob)))
        if perform:
            amp_factor = torch.rand((1,)) * (self.amp_range[1] - self.amp_range[0]) + self.amp_range[0]
            x_aug = torchaudio.sox_effects.apply_effects_tensor(torch.from_numpy(x).float(),
                                                            self.sr,
                                                             [['compand', f'{self.attack},{self.release}', f'6:-80,{self.threshold},{float(amp_factor)}']]
                                                            )[0].numpy()
            if (self.limit) and (x_aug.abs().max() > 1): 
                x_aug = x_aug / x_aug.abs().max()
            return x_aug
        else:
            return x

@gin.configurable(module="transforms")
class RandomGain(Transform):
    def __init__(self, gain_range: Tuple[int, int] = [-6, 3], prob: float = 0.5, limit = True):
        assert prob >= 0. and prob <= 1., "prob must be between 0. and 1."
        self.gain_range = gain_range
        self.prob = prob
        self.limit = limit

    def __call__(self, x: torch.Tensor):
        perform = bool(torch.bernoulli(torch.full((1,), self.prob)))
        if perform:
            gain_factor = torch.rand(1)[None, None][0] * (self.gain_range[1] - self.gain_range[0]) + self.gain_range[0]
            amp_factor = torch.pow(10, gain_factor / 20)
            x_amp = x * amp_factor
            if (self.limit) and (torch.abs(x_amp).max() > 1): 
                x_amp = x_amp / torch.abs(x_amp).max()
            return x
        else:
            return x


@gin.configurable(module="transforms")
class RandomMute(Transform):
    def __init__(self, prob: torch.Tensor = 0.1):
        assert prob >= 0. and prob <= 1., "prob must be between 0. and 1."
        self.prob = prob

    def __call__(self, x: torch.Tensor):
        mask = torch.bernoulli(torch.full((x.shape[0],), 1 - self.prob))
        return x * mask

@gin.configurable(module="transforms")
class RandomPhase(RandomApply):
    def __init__(self, 
                 p: float = 0.8, 
                 min_f: float = 20, 
                 max_f: float = 2000, 
                 amplitude: float = .99, 
                 **kwargs):
        super().__init__(p=p, **kwargs)
        self.min_f = min_f
        self.max_f = max_f
        self.amplitude = amplitude

    def __repr__(self):
        return f"RandomPhase(p={self.p}, min_f={self.min_f}, max_f={self.max_f}, amplitude={self.amplitude}, sr={self.sr})"

    def transform(self, x):
        return random_phase_mangle(x, self.min_f, self.max_f, self.amplitude, self.sr)


@gin.configurable(module="transforms")
class FrequencyMasking(Transform):
    def __init__(self, prob = 0.5, max_size: int = 80):
        self.prob = prob
        self.max_size = max_size

    def __call__(self, x: torch.Tensor):
        perform = bool(torch.bernoulli(torch.full((1,), self.prob)))
        if not perform:
            return x
        spectrogram = x.stft(x, nperseg=4096)[2]
        mask_size = random.randrange(1, self.max_size)
        freq_idx = random.randrange(0, spectrogram.shape[-2] - mask_size)
        spectrogram[..., freq_idx:freq_idx+mask_size, :] = 0
        x_inv = torch.fft.irfft(spectrogram)[1]
        return x_inv
            
@gin.configurable(module="transforms")
class Normalize(Transform):
    def __call__(self, x):
        return normalize_signal(x)

@gin.configurable(module="transforms")
class Derivator(Transform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._derivator = get_derivator(self.sr)
    def __call__(self, x):
        return self._derivator(x)

@gin.configurable(module="transforms")
class Integrator(Transform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._integrator = get_integrator(self.sr)
    def __call__(self, x):
        return self._integrator(x)
