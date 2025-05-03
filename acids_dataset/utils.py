import importlib
import inspect, pkgutil
import logging
import enum
from contextlib import ContextDecorator
import os
from pathlib import Path
import random
import math
import torch
import torchaudio
import gin



_CACHED_MODULES = {}
_VALID_BACKENDS = ['numpy', 'torch', 'jax']

def load_backend(backend: str):
    try:
        return importlib.import_module(backend)
    except ModuleNotFoundError:
        return None

def get_backend(backend: str):
    if backend not in _VALID_BACKENDS: 
        raise ValueError('backend %s not available.'%backend)
    if backend not in _CACHED_MODULES:
        _CACHED_MODULES[backend] = load_backend(backend)
    return _CACHED_MODULES[backend]

def load_file(file_path):
    return torchaudio.load(file_path)




class GinEnv(object):
    def __init__(self, paths=[], configs=[], bindings=[]):
        self._paths = checklist(paths)
        self._original_config = None
        self._new_configs = checklist(configs)
        self._new_bindings = checklist(bindings)
        self._loc_prefix = None
    
    def __enter__(self):
        self._loc_prefix = list(gin.config._LOCATION_PREFIXES)
        gin.config._LOCATION_PREFIXES = self._paths
        self._original_config = gin.config_str()
        with gin.unlock_config():
            gin.clear_config()
            gin.parse_config_files_and_bindings(self._new_configs, self._new_bindings)

    def __exit__(self, *args):
        gin.config._LOCATION_PREFIXES = self._loc_prefix
        with gin.unlock_config(): 
            gin.parse_config(self._original_config)


def loudness(waveform: torch.Tensor, sample_rate: int):
    r"""
    Custom extension of torchaudio loudness, allowing loudness computation for small chunks
    """

    if not torch.is_tensor(waveform):
        waveform = torch.from_numpy(waveform)

    if waveform.size(-2) > 5:
        raise ValueError("Only up to 5 channels are supported.")

    gate_duration = min(0.4, waveform.size(-1) / sample_rate)
    overlap = 0.75
    gamma_abs = -70.0
    kweight_bias = -0.691
    gate_samples = int(round(gate_duration * sample_rate))
    step = int(round(gate_samples * (1 - overlap)))

    # Apply K-weighting
    waveform = torchaudio.functional.treble_biquad(waveform, sample_rate, 4.0, 1500.0, 1 / math.sqrt(2))
    waveform = torchaudio.functional.highpass_biquad(waveform, sample_rate, 38.0, 0.5)

    # Compute the energy for each block
    energy = torch.square(waveform).unfold(-1, gate_samples, step)
    energy = torch.mean(energy, dim=-1)

    # Compute channel-weighted summation
    g = torch.tensor([1.0, 1.0, 1.0, 1.41, 1.41], dtype=waveform.dtype, device=waveform.device)
    g = g[: energy.size(-2)]

    energy_weighted = torch.sum(g.unsqueeze(-1) * energy, dim=-2)
    loudness = -0.691 + 10 * torch.log10(energy_weighted)

    # Apply absolute gating of the blocks
    gated_blocks = loudness > gamma_abs
    gated_blocks = gated_blocks.unsqueeze(-2)

    energy_filtered = torch.sum(gated_blocks * energy, dim=-1) / torch.count_nonzero(gated_blocks, dim=-1)
    energy_weighted = torch.sum(g * energy_filtered, dim=-1)
    gamma_rel = kweight_bias + 10 * torch.log10(energy_weighted) - 10

    # Apply relative gating of the blocks
    gated_blocks = torch.logical_and(gated_blocks.squeeze(-2), loudness > gamma_rel.unsqueeze(-1))
    gated_blocks = gated_blocks.unsqueeze(-2)

    energy_filtered = torch.sum(gated_blocks * energy, dim=-1) / torch.count_nonzero(gated_blocks, dim=-1)
    energy_weighted = torch.sum(g * energy_filtered, dim=-1)
    if energy_weighted.isnan():
        return torch.tensor(-torch.inf)
    LKFS = kweight_bias + 10 * torch.log10(energy_weighted)
    return LKFS

def checklist(item, n=1, copy=False):
    """Repeat list elemnts
    """
    if not isinstance(item, (list, )):
        if copy:
            item = [copy.deepcopy(item) for _ in range(n)]
        elif isinstance(item, torch.Size):
            item = [i for i in item]
        else:
            item = [item]*n
    return item

def get_random_hash(n=8):
    return "".join([chr(random.randrange(97,122)) for i in range(n)])


@gin.configurable()
def parse_features(features=None, device=None):
    if features is None: 
        return []
    else:
        return [f(device=device) for f in checklist(features)]

def feature_from_gin_config(config_path):
    with GinEnv(configs=config_path):
        feature = parse_features
    return feature


def get_available_cuda_device():
    #TODO (or not, let user decide with CUDA_VISIBLE_DEVICES)
    return torch.device('cuda')


def get_default_accelerated_device():
    cuda_available = os.environ.get('USE_CUDA', True) or torch.cuda.is_available()
    if cuda_available:
        if os.environ.get('CUDA_AVAILABLE_DEVICES'):
            get_available_cuda_device()
        else:
            return torch.device('cuda')
    mps_available = os.environ.get('USE_MPS', False) or torch.mps.is_available()
    if mps_available: 
        return torch.device('mps')
    return torch.device('cpu')

def get_accelerated_device(accelerator = None):
    if accelerator == "cpu":
        return torch.device("cpu")
    elif accelerator == "cuda":
        return torch.device('cpu') if not torch.cuda.is_available() else get_available_cuda_device()
    elif accelerator == "mps":
        return torch.device('cpu') if not torch.mps.is_available() else torch.device('mps')
    else:
        raise ValueError('accelerator value %s not handled.'%accelerator)


class PadMode(enum.Enum):
    DISCARD = 0
    ZERO = 1
    REPEAT = 2
    REPLICATE = 3
    REFLECT = 4


def pad(
        chunk,
        target_size,
        pad_mode,
        discard_if_lower_than: None | float | int = None
    ):
    if pad_mode == PadMode.DISCARD:
        return None
    if discard_if_lower_than is not None:
        if isinstance(discard_if_lower_than, int):
            if chunk.shape[-1] < discard_if_lower_than: 
                return None
        elif isinstance(discard_if_lower_than, float):
            if chunk.shape[-1] < int(discard_if_lower_than * target_size): 
                return None
        else: 
            raise TypeError('got type %s for discard_if_lower_than'%type(discard_if_lower_than))
    if pad_mode == PadMode.ZERO:
        return torch.nn.functional.pad(chunk, (0, target_size - chunk.shape[-1]), mode="constant", value=0.)
    elif pad_mode == PadMode.REPEAT:
        n_iter = 0
        while chunk.shape[-1] < target_size:
            if n_iter > 1: 
                logging.warning('applying repeat pad more than once ; may provoke undesired behaviour.')
            chunk = torch.nn.functional.pad(chunk, (0, min(target_size - chunk.shape[-1], chunk.shape[-1] - 1)), mode="circular")
            n_iter += 1
        return chunk
    elif pad_mode == PadMode.REPLICATE:
        return torch.nn.functional.pad(chunk, (0, target_size - chunk.shape[-1]), mode="replicate")
    elif pad_mode == PadMode.REFLECT:
        n_iter = 0
        while chunk.shape[-1] < target_size:
            chunk = torch.nn.functional.pad(chunk, (0, min(target_size - chunk.shape[-1], chunk.shape[-1] - 1)), mode="reflect")
            n_iter += 1
        if n_iter > 1: 
            logging.warning('applied reflect pad more than once ; may provoke undesired behaviour.')
        return chunk
    else:
        raise ValueError('pad mode %s not recognized.'%pad_mode)

def mirror_pad(tensor, target_size, mode="reflect", value=0):
    if (tensor.shape[-1] < target_size):
        pad_len = target_size - tensor.shape[-1]
        pad_bef, pad_aft = math.floor(pad_len / 2), math.ceil(pad_len / 2)
        return torch.nn.functional.pad(tensor, (pad_bef, pad_aft), mode=mode, value=value)
    elif(tensor.shape[-1] > target_size): 
        crop_len = target_size - tensor.shape[-1]
        crop_bef, crop_aft = math.floor(crop_len / 2), math.ceil(crop_len / 2)
        return tensor[..., crop_bef:-crop_aft]
    else:
        return tensor


def match_loudness(signal, target_signal, sr):
    if signal.ndim == 3: 
        return torch.stack(0, map(match_loudness, signal))
    current_loudness = torchaudio.functional.loudness(signal, sr)
    target_loudness = torchaudio.functional.loudness(target_signal, sr)
    return torchaudio.functional.gain(signal, target_loudness - current_loudness)
    

def walk_modules(package):
    yield package
    if hasattr(package, '__path__'):
        for _, name, ispkg in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
            submod = importlib.import_module(name)
            yield from walk_modules(submod)


def get_subclasses_from_package(package, filter_class):
    transform_list = []
    for mod in walk_modules(package):
        for name, obj in inspect.getmembers(mod):
            if isinstance(obj, type):
                if issubclass(obj, filter_class) and (obj not in transform_list):
                    transform_list.append(obj)
    return transform_list


def set_gin_constant(constant, value):
    if constant in gin.config._CONSTANTS:
        gin.config._CONSTANTS[constant] = value
    else:
        gin.constant(constant, value)