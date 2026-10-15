import inspect
import difflib
from absl import logging
import torch
from abc import abstractmethod, ABCMeta
from pathlib import Path
from typing import Optional, List, Dict, Any
from .. import transforms
from .utils import _outs_from_pattern, _transform_outputs, is_multi_index, multi_index_to_list, collate_items

TransformType = Optional[transforms.Transform | List[transforms.Transform] | Dict[str, transforms.Transform]]

_DATALOADER_ARGS = set(inspect.signature(torch.utils.data.DataLoader.__init__).parameters.keys()) - {'self', 'dataset'}


def is_preprocessed_path(path: str | Path) -> bool:
    """returns True if path is a pre-processed dataset folder (i.e. written by a dataset writer)."""
    if isinstance(path, (list, tuple)):
        return False
    path = Path(path)
    return path.is_dir() and (path / "metadata.yaml").exists() and (path / "data.mdb").exists()


def chunk_length_to_samples(chunk_length: int | float, sr: int) -> int:
    """chunk lengths follow parsers convention : int are samples, float are seconds."""
    if isinstance(chunk_length, int):
        return chunk_length
    return int(round(chunk_length * sr))


class _AudioDatasetMeta(ABCMeta):
    """chooses the dataset class from the path when AudioDataset is instantiated, and parses presets."""
    def __call__(cls, *args, preset: str | Path | None = None, preset_overrides: List[str] | None = None, **kwargs):
        if cls is AudioDataset:
            path = args[0] if len(args) > 0 else kwargs.get('path', kwargs.get('db_path'))
            cls = AudioDataset.dataset_class_from_path(path)
        if preset is None and not preset_overrides:
            return super().__call__(*args, **kwargs)
        return cls._init_from_preset(preset, preset_overrides or [], *args, **kwargs)


class AudioDataset(torch.utils.data.Dataset, metaclass=_AudioDatasetMeta):
    """Base class for audio datasets. Instantiating AudioDataset directly returns a PreprocessedAudioDataset
    if path is a pre-processed dataset folder, and an OnlineAudioDataset otherwise (raw audio folders). 
    
    Datasets can be instantiated with a preset (see acids_dataset.presets) with the preset keyword, and 
    additional gin bindings with preset_overrides. Explicit arguments override preset values."""

    # dataset arguments that override preset values
    _preset_values = {}

    @classmethod
    def _check_kwargs(cls, kwargs: Dict[str, Any], other_class: type) -> None:
        """raises an error for unknown arguments (with suggestions), and warns for arguments of the other dataset class 
        (such that the same arguments can be given for raw and pre-processed folders)."""
        def params(c):
            return [p for p in inspect.signature(c.__init__).parameters if p not in ["self", "kwargs"]]
        own_params, other_params = params(cls), params(other_class)
        unknown = []
        for k in kwargs:
            if k in other_params:
                logging.warning("argument %s is ignored for %s"%(k, cls.__name__))
            else:
                unknown.append(k)
        if len(unknown) > 0:
            suggestions = {k: difflib.get_close_matches(k, own_params, n=3, cutoff=0.5) for k in unknown}
            message = ", ".join(k + (" (did you mean %s?)"%" or ".join(s) if s else "") for k, s in suggestions.items())
            raise TypeError("%s got unexpected arguments : %s"%(cls.__name__, message))

    @classmethod
    def _preset_values_from_kwargs(cls, kwargs) -> Dict[str, Any]:
        return {value: kwargs[arg] for arg, value in cls._preset_values.items() if arg in kwargs}

    @classmethod
    def _init_from_preset(cls, preset, preset_overrides, *args, **kwargs):
        from ..presets import preset_env
        with preset_env(preset, overrides=preset_overrides, values=cls._preset_values_from_kwargs(kwargs)):
            return type.__call__(cls, *args, **kwargs)

    @staticmethod
    def dataset_class_from_path(path: str | Path | List[str | Path]) -> type:
        from .simple import PreprocessedAudioDataset
        from .online import OnlineAudioDataset
        if path is None:
            raise ValueError("AudioDataset requires a path.")
        if isinstance(path, (list, tuple)):
            if any(is_preprocessed_path(p) for p in path):
                raise ValueError("Several pre-processed datasets cannot be loaded at once ; please use CombinedAudioDataset.")
            return OnlineAudioDataset
        if is_preprocessed_path(path):
            return PreprocessedAudioDataset
        return OnlineAudioDataset

    def _init_common(self,
                     transforms: TransformType = None,
                     output_pattern: str = "waveform",
                     output_type: str = "torch",
                     channels: int | None = 1,
                     subindices = None,
                     max_samples: int | None = None) -> None:
        self._transforms = transforms
        self._output_pattern = output_pattern
        self._output_type = output_type
        self._n_channels = channels
        self._subindices = subindices
        self._max_samples = max_samples
        self._parent = None
        self._partitions = {}

    # abstract methods

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def _fragment_from_index(self, index):
        """returns the AudioFragment corresponding to index."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def features(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def chunk_length(self) -> int | None:
        """chunk length (in samples) of dataset items."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def files(self) -> List[Dict[str, Any]]:
        """audio files of the dataset, as dicts with (at least) audio_path, duration, sr and channels keys. File indices 
        of load_audio and get_fragment refer to this list."""
        raise NotImplementedError()

    @abstractmethod
    def load_audio(self, file_idx: int, start: float = 0., length: int | float | None = None, pad_mode = None) -> torch.Tensor:
        """loads audio of a given file at the sampling rate of the dataset, as a (channels, samples) tensor.

        Args:
            file_idx (int): file index (in self.files).
            start (float, optional): start position (seconds). Defaults to 0.
            length (int | float | None, optional): length, in samples (int) or seconds (float). None for the whole file. Defaults to None.
            pad_mode (PadMode | str | None, optional): padding mode if the requested audio exceeds the file. Defaults to None.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_fragment(self, file_idx: int, start: float = 0., length: int | float | None = None, features = None, **kwargs):
        """loads audio of a given file (see load_audio) as a fragment, with features computed on the loaded audio 
        (None for dataset features)."""
        raise NotImplementedError()

    @abstractmethod
    def split(self, partitions, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def load_partition(self, name: str | None, check: bool = False):
        raise NotImplementedError()

    def configure(self, **kwargs) -> "AudioDataset":
        """returns a dataset configured with given loading parameters (e.g. chunk_length)."""
        if len(kwargs) > 0:
            raise TypeError("%s got unexpected loading arguments : %s"%(type(self).__name__, list(kwargs.keys())))
        return self

    @staticmethod
    def _assign_by_ratio(groups: List[Any], weights: List[float], ratios: Dict[str, float]) -> Dict[str, List[Any]]:
        """assigns ordered groups to partitions such that partition weights follow ratios ; each group is 
        assigned to the partition containing its cumulative weight midpoint."""
        total = sum(weights)
        assignment = {k: [] for k in ratios}
        bounds, current = [], 0.
        for k, r in ratios.items():
            current += r
            bounds.append((current, k))
        cum = 0.
        for g, w in zip(groups, weights):
            midpoint = (cum + w / 2) / total if total > 0 else 0.
            cum += w
            for bound, k in bounds:
                if midpoint < bound:
                    assignment[k].append(g)
                    break
        return assignment

    @staticmethod
    def _assign_balanced(groups: List[Any], weights: List[float], ratios: Dict[str, float], tolerance: float = 0.1) -> Dict[str, List[Any]]:
        """assigns groups greedily (heaviest first) to the partition with the largest relative deficit. If ratios 
        sum below 1, the remaining ratio is an implicit partition whose groups are left out."""
        total = sum(weights)
        targets = {k: r * total for k, r in ratios.items()}
        leftover = 1. - sum(ratios.values())
        if leftover > 1e-6:
            targets[None] = leftover * total
        current = {k: 0. for k in targets}
        assignment = {k: [] for k in targets}
        for w, g in sorted(zip(weights, groups), key=lambda x: -x[0]):
            k = max([k for k in targets if targets[k] > 0], key=lambda k: (targets[k] - current[k]) / targets[k])
            assignment[k].append(g)
            current[k] += w
        for k in ratios:
            if targets[k] > 0 and abs(current[k] - targets[k]) > tolerance * targets[k]:
                logging.warning("partition %s : weight %.2f for target %.2f (tolerance %.2f)"%(k, current[k], targets[k], tolerance))
        assignment.pop(None, None)
        return assignment

    # common methods

    @staticmethod
    def _build_fragment(fragment_class, wav, sr, audio_path, start, file_length, features, output_type="torch", bformat="float32", metadata=None):
        """builds a fragment from loaded audio, and computes features on it."""
        fragment = fragment_class(
            audio_path = audio_path,
            audio = wav,
            start_pos = start,
            length = wav.shape[-1] / sr,
            file_length = file_length,
            bformat = bformat,
            sr = sr,
            # features are computed on numpy outputs, as in LMDBWriter
            output_type = "numpy",
            **{k: v for k, v in (metadata or {}).items() if v is not None}
        )
        for f in features:
            f.extract(fragment=fragment)
        fragment.output_type = output_type
        return fragment

    def __getitem__(self, index):
        return self.get(index)

    def get(self, index, output_pattern=None, transforms=None):
        """returns the item at index. Slices and sequences of indices return a batch of items (collated as in 
        torch.utils.data.DataLoader, or as a list if items cannot be collated)."""
        if is_multi_index(index):
            indices = multi_index_to_list(index, len(self))
            return collate_items([self.get(i, output_pattern=output_pattern, transforms=transforms) for i in indices])
        if hasattr(index, "item") and not isinstance(index, (str, bytes)):
            # 0-d tensors / numpy integers
            index = index.item()
        output_pattern = output_pattern or self.output_pattern
        transforms = transforms or self.transforms
        fg = self._fragment_from_index(index)
        outs = _outs_from_pattern(fg, output_pattern)
        outs = _transform_outputs(outs, transforms, self._n_channels)
        return outs

    def get_sampler(self, replacement=False, max_samples=None):
        max_samples = max_samples or self._max_samples or len(self)
        return torch.utils.data.RandomSampler(self, replacement=replacement, num_samples = max_samples)

    def get_loader(self, batch_size: int = 1, shuffle: bool = True, replacement: bool = False, max_samples: int | None = None, **kwargs) -> torch.utils.data.DataLoader:
        """returns a DataLoader for the dataset. Keyword arguments of torch.utils.data.DataLoader (num_workers,
        drop_last, ...) are given to the DataLoader ; other ones (chunk_length, ...) are given to self.configure."""
        loader_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in _DATALOADER_ARGS}
        dataset = self.configure(**kwargs)
        sampler = dataset.get_sampler(replacement=replacement, max_samples=max_samples) if shuffle else None
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, **loader_kwargs)

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, obj):
        assert issubclass(type(obj), type(self)), "parent of a dataset must be a subclass of %s"%(type(self))
        self._parent = obj

    @property
    def is_partition(self):
        return self._parent is not None

    @property
    def output_pattern(self):
        return self._output_pattern

    @output_pattern.setter
    def output_pattern(self, pattern):
        assert isinstance(pattern, str)
        self._output_pattern = pattern

    @property
    def transforms(self):
        return self._transforms

    @transforms.setter
    def transforms(self, transforms):
        self._transforms = transforms


__all__ = ['AudioDataset', 'is_preprocessed_path', 'TransformType']
