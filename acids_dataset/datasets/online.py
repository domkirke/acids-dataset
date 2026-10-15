import os
import json
import copy
import math
import bisect
import random
import statistics
import gin
import torch, torchaudio
from absl import logging
from pathlib import Path
from types import EllipsisType
from typing import List, Dict, Any, Iterable, Literal
from ..utils import checklist, audio_info, load_audio_range, pad, PadMode, parse_features, feature_from_gin_config
from ..writers.utils import audio_paths_from_dir
from ..features import AcidsDatasetFeature, append_meta_regexp
from .. import get_fragment_class
from .base import AudioDataset, TransformType, chunk_length_to_samples


BatchByType = Literal["files", "dataset", "auto"]


@gin.configurable(module="datasets")
class OnlineAudioDataset(AudioDataset):
    """AudioDataset that parses audio files on the fly instead of reading a pre-processed database.

    At initialization, target audio files are scanned for header information (length, sampling rate,
    channels, codec) and path-based metadata (meta_regexp). This information is cached in an index
    file (".ad" by default, in each dataset folder) so that further instantiations only re-scan
    new or modified files. Features are computed dynamically when audio is loaded.

    Items are chunks of chunk_length samples, whose distribution is set by batch_by:
        - "files": one item per file, whose position is sampled randomly in the file. Files shorter than 
            chunk_length are padded with pad_mode.
        - "dataset": each file is split in max(1, round(duration / chunk_length)) slots, such that the 
            dataset length is approximately total_duration / chunk_length. Each item is sampled randomly 
            within its slot, such that every position of the dataset may be sampled while items 
            never overlap two files.
        - "auto": "dataset" if median file duration is greater than auto_ratio * chunk_length 
            (a few long files), "files" otherwise (a lot of short files).
    """
    index_name = ".ad"
    index_version = 1
    batch_by_modes = ["files", "dataset", "auto"]
    _preset_values = {'sr': 'SAMPLE_RATE', 'channels': 'CHANNELS', 'chunk_length': 'CHUNK_LENGTH'}
    # context (in original samples) loaded around chunks before resampling
    _resampling_context = 256

    @classmethod
    def _init_from_preset(cls, preset, preset_overrides, *args, **kwargs):
        from ..presets import preset_env
        with preset_env(preset, overrides=preset_overrides, values=cls._preset_values_from_kwargs(kwargs)):
            # features of the preset are completed with given features, as in preprocess_dataset
            kwargs['features'] = parse_features() + cls._parse_feature_list(kwargs.get('features'))
            return type.__call__(cls, *args, **kwargs)

    @staticmethod
    def _feature_items(features) -> List[tuple]:
        """returns features as (name, feature) items ; names are given by dict keys, and None otherwise."""
        if isinstance(features, dict):
            return list(features.items())
        return [(None, f) for f in checklist(features or [])]

    @classmethod
    def _parse_feature_list(cls, features) -> List[AcidsDatasetFeature]:
        """features can be given as a list or a {name: feature} dict of feature objects or feature config names 
        (e.g. "loudness"), that are parsed in the current gin config. Dict keys set feature names."""
        from ..presets import quiet_gin_search
        parsed_features = []
        with quiet_gin_search():
            for name, f in cls._feature_items(features):
                parsed = feature_from_gin_config(f) if isinstance(f, (str, Path)) else [f]
                if name is not None:
                    if len(parsed) != 1:
                        raise ValueError("feature %s gives several features, and cannot be renamed"%name)
                    parsed[0].feature_name = name
                parsed_features.extend(parsed)
        return parsed_features

    def __init__(self,
                 path: str | Path | List[str | Path],
                 transforms: TransformType = None,
                 output_pattern: str = 'waveform',
                 *,
                 output_type: str = "torch",
                 channels: int | None = 1,
                 sr: int | None = None,
                 chunk_length: int | float | None = None,
                 batch_by: BatchByType = "auto",
                 pad_mode: PadMode | str = PadMode.ZERO,
                 auto_ratio: float = 2.,
                 index_dir: str | Path | None = None,
                 partition_dir: str | Path | None = None,
                 features: List[AcidsDatasetFeature] | None = None,
                 meta_regexp: List[str] = [],
                 filters: List[str] = [],
                 exclude: List[str] = [],
                 valid_exts: List[str] | None = None,
                 bformat: str = "float32",
                 fragment_class: str = "AcidsFragment",
                 rescan: bool = False,
                 write_index: bool = True,
                 max_samples: int | None = None,
                 **kwargs) -> None:
        """
        Args:
            path (str | Path | List[str | Path]): dataset folder(s).
            sr (int | None, optional): target sampling rate, None to keep original sampling rates. Defaults to None.
            channels (int | None, optional): target number of channels, None to keep original channels. Defaults to 1.
            chunk_length (int | float | None, optional): item length, in samples (int) or seconds (float). Can also be 
                given to configure / get_loader. Defaults to None.
            batch_by (str, optional): item sampling strategy ("files", "dataset", or "auto"). Defaults to "auto".
            pad_mode (PadMode | str, optional): padding mode for chunks exceeding files. Defaults to PadMode.ZERO.
            auto_ratio (float, optional): median duration / chunk_length ratio above which "auto" selects "dataset". Defaults to 2.
            index_dir (str | Path | None, optional): folder where index files are written. If None, index is written
                as a ".ad" file in each dataset folder ; otherwise, as "<folder_name>.ad" in index_dir. Defaults to None.
            partition_dir (str | Path | None, optional): folder where partitions are written. If None, partitions are 
                written in "partitions" of index_dir if given, or in ".ad_partitions" of the first dataset folder. Defaults to None.
            features (List[AcidsDatasetFeature | str] | Dict[str, AcidsDatasetFeature | str] | None, optional): features computed 
                dynamically on loaded audio, given as objects or feature config names (e.g. "loudness") ; dict keys set 
                feature names. Defaults to None.
            meta_regexp (List[str], optional): glob-like patterns for path-based metadata (e.g. "**/{{instrument}}_*.wav"). Defaults to [].
            filters (List[str], optional): glob-like patterns to filter audio paths. Defaults to [].
            exclude (List[str], optional): glob-like patterns to exclude audio paths. Defaults to [].
            valid_exts (List[str] | None, optional): valid audio extensions, None for default ones. Defaults to None.
            bformat (str, optional): audio format of loaded fragments. Defaults to "float32".
            rescan (bool, optional): ignore existing index files and re-scan every file. Defaults to False.
            write_index (bool, optional): write index files after scanning. Defaults to True.
        """
        self._paths = [Path(p).resolve() for p in checklist(path)]
        for p in self._paths:
            if not p.exists(): raise FileNotFoundError("%s folder does not exist."%p)
        # audio paths and index files are named after dataset folders
        folder_names = [p.name for p in self._paths]
        if len(set(folder_names)) != len(folder_names):
            raise ValueError("dataset folders must have distinct names, got %s"%self._paths)
        self._init_common(transforms=transforms, output_pattern=output_pattern, output_type=output_type, 
                          channels=channels, max_samples=max_samples)
        self._index_dir = None if index_dir is None else Path(index_dir).resolve()
        self._partition_dir = None if partition_dir is None else Path(partition_dir).resolve()
        self._filters = checklist(filters)
        self._exclude = checklist(exclude)
        self._valid_exts = valid_exts
        self._meta_regexp = checklist(meta_regexp)
        self._regexp_features = append_meta_regexp([], self._meta_regexp)
        from .simple import PreprocessedAudioDataset
        self._check_kwargs(kwargs, PreprocessedAudioDataset)
        if any(isinstance(f, (str, Path)) for _, f in self._feature_items(features)):
            # feature configs are parsed with the default preset, with dataset values
            from ..presets import preset_env
            with preset_env("default", values={'SAMPLE_RATE': sr, 'CHANNELS': channels, 'CHUNK_LENGTH': chunk_length}):
                features = self._parse_feature_list(features)
        else:
            features = self._parse_feature_list(features)
        self._features = list(features)
        self._sr = sr
        self._bformat = bformat
        self._fragment_class = get_fragment_class(fragment_class)
        self._db_path = self._paths[0]
        self._files = []
        self._invalid_files = []
        for p in self._paths:
            self._scan(p, rescan=rescan, write_index=write_index)
        if len(self._files) == 0:
            raise RuntimeError("no valid audio files found in %s"%self.path)
        self._auto_ratio = auto_ratio
        self._chunk_length = None
        self._batch_by = None
        self._pad_mode = None
        self._set_loading_params(chunk_length=chunk_length, batch_by=batch_by, pad_mode=pad_mode)
        torch.utils.data.Dataset.__init__(self)

    # index handling

    @classmethod
    def index_path_for(cls, dataset_path: str | Path, index_dir: str | Path | None = None) -> Path:
        """returns the index file location for a given dataset folder, without instantiating the dataset."""
        dataset_path = Path(dataset_path).resolve()
        if index_dir is None:
            return dataset_path / cls.index_name
        return Path(index_dir).resolve() / f"{dataset_path.name}{cls.index_name}"

    def index_path(self, dataset_path: str | Path) -> Path:
        """returns the index file location for a given dataset folder."""
        return self.index_path_for(dataset_path, self._index_dir)

    def _read_index(self, dataset_path: Path) -> Dict[str, Any]:
        index_path = self.index_path(dataset_path)
        if not index_path.exists():
            return {}
        try:
            with open(index_path, "r") as f:
                index = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logging.warning("could not read index %s (%s) ; re-scanning"%(index_path, e))
            return {}
        if index.get('version') != self.index_version:
            return {}
        return index

    def _write_index(self, dataset_path: Path, index: Dict[str, Any]) -> None:
        index_path = self.index_path(dataset_path)
        os.makedirs(index_path.parent, exist_ok=True)
        tmp_path = index_path.with_name(index_path.name + ".tmp")
        with open(tmp_path, "w+") as f:
            json.dump(index, f, indent=1)
        os.replace(tmp_path, index_path)

    @staticmethod
    def _file_stamp(file_path: Path) -> Dict[str, Any]:
        stat = os.stat(file_path)
        return {'size': stat.st_size, 'mtime': stat.st_mtime}

    def _parse_meta_regexp(self, audio_path: str) -> Dict[str, Any]:
        return {f.feature_name: f._extract_id_from_path(audio_path) for f in self._regexp_features}

    def _scan(self, dataset_path: Path, rescan: bool = False, write_index: bool = True) -> None:
        """scans audio files of dataset_path, using the index as a cache for unmodified files."""
        index = {} if rescan else self._read_index(dataset_path)
        cached_files = index.get('files', {})
        cached_invalid = index.get('invalid', {})
        audio_files = audio_paths_from_dir(dataset_path, valid_exts=self._valid_exts, flt=self._filters, exclude=self._exclude)
        n_scanned = 0
        for audio_file in sorted(audio_files):
            audio_file = Path(audio_file)
            rel_path = str(audio_file.relative_to(dataset_path))
            stamp = self._file_stamp(audio_file)
            entry = cached_files.get(rel_path)
            if entry is None or any(entry.get(k) != v for k, v in stamp.items()):
                if cached_invalid.get(rel_path) == stamp:
                    continue
                try:
                    entry = {**audio_info(audio_file), **stamp}
                except Exception as e:
                    logging.warning("could not read file %s : %s"%(audio_file, e))
                    cached_files.pop(rel_path, None)
                    cached_invalid[rel_path] = stamp
                    continue
                cached_invalid.pop(rel_path, None)
                n_scanned += 1
            # path-based metadata use the same convention as LMDBWriter (path relative to dataset parent)
            entry['metadata'] = self._parse_meta_regexp(str(Path(dataset_path.name) / rel_path))
            cached_files[rel_path] = entry
            self._files.append({'path': str(audio_file), 'audio_path': str(Path(dataset_path.name) / rel_path), **entry})
        self._invalid_files.extend(str(dataset_path / p) for p in cached_invalid)

        # remove deleted files from cache
        for rel_path in list(cached_files.keys()):
            if not (dataset_path / rel_path).exists(): del cached_files[rel_path]
        for rel_path in list(cached_invalid.keys()):
            if not (dataset_path / rel_path).exists(): del cached_invalid[rel_path]

        if write_index:
            self._write_index(dataset_path, {
                'version': self.index_version,
                'dataset_path': str(dataset_path),
                'meta_regexp': self._meta_regexp,
                'files': cached_files,
                'invalid': cached_invalid
            })
        logging.info("%s : %d files (%d scanned, %d invalid)"%(dataset_path, len(audio_files), n_scanned, len(cached_invalid)))

    # properties

    @property
    def path(self):
        return [str(p) for p in self._paths]

    @property
    def files(self) -> List[Dict[str, Any]]:
        """list of every scanned audio file, with header information and path-based metadata. File indices 
        (used in get_fragment, chunk_position, partitions...) refer to this list."""
        return copy.deepcopy(self._files)

    @property
    def file_indices(self) -> List[int]:
        """indices of files used by the dataset (all files, or a subset for partitions)."""
        return list(range(len(self._files))) if self._subindices is None else self._subindices

    @property
    def invalid_files(self) -> List[str]:
        return list(self._invalid_files)

    @property
    def features(self) -> List[AcidsDatasetFeature]:
        return list(self._features)

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            'n_files': len(self.file_indices),
            'n_seconds': sum(self._files[i]['duration'] for i in self.file_indices),
            'sr': self.sr,
            'channels': self._n_channels,
            'chunk_length': self.chunk_length,
            'batch_by': self.batch_by,
            'pad_mode': self._pad_mode.name.lower(),
            'features': {f.feature_name: str(f) for f in self._features},
            'meta_regexp': self._meta_regexp,
            'filters': self._filters,
            'exclude': self._exclude,
        }

    @property
    def loader(self):
        raise NotImplementedError("OnlineAudioDataset has no database loader.")

    @property
    def feature_hash(self):
        raise NotImplementedError()

    @property
    def keys(self):
        raise NotImplementedError()

    # loading parameters

    @property
    def sr(self) -> int | None:
        """sampling rate of dataset items ; None if not set and files have different sampling rates."""
        if self._sr is not None:
            return self._sr
        srs = set(f['sr'] for f in self._files)
        return srs.pop() if len(srs) == 1 else None

    @property
    def chunk_length(self) -> int | None:
        """chunk length of dataset items, in samples."""
        if self._chunk_length is None:
            return None
        if isinstance(self._chunk_length, int):
            return self._chunk_length
        if self.sr is None:
            raise ValueError("files have different sampling rates ; please provide sr to dataset")
        return chunk_length_to_samples(self._chunk_length, self.sr)

    @property
    def batch_by(self) -> str | None:
        """resolved item sampling strategy ("files" or "dataset")."""
        return self._batch_by

    @property
    def pad_mode(self) -> PadMode:
        return self._pad_mode

    def _resolve_batch_by(self, batch_by: str) -> str:
        if batch_by not in self.batch_by_modes:
            raise ValueError("batch_by must be in %s, got %s"%(self.batch_by_modes, batch_by))
        if batch_by != "auto":
            return batch_by
        # resolved on every file, such that partitions share the same mode
        median_duration = statistics.median(f['duration'] for f in self._files)
        return "dataset" if median_duration >= self._auto_ratio * self.chunk_length / self.sr else "files"

    def _set_loading_params(self, chunk_length=None, batch_by=None, pad_mode=None) -> None:
        if pad_mode is not None:
            if isinstance(pad_mode, str): pad_mode = getattr(PadMode, pad_mode.upper())
            if pad_mode == PadMode.DISCARD:
                raise ValueError("PadMode.DISCARD is not available for OnlineAudioDataset")
            self._pad_mode = pad_mode
        if chunk_length is not None:
            if chunk_length <= 0: raise ValueError("chunk_length must be positive")
            self._chunk_length = chunk_length
        if batch_by is not None:
            self._batch_by_arg = batch_by
        if self._chunk_length is None:
            # items cannot be defined yet
            self._batch_by = None
            self._cum_chunks = None
            return
        if self.sr is None:
            raise ValueError("files have different sampling rates ; please provide sr to dataset")
        self._batch_by = self._resolve_batch_by(self._batch_by_arg)
        self._build_chunk_map()

    def _n_chunks_for_file(self, file) -> int:
        if self._batch_by == "files":
            return 1
        return max(1, round(file['duration'] * self.sr / self.chunk_length))

    def _build_chunk_map(self) -> None:
        self._cum_chunks = []
        n_chunks = 0
        for file_idx in self.file_indices:
            n_chunks += self._n_chunks_for_file(self._files[file_idx])
            self._cum_chunks.append(n_chunks)

    def configure(self, chunk_length: int | float | None = None, batch_by: BatchByType | None = None, pad_mode: PadMode | str | None = None, **kwargs) -> "OnlineAudioDataset":
        """returns a copy of the dataset with given loading parameters."""
        super().configure(**kwargs)
        dataset = copy.copy(self)
        dataset._set_loading_params(chunk_length=chunk_length, batch_by=batch_by, pad_mode=pad_mode)
        return dataset

    def _check_loading_params(self):
        if self._cum_chunks is None:
            raise ValueError("chunk_length must be given to OnlineAudioDataset (in constructor, configure, or get_loader)")

    def __len__(self):
        self._check_loading_params()
        return self._cum_chunks[-1] if len(self._cum_chunks) > 0 else 0

    def chunk_position(self, index: int, random_offset: bool = True):
        """returns (file index, start position in seconds) of a given item. If random_offset is False, the 
        chunk is taken at the beginning of its slot."""
        self._check_loading_params()
        if index < 0: index += len(self)
        if not (0 <= index < len(self)):
            raise IndexError("index %d out of range for dataset of length %d"%(index, len(self)))
        position = bisect.bisect_right(self._cum_chunks, index)
        file_idx = self.file_indices[position]
        n_chunks = self._n_chunks_for_file(self._files[file_idx])
        chunk_idx = index - (self._cum_chunks[position - 1] if position > 0 else 0)
        duration = self._files[file_idx]['duration']
        chunk_duration = self.chunk_length / self.sr
        # each chunk is sampled in its slot, without exceeding the file if possible
        slot_length = duration / n_chunks
        max_start = max(0., duration - chunk_duration)
        slot_start = min(chunk_idx * slot_length, max_start)
        slot_end = min(slot_start + max(0., slot_length - chunk_duration), max_start)
        start = random.uniform(slot_start, slot_end) if random_offset else slot_start
        return file_idx, start

    def _fragment_from_index(self, index):
        file_idx, start = self.chunk_position(index)
        return self.get_fragment(file_idx, start=start, length=self.chunk_length)

    # loading

    def _adapt_channels(self, wav: torch.Tensor) -> torch.Tensor:
        channels = self._n_channels
        if channels is None:
            return wav
        if wav.shape[0] > channels:
            wav = wav[:channels]
        elif wav.shape[0] < channels:
            wav = wav[torch.arange(channels) % wav.shape[0]]
        return wav

    def load_audio(self, file_idx: int, start: float = 0., length: int | float | None = None, pad_mode: PadMode | str | None = None) -> torch.Tensor:
        """loads a chunk of a given file, resampled to target sr and adapted to target channels.

        Args:
            file_idx (int): file index (in self.files).
            start (float, optional): start position (seconds). Defaults to 0.
            length (int | float | None, optional): chunk length, in samples (int, at target sr) or seconds (float). 
                None for the whole file. Defaults to None.
            pad_mode (PadMode | str | None, optional): padding mode if the chunk exceeds the file ; None for dataset's pad_mode. Defaults to None.
        """
        file = self._files[file_idx]
        orig_sr = file['sr']
        target_sr = self._sr or orig_sr
        frame_offset = int(round(start * orig_sr))
        target_length = None if length is None else chunk_length_to_samples(length, target_sr)
        num_frames = -1 if length is None else int(math.ceil(target_length * orig_sr / target_sr))
        # resampled chunks are loaded with surrounding context, such that they match chunks of resampled files (as in 
        # pre-processing). Left context is a multiple of the resampling period, to be trimmed exactly after resampling.
        left_context, right_context = 0, 0
        if orig_sr != target_sr:
            period = orig_sr // math.gcd(orig_sr, target_sr)
            context = int(math.ceil(self._resampling_context / period)) * period
            left_context = min(context, (frame_offset // period) * period)
            right_context = context if num_frames >= 0 else 0
        wav, orig_sr = load_audio_range(file['path'], frame_offset=frame_offset - left_context, 
                                        num_frames=num_frames + left_context + right_context if num_frames >= 0 else -1, 
                                        backend=file.get('backend'))
        if orig_sr != target_sr and wav.shape[-1] > 0:
            wav = torchaudio.functional.resample(wav, orig_sr, target_sr)
            wav = wav[..., left_context * target_sr // orig_sr:]
        wav = self._adapt_channels(wav)
        if target_length is not None:
            if wav.shape[-1] > target_length:
                wav = wav[..., :target_length]
            elif wav.shape[-1] < target_length:
                pad_mode = pad_mode or self._pad_mode
                if isinstance(pad_mode, str): pad_mode = getattr(PadMode, pad_mode.upper())
                wav = pad(wav, target_length, pad_mode)
        return wav

    def get_fragment(self, file_idx: int, start: float = 0., length: float | None = None, features: Iterable[AcidsDatasetFeature] | None = None, **kwargs):
        """loads a chunk of a given file as a fragment, with path-based metadata and dynamically computed features.

        Args:
            file_idx (int): file index (in self.files).
            start (float, optional): start position (seconds). Defaults to 0.
            length (int | float | None, optional): chunk length, in samples (int, at target sr) or seconds (float). 
                None for the whole file. Defaults to None.
            features (Iterable[AcidsDatasetFeature] | None, optional): features to compute ; None for dataset features. Defaults to None.
        """
        file = self._files[file_idx]
        wav = self.load_audio(file_idx, start=start, length=length, **kwargs)
        return self._build_fragment(self._fragment_class, wav, self._sr or file['sr'], file['audio_path'], start, file['duration'], 
                                    self._features if features is None else features, output_type=self._output_type, 
                                    bformat=self._bformat, metadata=file['metadata'])

    # feature search
    # path-based features (original_path and meta_regexp) are known for every file without loading audio ; 
    # searching and partitioning are thus performed on files, such that chunks of a given file always 
    # belong to the same partition.

    @property
    def path_features(self) -> List[str]:
        """names of features available for search and partitioning."""
        return ['original_path'] + [f.feature_name for f in self._regexp_features]

    def _file_feature_value(self, file_idx: int, feature_name: str):
        file = self._files[file_idx]
        if feature_name == "original_path":
            return file['audio_path']
        value = file['metadata'].get(feature_name)
        return tuple(value) if isinstance(value, list) else value

    def _check_path_features(self, features: Iterable[str]) -> None:
        for f in features:
            if f not in self.path_features:
                raise ValueError("feature %s cannot be used for search or partitioning ; available features : %s"%(f, self.path_features))

    @property
    def feature_hash(self) -> Dict[str, Dict[Any, List[int]]]:
        """{feature_name: {value: [file indices]}} for path-based features of dataset files."""
        feature_hash = {f: {} for f in self.path_features}
        for i in self.file_indices:
            for f in feature_hash:
                value = self._file_feature_value(i, f)
                if value is not None:
                    feature_hash[f].setdefault(value, []).append(i)
        return feature_hash

    def index_for_features(self, **kwargs) -> Dict[tuple, set]:
        """returns {(value_1, ..., value_n): set of file indices} for given values of path-based features 
        (e.g. index_for_features(instrument=["piano", "guitar"], idx=0))."""
        self._check_path_features(kwargs.keys())
        feature_hash = self.feature_hash
        idx_dict = {(): set(self.file_indices)}
        for feature_name, feature_value in kwargs.items():
            new_idx_dict = {}
            for v in checklist(feature_value):
                # regexp values are strings
                hash_tmp = feature_hash[feature_name].get(v, feature_hash[feature_name].get(str(v)))
                if not hash_tmp: 
                    raise IndexError("could not find value %s for feature %s"%(v, feature_name))
                for kk, vv in idx_dict.items():
                    new_idx_dict[kk + (v,)] = vv.intersection(hash_tmp)
            idx_dict = new_idx_dict
        return idx_dict

    def item_indices_for_files(self, file_indices: Iterable[int]) -> List[int]:
        """returns dataset item indices corresponding to given files."""
        self._check_loading_params()
        positions = {f: p for p, f in enumerate(self.file_indices)}
        items = []
        for f in file_indices:
            if f not in positions: 
                raise IndexError("file %d not in dataset"%f)
            p = positions[f]
            items.extend(range(self._cum_chunks[p - 1] if p > 0 else 0, self._cum_chunks[p]))
        return items

    def query_from_features(self, n: int | EllipsisType, _random: bool = True, **kwargs) -> Dict[tuple, List[Any]]:
        """returns {(value_1, ..., value_n): [items]} for given values of path-based features. n is the number of 
        items per entry (Ellipsis for all items) ; if _random, items are randomly chosen (with replacement)."""
        assert isinstance(n, (int, EllipsisType)), "n must be either an int or Ellipsis"
        idx_dict = self.index_for_features(**kwargs)
        data_dict = {}
        for k, file_indices in idx_dict.items():
            items = self.item_indices_for_files(sorted(file_indices))
            if n is not Ellipsis and len(items) > 0:
                items = random.choices(items, k=n) if _random else items[:n]
            data_dict[k] = [self[i] for i in items]
        return data_dict

    # partitioning

    _default_partition_name = "partitions"

    @property
    def partition_dir(self) -> Path:
        if self._partition_dir is not None:
            return self._partition_dir
        if self._index_dir is not None:
            return self._index_dir / "partitions"
        return self._paths[0] / ".ad_partitions"

    def _partition_path(self, name: str | bool | None) -> Path:
        if name is None or name is True:
            name = self._default_partition_name
        if os.path.splitext(name)[1] != ".json": 
            name += ".json"
        return self.partition_dir / name

    def get_subdataset(self, subindices: Iterable[int], check: bool = False) -> "OnlineAudioDataset":
        """returns a dataset restricted to given file indices."""
        subindices = sorted(set(int(i) for i in subindices))
        if check:
            for i in subindices:
                if not (0 <= i < len(self._files)): 
                    raise IndexError("file index %d out of range"%i)
        subdataset = copy.copy(self)
        subdataset._subindices = subindices
        subdataset.parent = self
        if subdataset._cum_chunks is not None:
            subdataset._build_chunk_map()
        return subdataset

    def _file_weights(self, file_indices: List[int]) -> List[float]:
        # number of items if loading parameters are set, durations otherwise
        if self._cum_chunks is not None:
            return [float(self._n_chunks_for_file(self._files[i])) for i in file_indices]
        return [self._files[i]['duration'] for i in file_indices]

    def _split_without_feature(self, partitions: Dict[str, float], rng=random) -> Dict[str, List[int]]:
        files = rng.sample(self.file_indices, k=len(self.file_indices))
        return self._assign_by_ratio(files, self._file_weights(files), partitions)

    def _split_with_features(self, partitions: Dict[str, float], features: List[str], tolerance: float = 0.1, balance_cardinality: bool = False, rng=random) -> Dict[str, List[int]]:
        self._check_path_features(features)
        groups = {}
        n_missing = 0
        for i in self.file_indices:
            label = tuple(self._file_feature_value(i, f) for f in features)
            if None in label:
                n_missing += 1
                continue
            groups.setdefault(label, []).append(i)
        if n_missing > 0:
            logging.warning("%d files have no value for features %s, and are not partitioned"%(n_missing, features))
        labels = rng.sample(list(groups.keys()), k=len(groups))
        if balance_cardinality:
            weights = [sum(self._file_weights(groups[l])) for l in labels]
            assignment = self._assign_balanced(labels, weights, partitions, tolerance=tolerance)
        else:
            assignment = self._assign_by_ratio(labels, [1.] * len(labels), partitions)
        return {k: sum([groups[l] for l in v], []) for k, v in assignment.items()}

    def _split_from_feature(self, partition_key: str, partitions: Iterable[str]) -> Dict[str, List[int]]:
        feature_hash = self.feature_hash
        assert partition_key in feature_hash, "key %s not found in feature_hash"%(partition_key)
        for p in partitions:
            assert p in feature_hash[partition_key], "partition %s not found in %s"%(p, partition_key)
        return {p: feature_hash[partition_key][p] for p in partitions}

    def _write_partition(self, partitions: Dict[str, "OnlineAudioDataset"], write: str | bool) -> None:
        partition_path = self._partition_path(write)
        os.makedirs(partition_path.parent, exist_ok=True)
        # files are referred with their audio_path, such that partitions are robust to dataset modifications
        with open(partition_path, "w+") as f:
            json.dump({p: [self._files[i]['audio_path'] for i in v.file_indices] for p, v in partitions.items()}, f, indent=1)

    def has_partition(self, name: str) -> bool:
        return self._partition_path(name).exists()

    def split(self, 
              partitions: Dict[str, float], 
              features: List[str] | None = None, 
              use_meta_if_available: str | bool | None = True,
              load_if_available: bool | str | None = False,
              balance_cardinality: bool = False, 
              tolerance: float = 0.1, 
              write: str | bool | None = None,
              seed: int | None = None) -> Dict[str, "OnlineAudioDataset"]:
        """splits dataset files in partitions.

        Args:
            partitions (Dict[str, float]): partition ratios (e.g. {"train": 0.8, "valid": 0.2}).
            features (List[str] | None, optional): path-based features (original_path, or meta_regexp features) used to group 
                files, such that each feature value belongs to a single partition. Defaults to None.
            use_meta_if_available (str | bool | None, optional): if a path-based feature with this name (or "partition" if True) 
                exists, its values are used as partitions. Defaults to True.
            load_if_available (bool | str | None, optional): loads the partition with this name (or default name if True) 
                if it exists. Defaults to False.
            balance_cardinality (bool, optional): balance partitions sizes rather than number of feature values. Defaults to False.
            tolerance (float, optional): tolerance for balanced partitions. Defaults to 0.1.
            write (str | bool | None, optional): writes partition with this name (or default name if True). Defaults to None.
            seed (int | None, optional): seed of random partitions ; partitions with the same seed are identical (e.g. across 
                processes of distributed training). If None, the global random state is used. Defaults to None.
        """
        assert not self.is_partition, "dataset is already a partition of an existing dataset."
        rng = random.Random(seed) if seed is not None else random
        if load_if_available:
            name = None if load_if_available is True else load_if_available
            if self.has_partition(name):
                if features is not None: logging.warning("found partition to load, but features was not None; feature-based partitioning ignored")
                return self.load_partition(name)

        file_partitions = None
        if use_meta_if_available:
            partition_key = "partition" if use_meta_if_available is True else use_meta_if_available
            if partition_key in self.path_features:
                try: 
                    file_partitions = self._split_from_feature(partition_key, partitions.keys())
                    if features is not None: logging.warning("found partition feature, but features was not None; feature-based partitioning ignored")
                except AssertionError as e:
                    logging.warning('Could not find partition using feature %s (%s). Creating new partition'%(partition_key, e))
        if file_partitions is None:
            if features is None:
                file_partitions = self._split_without_feature(partitions, rng=rng)
            else:
                file_partitions = self._split_with_features(partitions, checklist(features), tolerance=tolerance, balance_cardinality=balance_cardinality, rng=rng)

        partitions = {k: self.get_subdataset(v) for k, v in file_partitions.items()}
        if write: 
            self._write_partition(partitions, write)
        return partitions

    def load_partition(self, name: str | None = None, check: bool = False) -> Dict[str, "OnlineAudioDataset"]:
        partition_path = self._partition_path(name)
        if not partition_path.exists():
            raise FileNotFoundError("could not load partition %s"%partition_path)
        with open(partition_path, "r") as f:
            partition_paths = json.load(f)
        file_map = {f['audio_path']: i for i, f in enumerate(self._files)}
        partitions = {}
        for p, audio_paths in partition_paths.items():
            missing = [a for a in audio_paths if a not in file_map]
            if len(missing) > 0:
                if check: raise FileNotFoundError("partition %s : files %s not found in dataset"%(p, missing))
                logging.warning("partition %s : %d files not found in dataset"%(p, len(missing)))
            partitions[p] = self.get_subdataset([file_map[a] for a in audio_paths if a in file_map])
        return partitions


__all__ = ['OnlineAudioDataset']
