from typing import List
import bisect
import math
import copy
from absl import logging
import torch
from functools import reduce
from .base import AudioDataset
from ..utils import checklist
from .utils import is_multi_index, multi_index_to_list, collate_items

def cumsum(l): 
    return [sum(l[:i+1]) for i in range(len(l))]


class CombinedAudioDataset(torch.utils.data.Dataset):
    _fields_to_compare_ = ['_output_pattern', '_n_channels']
    def __init__(self, datasets: List[AudioDataset], weights: List[float] | None = None, max_samples: int | None = None, use_cache: bool = False):
        assert len(datasets) > 1, "CombinedAudioDataset must have at least 2 datasets as inputs"
        self._compare_datasets(datasets)
        if weights is None: 
            weights = [1 / len(datasets) ] * len(datasets)
        assert len(weights) == len(datasets), "datasets and weights must be the same size"
        self._weights = weights
        self._datasets = datasets
        self._dataset_map = cumsum([len(d) for d in self._datasets])
        self._output_pattern = self._datasets[0]._output_pattern
        self._n_channels = self._datasets[0]._n_channels
        self._parent = None
        self._subindices = None
        self._max_samples = max_samples
        self._partitions = {}
        if use_cache: logging.warning("use_cache is not implemented yet in CombinedAudioDataset")

    def _compare_datasets(self, datasets):
        for field in self._fields_to_compare_:
            assert len(set([getattr(d, field) for d in datasets])) == 1

    def __getitem__(self, index): 
        if is_multi_index(index):
            return collate_items([self[i] for i in multi_index_to_list(index, len(self))])
        if hasattr(index, "item"): index = index.item()
        if index < 0: index += len(self)
        if not (0 <= index < len(self)):
            raise IndexError("index %d out of range for dataset of length %d"%(index, len(self)))
        target_dataset = bisect.bisect_right(self._dataset_map, index)
        if self._subindices is None: 
            if target_dataset > 0: index -= self._dataset_map[target_dataset - 1]
        else:
            if target_dataset > 0: index -= self._dataset_map[target_dataset - 1]
            index = self._subindices[target_dataset][index]
        return self._datasets[target_dataset][index]

    def __len__(self): 
        if self._subindices is None:
            return sum([len(d) for d in self._datasets])
        else:
            return sum([len(s) for s in self._subindices])

    def get_sampler(self, valid: bool = False, replacement: bool = False):
        max_samples = len(self) if self._max_samples is None else self._max_samples
        weights = sum([[self._weights[i]] * len(self._datasets[i]) for i in range(len(self._datasets))], [])
        if valid:
            return torch.utils.data.WeightedRandomSampler(weights, max_samples, replacement=replacement)
        else: 
            return torch.utils.data.WeightedRandomSampler(weights, max_samples, replacement=replacement, generator=torch.Generator().manual_seed(42))

    @property 
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, obj):
        assert issubclass(type(obj), type(self)), "parent of a dataset must be a subclass of %s"%(type(self))
        self._parent = obj

    @property
    def keys(self) -> List[List[str]]: 
        return [d.keys for d in self._datasets]

    def get_subdataset(self, subindices, check=False):
        assert len(subindices) == len(self._datasets)
        subdataset = copy.copy(self)
        subdataset._subindices = subindices
        subdataset._dataset_map = cumsum([len(d) for d in subdataset._subindices])
        subdataset.parent = self
        if self._max_samples:
            subdataset._max_samples = int(self._max_samples * (len(subindices) / len(self)))
        return subdataset

    def _from_partitions(self, datasets: List[AudioDataset]) -> "CombinedAudioDataset":
        """returns a combined dataset of partitions of the combined datasets."""
        assert len(datasets) == len(self._datasets)
        subdataset = copy.copy(self)
        subdataset._datasets = list(datasets)
        subdataset._dataset_map = cumsum([len(d) for d in datasets])
        subdataset._subindices = None
        subdataset.parent = self
        if self._max_samples:
            subdataset._max_samples = int(self._max_samples * (len(subdataset) / len(self)))
        return subdataset

    def _check_feature_in_subdatasets(self, features):
        for f in checklist(features): 
            for i, d in enumerate(self._datasets):
                assert f in d.feature_hash, "feature %s not present in dataset %d (path : %s)"%(f, i, d.path)

    def split(self, 
            partitions, 
            features = None,  
            **kwargs):
        """splits each combined dataset with the same partitions (see AudioDataset.split), and returns combined datasets 
        of the corresponding partitions."""
        assert self.parent is None, "dataset is already a partition"
        if features is not None: 
            self._check_feature_in_subdatasets(features)
        partitions_list = [d.split(partitions, features=features, **kwargs) for d in self._datasets]
        return {p: self._from_partitions([d[p] for d in partitions_list]) for p in partitions_list[0].keys()}

    def load_partition(self, name: str | None, check: bool = False): 
        partitions_list = [d.load_partition(name, check=check) for d in self._datasets]
        return {p: self._from_partitions([d[p] for d in partitions_list]) for p in partitions_list[0].keys()}
