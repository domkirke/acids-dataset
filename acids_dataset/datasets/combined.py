from typing import List
import bisect
import math
import copy
from absl import logging
import torch
from functools import reduce
from .simple import AudioDataset
from ..utils import checklist

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

    def _check_feature_in_subdatasets(self, features):
        for f in checklist(features): 
            for i, d in enumerate(self._datasets):
                assert f in d.features or f == "original_path", "feature %s not present in dataset %d (path : %s)"%(f, i, d._db_path)

    def split(self, 
            partitions, 
            features = None,  
            **kwargs):
        assert self.parent is None, "dataset is already a partition"
        if features is not None: 
            self._check_feature_in_subdatasets(features)
        partitions_list = [
            d.split(partitions, **kwargs)
            for d in self._datasets
        ]
        partitions = {}
        for p in partitions_list[0].keys():
            partitions[p] = [d[p] for d in partitions_list]
        return {p: self.get_subdataset(partitions[p]) for p in partitions.keys()}

    def load_partition(self, name: str | None, check: bool = False): 
        partitions = {}
        for i, d in enumerate(self._datasets):
            dpart = d.load_partition(name, check=check)
            if i == 0: partitions.update({k: [None] * len(self._datasets) for k in dpart.keys()})
            for k in dpart.keys(): 
                partitions[k][i] = dpart[k]._subindices
        return {p: self.get_subdataset(partitions[p]) for p in partitions.keys()}
        
    



    