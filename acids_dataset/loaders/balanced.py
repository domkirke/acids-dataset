import gin.torch
import random
from itertools import product, batched
from typing import List
from torch.utils.data import Sampler, Dataset

def balanced_batches(data_map, 
                     batch_size, 
                     normalize_classes, 
                     keep_unbalanced = False,
                     ):
    min_amount = 0
    if normalize_classes:
        min_amount = min([len(v) for v in data_map.values()])
    has_finished = False
    _unused_items = {}
    keys = list(data_map.keys())
    while not has_finished:
        current_keys = []
        while len(current_keys) < batch_size:
            if len(keys) < batch_size: 
                has_finished = True
                break
            random_key_idx = random.randrange(len(keys))
            if len(data_map[keys[random_key_idx]]) <= min_amount:
                _unused_items[keys[random_key_idx]] = data_map[keys[random_key_idx]]
                del keys[random_key_idx]
            else:
                current_keys.append(keys[random_key_idx])
        data = []
        for current_key in current_keys:
            random_data_idx = random.randrange(len(data_map[current_key]))
            current_data = data_map[current_key].pop(random_data_idx)
            data.append(current_data)
        yield data
    
    if keep_unbalanced: 
        left_data = sum(list(_unused_items.values()), [])
        for b in batched(left_data, batch_size):
            yield b
        raise StopIteration
    else:
        raise NotImplementedError


@gin.configurable
class BalancedSampler(Sampler[List[int]]):
    def __init__(self, 
                 dataset: Dataset, 
                 batch_size: int, 
                 shuffle: bool = True, 
                 features: str | List[str] = "original_path", 
                 normalize_classes: bool = False,
                 pre_init: bool = False):
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._init_feature_hash(dataset, features)
        self.normalize_classes = normalize_classes
        if pre_init:
            self._init_batches(True)

    def _init_feature_hash(self, dataset, features):
        assert hasattr(dataset, "feature_hash")
        feature_hash = dataset.feature_hash
        missing_keys = set(list(feature_hash.keys())).difference(features)
        if len(missing_keys) > 0:
            raise ValueError('Could not find features %s into dataset feature hash.'%(missing_keys))
        self.feature_hash = {}
        for f in features: 
            self.feature_hash[f] = feature_hash[f]

    @staticmethod
    def make_data_map(feature_hash):
        data_map = dict()
        for datas in product([v.items() for v in feature_hash.values()]):
            key_tuple = tuple(d[0] for d in datas)
            value_tuple = tuple(d[1] for d in datas)
            value_set = set(value_tuple[0])
            for i in range(1, len(value_tuple)):
                value_set.intersection_update(value_tuple[i])
            data_map[key_tuple] = value_set

    def _init_batches(self, pre_init: bool):
        if len(self.feature) == 1:
            data_map = self.feature_hash
        else:
            data_map = self.make_data_map(self.feature_hash)
        if pre_init:
            self._cached_batches = list(balanced_batches(data_map, self._batch_size, self.normalize_classes))
        else:
            self._cached_batches = None
        return data_map

    def __len__(self):
        if not self._cached_batches: 
            self._init_batches(True)
        return len(self._cached_batches)
        
    def __iter__(self):
        batches = self._cached_batches or balanced_batches(self._init_batches(False), self._batch_size, self.normalize_classes)
        for b in batches:
            yield b
