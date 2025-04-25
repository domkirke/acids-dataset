import torch
import random, math, itertools
import copy
from typing import Optional, List , Dict, Iterable
from .. import writers
from .. import transforms, get_writer_class_from_path, get_metadata_from_path
from .utils import _outs_from_pattern, _transform_outputs

TransformType = Optional[transforms.Transform | List[transforms.Transform] | Dict[str, transforms.Transform]]

def _parse_transforms_with_pattern(transform, pattern):
    return transform

class AudioDataset(torch.utils.data.Dataset):

    def __init__(self,
                 db_path: str,
                 transforms: TransformType = None, 
                 output_pattern: str = 'waveform',
                 channels: int = 1, 
                 lazy_import: bool = False, 
                 lazy_paths: str = False,
                 subindices: Iterable[int] | Iterable[bytes] | None = None,
                 parent = None,
                 **kwargs) -> None:
        self._db_path = db_path
        if lazy_import or lazy_paths:
            raise NotImplementedError()
        self._loader = getattr(writers, get_writer_class_from_path(db_path).loader)(self._db_path, 
                                                                                    output_type="torch") 
        self._metadata = get_metadata_from_path(db_path)
        self._output_pattern = output_pattern
        self._transforms = _parse_transforms_with_pattern(transforms, self._output_pattern)
        self._channels = channels
        self._subindices = subindices
        self.parent = parent
        super(AudioDataset, self).__init__(**kwargs)

    def __getitem__(self, index):
        fg = self._loader[index]
        outs = _outs_from_pattern(fg, self.output_pattern)
        outs = _transform_outputs(outs, self.transforms)
        return outs
    
    def __len__(self):
        return len(self._loader)

    @property 
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, obj):
        assert issubclass(type(obj), type(self)), "parent of a dataset must be a subclass of %s"%(type(obj))

    @property 
    def metadata(self):
        return copy.copy(self._metadata)

    @property
    def is_partition(self):
        return self._subindices is not None

    @property
    def output_pattern(self):
        return self._output_pattern
    
    @output_pattern.setter
    def output_pattern(self, pattern):
        assert isinstance(pattern, str)
        self._output_pattern = pattern
        self._transforms = _parse_transforms_with_pattern(self._transforms, self._output_pattern)

    @property
    def transforms(self):
        return self._transforms

    @transforms.setter
    def transforms(self, transforms):
        self._transforms = _parse_transforms_with_pattern(transforms, self._output_pattern)

    @property
    def keys(self) -> List[str]:
        return list(self._loader.iter_fragment_keys())

    def split(self, feature=None, partitions=None):
        """automatically look for set (or provided) feature, otherwise split randomly"""
        assert feature is not None or partitions is not None, "either feature or partitions must be provided"
        if feature is None: 
            feature = "set" if "set" in map(lambda x: x.feature_name, self._loader.features) else None
            if feature is not None: 
                return self.split_by_feature(feature)
        if partitions is not None:
            return self.split_random(**partitions)

    def split_random(self, **kwargs):
        assert not self.is_partition, "dataset is already a partition of an existing dataset."
        """randomly split the feature"""
        ratios = {k: float(v) for k, v in kwargs.items()}
        n_items = len(self._loader)
        idx_perm = random.sample(range(n_items), k=n_items)
        idx_slices = list(itertools.accumulate([r * n_items for r in ratios]))
        raise NotImplementedError

    def split_by_feature(self, feature_name: str):
        assert not self.is_partition, "dataset is already a partition of an existing dataset."
        raise NotImplementedError

        


