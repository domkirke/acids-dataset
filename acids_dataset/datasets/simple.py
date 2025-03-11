import torch
import copy
from typing import Optional, List , Dict
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
        super(AudioDataset, self).__init__(**kwargs)

    @property 
    def metadata(self):
        return copy.copy(self._metadata)

    def __len__(self):
        return len(self._loader)

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

    def __getitem__(self, index):
        fg = self._loader[index]
        outs = _outs_from_pattern(fg, self.output_pattern)
        outs = _transform_outputs(outs, self.transforms)
        return outs

        


