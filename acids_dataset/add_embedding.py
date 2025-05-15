import os
import gin
from typing import List
import torch, torch.nn as nn
from pathlib import Path
from absl import flags, app

import sys
import sys; sys.path.append(str(Path(__file__).parent.parent))

from acids_dataset import get_metadata_from_path, get_writer_class_from_path
from acids_dataset import writers, transforms as adt
from acids_dataset.features import ModuleEmbedding
from acids_dataset.transforms import Transform
from acids_dataset.utils import GinEnv, parse_features, transform_from_gin_config, set_gin_constant


def import_flags():
    flags.DEFINE_string('path', None, 'dataset path', required=True)
    flags.DEFINE_string('module', None, 'path to module (.ts)', required=True)
    flags.DEFINE_string('method', "forward", 'model method to call', required=False)
    flags.DEFINE_string('name', None, 'embedding name (default : filename)', required=False)
    flags.DEFINE_multi_string('transform', [], help='transforms for different embedding views.')
    flags.DEFINE_integer('model_sr', None, 'original sample rate of target model', required=False)
    flags.DEFINE_string('device', None, 'device id used for computation')
    flags.DEFINE_boolean('overwrite', False, help="recomputes the feature if already present in the dataset, and overwrites existing files")
    flags.DEFINE_boolean('check', True, help="recomputes the feature if already present in the dataset")

def add_embedding_to_dataset(
        path, 
        module: nn.Module = None, 
        module_path: str | Path | None = None,
        module_sr: int | None = None,
        method: str = "forward",
        transforms: List[str | adt.Transform] = [],
        name: str | None = None,
        device: str | None = None,
        check: bool = False,
        overwrite: bool = False,
    ):

    path = Path(path)
    if name is None: 
        if module is None: 
            name = name or Path(module_path).stem
        elif module_path is None:
            name = name or type(module).__name__.capitalize()
        else:
            raise ValueError('either module or module_path must be given')

    # parse gin constants
    gin.add_config_file_search_path(Path(__file__).parent / "configs")
    gin.add_config_file_search_path(path)
    metadata = get_metadata_from_path(path)
    set_gin_constant('SAMPLE_RATE', metadata['sr'])
    set_gin_constant('CHANNELS', metadata['channels'])
    set_gin_constant('DEVICE', device)

    operative_transforms = []
    for t in transforms:
        if isinstance(t, str):
            operative_transforms.append(transform_from_gin_config(t))
        elif isinstance(t, Transform):
            operative_transforms.append(t)

    print(transforms, operative_transforms)


    # parse module
    module_feature = ModuleEmbedding(module=module, 
                                     module_path=module_path, 
                                     module_sr=module_sr, 
                                     method=method, 
                                     transforms=operative_transforms, 
                                     sr=metadata['sr'], 
                                     name=name, 
                                     device=device)

    # build writer
    writer_class = get_writer_class_from_path(path)
    writer_class.update(
        path, 
        features=[module_feature],
        check=check, 
        overwrite=overwrite
    )


def main(argv):
    FLAGS = flags.FLAGS
    add_embedding_to_dataset(
        FLAGS.path, 
        module_path=FLAGS.module, 
        module_sr=FLAGS.model_sr, 
        method=FLAGS.method, 
        transforms=FLAGS.transform, 
        name=FLAGS.name,
        device=FLAGS.device, 
        check=FLAGS.check, 
        overwrite=FLAGS.overwrite
    )


if __name__ == "__main__":
    app.run(main)


__name__ = "add_embedding"