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
from acids_dataset.features import AcidsDatasetFeature, ModuleEmbedding
from acids_dataset.utils import GinEnv, parse_features, feature_from_gin_config, set_gin_constant


def add_embedding_to_dataset(
    path, 
    module: nn.Module = None, 
    module_path: str | Path | None = None,
    module_sr: int | None = None,
    method: str = "forward",
    transforms: List[str | adt.Transform] = [],
    embedding_name: str | None = None,
    device: str | None = None,
    check: bool = False,
    overwrite: bool = False,
    ):
    path = Path(path)

    # parse gin constants
    gin.add_config_file_search_path(Path(__file__).parent / "configs")
    gin.add_config_file_search_path(path)
    metadata = get_metadata_from_path(path)
    set_gin_constant('SAMPLE_RATE', metadata['sr'])
    set_gin_constant('CHANNELS', metadata['channels'])
    set_gin_constant('DEVICE', device)

    if ((embedding_name is None) and (module_path is not None)):
        embedding_name = os.path.splitext(os.path.basename(module_path))[0]

    # parse module
    module_feature = ModuleEmbedding(module=module, 
                                     module_path=module_path, 
                                     module_sr=module_sr, 
                                     method=method, 
                                     transforms=transforms, 
                                     sr=metadata['sr'], 
                                     name=embedding_name)

    # parse contrastive transforms
    operative_transforms = []
    for i, f in enumerate(transforms):
        if isinstance(f, str):
            if os.path.splitext(f)[1] == "": f += ".gin"
            if os.path.exists(f):
                gin.add_config_file_search_path(Path(f).parent)
            try:
                gin.parse_config_file(f)
            except TypeError as e:
                print('[error] problem parsing configuration %s'%f)
                raise e
            with GinEnv(f):
                operative_transforms.extend(feature_from_gin_config(f))
        elif isinstance(f, AcidsDatasetFeature):
            operative_transforms.append(f)

    # build writer
    writer_class = get_writer_class_from_path(path)
    writer_class.update(
        path, 
        features=[module_feature],
        check=check, 
        overwrite=overwrite
    )


def main(argv):
    add_embedding_to_dataset(
        FLAGS.path, 
        module_path=FLAGS.module, 
        module_sr=FLAGS.model_sr, 
        method=FLAGS.method, 
        transforms=FLAGS.transform, 
        embedding_name=FLAGS.embedding_name,
        device=FLAGS.device, 
        check=FLAGS.check, 
        overwrite=FLAGS.overwrite
    )
        

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_string('path', None, 'dataset path', required=True)
    flags.DEFINE_string('module', None, 'path to module (.ts)', required=True)
    flags.DEFINE_string('method', "forward", 'model method to call', required=True)
    flags.DEFINE_multi_string('transforms', [], help='transforms for different embedding views.')
    flags.DEFINE_string('name', None, 'embedding name (default: model name)', required=True)
    flags.DEFINE_integer('model_sr', None, 'original sample rate of target model', required=True)
    flags.DEFINE_string('device', None, 'device id used for computation')
    flags.DEFINE_boolean('overwrite', False, help="recomputes the feature if already present in the dataset, and overwrites existing files")
    flags.DEFINE_boolean('check', True, help="recomputes the feature if already present in the dataset")
    app.run(main)