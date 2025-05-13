import os
import gin
from typing import List
from pathlib import Path
from absl import flags, app

import sys
import sys; sys.path.append(str(Path(__file__).parent.parent))

from acids_dataset import get_metadata_from_path, get_writer_class_from_path, get_feature_config_paths
from acids_dataset import writers, import_database_configs
from acids_dataset.features import AcidsDatasetFeature, append_meta_regexp 
from acids_dataset.utils import GinEnv, parse_features, feature_from_gin_config, set_gin_constant


def import_flags():
    flags.DEFINE_string('path', None, 'dataset path', required=True)
    flags.DEFINE_multi_string('data', [], help='add audio files to the target dataset.')
    flags.DEFINE_multi_string('feature', [], help='add features to the target dataset.')
    flags.DEFINE_multi_string('filter', [], help="wildcard to filter target files")
    flags.DEFINE_multi_string('exclude', [], help="wildcard to exclude target files")
    flags.DEFINE_multi_string('meta_regexp', [], 'parses additional blob wildcards as features.')
    flags.DEFINE_multi_string('override', [], 'add audio files to the target dataset.')
    flags.DEFINE_boolean('overwrite', False, help="recomputes the feature if already present in the dataset, and overwrites existing files")
    flags.DEFINE_boolean('check', True, help="recomputes the feature if already present in the dataset")


def update_dataset(
        path, 
        features: List[str | AcidsDatasetFeature] | None = None,
        data: List[str] | None = None,
        check: bool = False, 
        flt = [],
        exclude = [],
        meta_regexp = [], 
        overwrite: bool = False,
        override = [], 
        device: str | None = None,
        max_db_size: int | None = None
    ):
    path = Path(path)

    # parse original config
    import_database_configs(path)
    metadata = get_metadata_from_path(path)
    gin.parse_config_files_and_bindings([str(path / "config.gin")], override, finalize_config=False)
    set_gin_constant('DEVICE', device or "cpu", True)

    # parse features
    features = features or []
    operative_features = parse_features()
    operative_features = append_meta_regexp(operative_features, meta_regexp=meta_regexp)
    for i, f in enumerate(features):
        if isinstance(f, str):
            operative_features.extend(feature_from_gin_config(f))
        elif isinstance(f, AcidsDatasetFeature):
            operative_features.append(f)

    gin.finalize()
    # build writer
    writer_class = get_writer_class_from_path(path)
    writer_class = writers.get_writer_class(writer_class, flt, exclude)
    writer_class.update(
        path, 
        operative_features,
        data, 
        check=check, 
        overwrite=overwrite, 
        max_db_size=max_db_size,
    )


def main(argv):
    FLAGS = flags.FLAGS
    update_dataset(
        FLAGS.path, 
        features=FLAGS.feature,
        data = FLAGS.data, 
        overwrite = FLAGS.overwrite, 
        check = FLAGS.check
    )
        

if __name__ == "__main__":
    app.run(main)