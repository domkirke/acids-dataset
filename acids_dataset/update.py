import os
import re
import gin
import logging
from typing import List
from pathlib import Path
from . import get_metadata_from_path, get_writer_class_from_path
from . import writers
from .features import AcidsDatasetFeature, append_meta_regexp
from .utils import GinEnv, append_features


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
    device: str | None = None
    ):
    path = Path(path)
    # parse gin constants
    gin.add_config_file_search_path(Path(__file__).parent / "configs")
    gin.add_config_file_search_path(path)
    metadata = get_metadata_from_path(path)
    gin.constant('SAMPLE_RATE', metadata['sr'])
    gin.constant('CHANNELS', metadata['channels'])
    gin.constant('DEVICE', device)

    # parse features
    operative_features = []
    for i, f in enumerate(features):
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
                operative_features.extend(append_features())
        elif isinstance(f, AcidsDatasetFeature):
            operative_features.append(f)

    # parse original config
    gin.parse_config_files_and_bindings([str(path / "config.gin")], override)
    operative_features = append_meta_regexp(operative_features, meta_regexp=meta_regexp)

    # build writer
    writer_class = get_writer_class_from_path(path)
    writer_class = writers.get_writer_class(writer_class, flt, exclude)
    writer_class.update(
        path, 
        operative_features,
        data, 
        check=check, 
        overwrite=overwrite
    )