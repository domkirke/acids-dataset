import os
import re
import torch
import gin
import logging
from pathlib import Path
from .writers import LMDBWriter
from typing import List

from . import import_database_configs
from .writers import get_writer_class
from .features import AcidsDatasetFeature, append_meta_regexp
from .utils import append_features, GinEnv


def get_default_output_path(path):
    return (Path(".") / f"{path.stem}_preprocessed").resolve().absolute()


def preprocess_dataset(
    path, 
    out = None, 
    config: str = 'default.gin', 
    features: List[str | AcidsDatasetFeature] | None = None,
    check=False, 
    chunk_length: int = None,
    hop_length: int = None, 
    sample_rate = 44100,
    channels = 1,
    flt = [],
    exclude = [],
    meta_regexp = [], 
    force: bool = False, 
    waveform: bool = True, 
    override = [],
    device: str | None = None, 
    max_db_size: int = 100, 
    compact: bool = False
    ):
    # parse gin constants
    import_database_configs()
    gin.add_config_file_search_path(Path(__file__).parent / "configs")
    gin.add_config_file_search_path(path)
    gin.constant('SAMPLE_RATE', sample_rate)
    if chunk_length is not None:
        gin.constant('CHUNK_LENGTH', chunk_length)
        gin.constant('HOP_LENGTH', hop_length or chunk_length // 2)
    gin.constant('CHANNELS', channels)
    gin.constant('DEVICE', device)

    # parse features
    features = features or []
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
                operative_features.extend(append_features(device=device))
        elif isinstance(f, AcidsDatasetFeature):
            operative_features.append(f)

    gin.bind_parameter('append_features.features', operative_features)
    if os.path.splitext(config)[-1] != ".gin":
        config += ".gin"
    gin.parse_config_files_and_bindings([config], override)
    path = Path(path)
    out = out or get_default_output_path(path)

    operative_features = append_meta_regexp(operative_features, meta_regexp=meta_regexp)
    writer_class = get_writer_class(filters=flt, exclude=exclude)
    writer = writer_class(path, 
                          out, 
                          features=operative_features, 
                          check=check, 
                          force=force, 
                          waveform=waveform, 
                          max_db_size=max_db_size) 
    writer.build(compact=compact)