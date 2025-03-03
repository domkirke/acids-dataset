from pathlib import Path
import yaml
import gin
gin.add_config_file_search_path(Path(__file__).parent / "configs")
gin.add_config_file_search_path(Path(__file__).parent.parent / "custom_configs")

from . import utils
from . import transforms
from . import features
from . import fragments
from . import datasets

def get_fragment_class(class_name):
    return getattr(fragments,class_name) 

def get_metadata_from_path(dataset_path):
    dataset_path = Path(dataset_path)
    metadata_path = dataset_path / "metadata.yaml"
    with open(metadata_path, "r") as f:
        metadata = yaml.safe_load(f)
    return metadata

def get_writer_from_metadata(metadata):
    return getattr(datasets, metadata.get('writer_class', 'LMDBWriter'))

def get_writer_class_from_path(path):
    metadata = get_metadata_from_path(path)
    return getattr(datasets, metadata.get('writer_class', 'LMDBWriter'))

def get_env_from_path(path):
    writer_class = get_writer_class_from_path(path)
    return writer_class.open(path)

def get_feature_hash_from_path(path):
    writer_class = get_writer_class_from_path(path)
    return writer_class.get_feature_hash(path)

def get_fragment_class_from_path(path):
    metadata = get_metadata_from_path(path)
    return get_fragment_class(metadata['fragment_class'])

from . import datasets
from .preprocess import *