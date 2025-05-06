from pathlib import Path
import yaml
import gin
gin.enter_interactive_mode()

ACIDS_DATASET_CONFIG_PATH = Path(__file__).parent / "configs"
ACIDS_DATASET_CUSTOM_CONFIG_PATH = Path(__file__).parent.parent / "custom_configs"
TRANSFORM_GIN_PATH = ACIDS_DATASET_CONFIG_PATH / "transforms"
FEATURES_GIN_PATH = ACIDS_DATASET_CONFIG_PATH / "features"

def import_database_configs():
    gin.add_config_file_search_path(ACIDS_DATASET_CONFIG_PATH)
    gin.add_config_file_search_path(ACIDS_DATASET_CUSTOM_CONFIG_PATH)

from . import utils
from . import transforms

transforms.check_transform_configs(transforms, TRANSFORM_GIN_PATH)
def get_transform_config_path(): 
    return TRANSFORM_GIN_PATH

from . import parsers
from . import features

features.check_feature_configs(features, FEATURES_GIN_PATH)
def get_feature_config_path(): 
    return FEATURES_GIN_PATH

from . import fragments

def get_fragment_class(class_name):
    obj = getattr(fragments, class_name, None) 
    if not issubclass(obj, fragments.AudioFragment):
        raise TypeError('fragment class %s not valid'%class_name)
    return obj

def get_parser_class(class_name):
    obj = getattr(parsers, class_name, None) 
    # if not issubclass(obj, fragments.AudioFragment):
    #     raise TypeError('fragment class %s not valid'%class_name)
    return obj

def get_metadata_from_path(dataset_path):
    dataset_path = Path(dataset_path)
    metadata_path = dataset_path / "metadata.yaml"
    with open(metadata_path, "r") as f:
        metadata = yaml.safe_load(f)
    return metadata

def get_writer_from_metadata(metadata):
    return getattr(writers, metadata.get('writer_class', 'LMDBWriter'))

def get_writer_class_from_path(path):
    metadata = get_metadata_from_path(path)
    return getattr(writers, metadata.get('writer_class', 'LMDBWriter'))

def get_env_from_path(path):
    writer_class = get_writer_class_from_path(path)
    return writer_class.open(path)

def get_feature_hash_from_path(path):
    writer_class = get_writer_class_from_path(path)
    return writer_class.get_feature_hash(path)

def get_fragment_class_from_path(path):
    metadata = get_metadata_from_path(path)
    return get_fragment_class(metadata['fragment_class'])

def get_parser_class_from_path(path):
    metadata = get_metadata_from_path(path)
    return get_parser_class(metadata['parser_class'])

from . import writers
from . import datasets
from .preprocess import *
from .update import *
from .utils import GinEnv