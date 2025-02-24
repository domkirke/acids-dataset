import pytest
import os
from pathlib import Path
import shutil
import gin
from . import OUT_TEST_DIR, test_name
from .datasets import get_available_datasets, get_dataset

from acids_dataset.datasets import LMDBWriter
from acids_dataset.features.mel import Mel


def get_feature_configs(feature_name):
    feature_path = Path(__file__).parent / "feature_configs" / feature_name
    if not feature_path.exists():
        return []
    feature_configs = list(filter(lambda x: os.path.splitext(x)[1] == ".gin", os.listdir(feature_path.resolve())))
    return [(feature_path.resolve(), fc) for fc in feature_configs]


@pytest.mark.parametrize('config', ['default.gin'])
@pytest.mark.parametrize("dataset", get_available_datasets())
@pytest.mark.parametrize('feature_path,feature_config', get_feature_configs('mel'))
def test_mel(config, feature_path, feature_config, dataset, test_name):
    gin.add_config_file_search_path(feature_path)
    gin.constant('SAMPLE_RATE', 44100)
    gin.constant('CHANNELS', 1)
    gin.parse_config_file(config)
    gin.parse_config_file(feature_config)

    dataset_path = get_dataset(dataset)
    dataset_out = OUT_TEST_DIR / "compiled" / test_name
    if dataset_out.exists():
        shutil.rmtree(dataset_out.resolve())

    # extract mel
    mel_feature = Mel()

    # build dataset
    writer = LMDBWriter(dataset_path, dataset_out, features=[mel_feature])
    writer.build()
    
        

