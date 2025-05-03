import os, sys
import random
import lmdb
import shutil
import torch
import torchaudio
import pytest
import gin

from . import OUT_TEST_DIR, test_name, get_available_features
from pathlib import Path
from acids_dataset.writers import audio_paths_from_dir, LMDBWriter, read_metadata
from acids_dataset.parsers import raw_parser as raw
from acids_dataset.datasets import AudioDataset
from acids_dataset.utils import loudness, GinEnv, feature_from_gin_config
from acids_dataset import transforms, ACIDS_DATASET_CONFIG_PATH
from acids_dataset import get_fragment_class, features, preprocess_dataset
from .datasets import get_available_datasets, get_dataset, get_available_datasets_with_filters




 
@pytest.mark.parametrize('config', ['rave.gin'])
@pytest.mark.parametrize("dataset", ["simple"])
@pytest.mark.parametrize("feature", get_available_features())
def test_update_dataset_features(config, dataset, feature, test_name, test_k = 1):
    # test writing
    gin.constant('SAMPLE_RATE', 44100)
    gin.constant('CHANNELS', 1)
    gin.constant('DEVICE', "cpu")
    gin.parse_config_file(config)
    dataset_path = get_dataset(dataset)
    dataset_out = OUT_TEST_DIR / "compiled" / test_name
    if dataset_out.exists():
        shutil.rmtree(dataset_out.resolve())

    # create feature
    with GinEnv(paths=ACIDS_DATASET_CONFIG_PATH):
        features = feature_from_gin_config(feature)

    writer = LMDBWriter(dataset_path, dataset_out)
    writer.build()

    # test loading
    env = lmdb.open(str(dataset_out), lock=False, readonly=True)
    fragment_class = get_fragment_class(read_metadata(dataset_out)['fragment_class'])
    with env.begin() as txn:
        dataset_keys = list(txn.cursor().iternext(values=False))
        # pick a random item
        random_keys = random.choices(dataset_keys, k=test_k)
        for key in random_keys:
            ae = fragment_class(txn.get(key))
            audio = ae.get_audio("waveform")
