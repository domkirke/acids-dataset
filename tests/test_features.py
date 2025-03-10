import pytest
import os
from pathlib import Path
import shutil
import gin
from . import OUT_TEST_DIR, test_name
import lmdb, random
from .datasets import get_available_datasets, get_dataset

from acids_dataset import get_fragment_class
from acids_dataset.writers import LMDBWriter, read_metadata
from acids_dataset.features import Mel, Loudness, AfterMIDI


def get_feature_configs(feature_name):
    feature_path = Path(__file__).parent / "feature_configs" / feature_name
    if not feature_path.exists():
        return []
    feature_configs = list(filter(lambda x: os.path.splitext(x)[1] == ".gin", os.listdir(feature_path.resolve())))
    return [(feature_path.resolve(), fc) for fc in feature_configs]


@pytest.mark.parametrize('config', ['default.gin'])
@pytest.mark.parametrize("dataset", get_available_datasets())
@pytest.mark.parametrize('feature_path,feature_config', get_feature_configs('mel'))
def test_mel(config, feature_path, feature_config, dataset, test_name, test_k=10):
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

    env = lmdb.open(str(dataset_out), lock=False, readonly=True)
    fragment_class = get_fragment_class(read_metadata(dataset_out)['fragment_class'])
    mel_key = read_metadata(dataset_out)['features'][0]
    with env.begin() as txn:
        dataset_keys = list(txn.cursor().iternext(values=False))
        # pick a random item
        random_keys = random.choices(dataset_keys, k=test_k)
        for key in random_keys:
            ae = fragment_class(txn.get(key))
            audio = ae.get_audio("waveform")
            mel = ae.get_array(mel_key)
    
        


@pytest.mark.parametrize('config', ['default.gin'])
@pytest.mark.parametrize("dataset", get_available_datasets())
def test_loudness(config,  dataset, test_name, test_k=10):
    gin.constant('SAMPLE_RATE', 44100)
    gin.constant('CHANNELS', 1)
    gin.parse_config_file(config)

    dataset_path = get_dataset(dataset)
    dataset_out = OUT_TEST_DIR / "compiled" / test_name
    if dataset_out.exists():
        shutil.rmtree(dataset_out.resolve())

    # extract mel
    mel_feature = Loudness()

    # build dataset
    writer = LMDBWriter(dataset_path, dataset_out, features=[mel_feature])
    writer.build()

    env = lmdb.open(str(dataset_out), lock=False, readonly=True)
    fragment_class = get_fragment_class(read_metadata(dataset_out)['fragment_class'])
    with env.begin() as txn:
        dataset_keys = list(txn.cursor().iternext(values=False))
        # pick a random item
        random_keys = random.choices(dataset_keys, k=test_k)
        for key in random_keys:
            ae = fragment_class(txn.get(key))
            audio = ae.get_audio("waveform")
            loudness = ae.get_array("loudness")
    
        

@pytest.mark.parametrize('config', ['default.gin'])
@pytest.mark.parametrize("dataset", get_available_datasets())
@pytest.mark.parametrize('feature_path,feature_config', get_feature_configs('midi'))
def test_after_midi(config, dataset, feature_path, feature_config, test_name, test_k=10):
    gin.add_config_file_search_path(feature_path)
    gin.constant('SAMPLE_RATE', 44100)
    gin.constant('CHANNELS', 1)
    gin.parse_config_file(config)

    dataset_path = get_dataset(dataset)
    dataset_out = OUT_TEST_DIR / "compiled" / test_name
    if dataset_out.exists():
        shutil.rmtree(dataset_out.resolve())
    gin.parse_config_file(feature_config)

    # extract mel
    mel_feature = AfterMIDI()

    # build dataset
    writer = LMDBWriter(dataset_path, dataset_out, features=[mel_feature])
    writer.build()

    env = lmdb.open(str(dataset_out), lock=False, readonly=True)
    fragment_class = get_fragment_class(read_metadata(dataset_out)['fragment_class'])
    with env.begin() as txn:
        dataset_keys = list(txn.cursor().iternext(values=False))
        # pick a random item
        random_keys = random.choices(dataset_keys, k=test_k)
        for key in random_keys:
            ae = fragment_class(txn.get(key))
            audio = ae.get_audio("waveform")
            midi = ae.get_data("midi")
    
        
