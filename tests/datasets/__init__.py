import os
import importlib
import itertools
from pathlib import Path

__DATASET_HASH = {
    'simple': ['simple_dataset', tuple()],
    'simple_midi': ['simple_midi_dataset', tuple()],
    'slakh_like': ['slakh_like', tuple()]
}

__DEFAULT_PATH = Path(__file__).resolve().parent

def download_dataset(path, mirror):
    return False

def is_dataset_available(dataset_card, path=None):
    if dataset_card not in __DATASET_HASH:
        raise ValueError("dataset %s not known"%dataset_card)
    model_path, _ = __DATASET_HASH[dataset_card]
    path = path or __DEFAULT_PATH
    target_path = path / model_path
    return target_path.exists()

def get_available_datasets(path=None):
    path = path or __DEFAULT_PATH
    available_datasets = []
    for card, (dataset_path, mirrors) in __DATASET_HASH.items():
        if is_dataset_available(card, path): available_datasets.append(card)
    return available_datasets

def get_dataset(dataset_card, path=None):
    if dataset_card not in __DATASET_HASH:
        raise ValueError("dataset %s not known"%dataset_card)
    model_path, mirrors = __DATASET_HASH[dataset_card]
    path = path or __DEFAULT_PATH
    target_path = path / model_path
    if not target_path.exists():
        for m in mirrors:
            out = download_dataset(path, m)
            if out: break
    return target_path

def get_filters_from_dataset(dataset_card, path=None):
    path = path or __DEFAULT_PATH
    model_path, _ = __DATASET_HASH[dataset_card]
    target_path = path / model_path
    metadata_path = (target_path / "test_metadata.py")
    if not metadata_path.exists():
        return [([], [])]
    spec = importlib.util.spec_from_file_location('metadata_path', metadata_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path_filters = getattr(module, "path_filters", []) + [([], [])]
    return path_filters
    
def get_available_datasets_with_filters(path=None):
    datasets = get_available_datasets(path)
    data_and_flt = []
    for d in datasets:
        filters = get_filters_from_dataset(d, path=path)
        data_and_flt.extend([d, flt] for flt in filters)
    return data_and_flt

    



