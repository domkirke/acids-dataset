import pytest
import acids_dataset
from acids_dataset import transforms as atf
from . import OUT_TEST_DIR
from .datasets import get_available_datasets, get_dataset



@pytest.mark.parametrize("dataset", get_available_datasets())
@pytest.mark.parametrize("output_pattern,transforms", [
    ("waveform", []),
    ("waveform,", [atf.Gain()]),
    ("{waveform,}", {'waveform': atf.Gain()})
])
def test_audio_dataset(dataset, transforms, output_pattern, test_name):
    preprocessed_path = OUT_TEST_DIR / f"{dataset}_preprocessed"
    if not preprocessed_path.exists():
        dataset_path = get_dataset(dataset)
        atf.preprocess_dataset(dataset_path, out = preprocessed_path)
    dataset = atf.AudioDataset(preprocessed_path, transforms, output_pattern)
    assert len(dataset) > 0, "dataset seems empty"
    for i in range(len(dataset)):
        out = dataset[i]