import os, sys
import shutil
import torch
import torchaudio
import pytest
import gin


from pathlib import Path

CURRENT_TEST_DIR = Path(__file__).parent
OUT_TEST_DIR = CURRENT_TEST_DIR / "outs"
os.makedirs(OUT_TEST_DIR, exist_ok=True)

sys.path.append(str((CURRENT_TEST_DIR / '..').resolve()))
gin.add_config_file_search_path(str((CURRENT_TEST_DIR / '..' / 'configs').resolve()))

from acids_dataset.datasets import audio_paths_from_dir, LMDBWriter
from acids_dataset.parsers import raw_parser as raw
from acids_dataset.utils import loudness
from datasets import get_available_datasets, get_dataset, get_available_datasets_with_filters

@pytest.fixture
def test_name(request):
    return request.node.name

@pytest.mark.parametrize("dataset,filters", get_available_datasets_with_filters())
def test_parse_dataset_files(dataset, filters, test_name):
    dataset_path = get_dataset(dataset)
    flt, ex = filters
    valid_files = audio_paths_from_dir(dataset_path, flt=flt, exclude=ex)
    with open(Path(OUT_TEST_DIR) / f"{test_name}.txt", "w+") as f:
        f.write(f"filters: {flt}\n")
        f.write(f"exclude: {ex}\n")
        f.write("\n".join(valid_files))

# @pytest.mark.parametrize("dataset,filters", get_available_datasets_with_filters())
@pytest.mark.parametrize("loudness_threshold", [None, -70, -10])
@pytest.mark.parametrize("hop_length,overlap", [(0.5, None), (None, 0.8), (4096, None), (None, 4096)])
@pytest.mark.parametrize("chunk_length", [1., 8192])
@pytest.mark.parametrize("pad_mode", list(raw.PadMode.__members__.keys()))
@pytest.mark.parametrize("import_backend", list(raw.ImportBackend.__members__.keys()))
@pytest.mark.parametrize("parser", [raw.RawParser])
@pytest.mark.parametrize("dataset", get_available_datasets())
def test_raw_parser(dataset, test_name, parser, import_backend, pad_mode, chunk_length, hop_length, overlap, loudness_threshold):
    dataset_path = get_dataset(dataset)
    valid_files = audio_paths_from_dir(dataset_path)
    file_exceptions = []
    file_stats = {}
    for path in valid_files:
        current_parser = parser(
            path, 
            chunk_length=chunk_length, 
            hop_length=hop_length, 
            overlap=overlap, 
            sr = 44100,
            pad_mode=pad_mode, 
            import_backend=import_backend, 
            loudness_threshold=loudness_threshold            
        )
        for obj in current_parser:
            try:
                data = obj()
                assert len(list(filter(lambda x, l = current_parser.chunk_length_smp: x.shape[-1] != l, data))) == 0
                if loudness_threshold is not None:
                    assert len(list(filter(lambda x, t = loudness_threshold, sr = current_parser.sr: loudness(torch.Tensor(x), sr) < t, data))) == 0
                file_stats[path] = {"n_chunks": len(data)}
            except raw.FileNotReadException as e:
                file_exceptions.append(e)

    with open(Path(OUT_TEST_DIR) / f"{test_name}.txt", "w+") as f:
        f.write(f"--failed files: \n{'\n'.join(map(str, file_exceptions))}")
        f.write("\n--filewise information :\n")
        for p in valid_files:
            p_name = Path(p).relative_to(dataset_path)
            f.write(f"{p_name}\t{file_stats.get(p, "MISSING")}\n")
            
@pytest.mark.parametrize('config', ['rave.gin'])
@pytest.mark.parametrize("dataset", get_available_datasets())
def test_build_dataset(config, dataset, test_name):
    gin.constant('SAMPLE_RATE', 44100)
    gin.constant('CHANNELS', 1)
    gin.parse_config_file(config)
    dataset_path = get_dataset(dataset)
    dataset_out = OUT_TEST_DIR / "compiled" / test_name
    if dataset_out.exists():
        shutil.rmtree(dataset_out.resolve())
    writer = LMDBWriter(dataset_path, dataset_out)
    writer.build()

        

