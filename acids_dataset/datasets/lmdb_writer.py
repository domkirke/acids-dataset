import os
import lmdb
import tqdm
from pathlib import Path
from .utils import audio_paths_from_dir
from ..features import AcidsDatasetFeature
from ..parsers import RawParser, FileNotReadException
from typing import List, Callable
import gin
import yaml

@gin.configurable(module="writer")
class LMDBWriter(object):
    def __init__(
        self, 
        dataset_path: str | Path, 
        output_path: str | Path,
        fragment_class: str, 
        features: List[AcidsDatasetFeature] | None = None, 
        parser: str | object = RawParser, 
        max_db_size: int | float | None = 100, 
        valid_exts: List[str] | None = None,
        file_parser: Callable = audio_paths_from_dir,
        filters: List[str] = [], 
        exclude: List[str] = [] 
    ):
        self.dataset_path = Path(dataset_path)
        # parse dataset
        self._parse_dataset(file_parser, valid_exts, filters, exclude)
        # create output
        self.output_path = Path(output_path)
        if self.output_path.exists():
            raise ValueError('%s seems to exist. Please provide a free path')
        os.makedirs(self.output_path.resolve())
        # record parsers
        self.fragment_class = fragment_class
        self.features = features
        self.parser = parser
        self.max_db_size = max_db_size

    def _parse_dataset(self, file_parser, valid_exts = None, filters=None, exclude=None, dataset_path=None):
        dataset_path = dataset_path or self.dataset_path
        self._files = file_parser(dataset_path, valid_exts = valid_exts, flt=filters, exclude=exclude)
        self.valid_exts = valid_exts
        self.filters = filters
        self.exclude = exclude

    def _update_metadata(self, metadata, metadata_dict):
        for k, v in metadata.items():
            if k not in metadata_dict: metadata_dict[k] = v
        return metadata_dict
    
    def _extract_features(self, fragment):
        for feature in self.features:
            feature.get(fragment)

    def build(self):
        env = lmdb.open(str(self.output_path.resolve()), map_size=self.max_db_size * 1024 ** 3)
        audio_id = 0
        metadata = {}
        for current_file in tqdm.tqdm(self._files):
            current_parser = self.parser(current_file)
            metadata = self._update_metadata(current_parser.get_metadata(), metadata)
            for load_fn in current_parser:
                try:
                    current_data = load_fn(self.fragment_class)
                    for fragment in current_data:
                        self._extract_features(current_data)
                    with env.begin(write=True) as txn:
                        current_key = f"{audio_id:09d}"
                        for fragment in current_data:
                            txn.put(
                                current_key.encode(), 
                                fragment.serialize()
                            )
                        audio_id += 1
                except FileNotReadException: 
                        pass
        metadata_path = self.output_path / "metadata.yaml"
        with open(metadata_path, "w+") as f:
            yaml.safe_dump({
                "n_seconds": audio_id * metadata.get('chunk_length', 0),
                **metadata,
            }, f)
        env.close()




        


