import os
import re
import dill 
import shutil
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
        exclude: List[str] = [],
        check: bool = False, 
    ):
        # parse dataset
        self.dataset_path = Path(dataset_path)
        self._parse_dataset(file_parser, valid_exts, filters, exclude)
        # create output
        self.output_path = Path(output_path)
        if self.output_path.exists():
            raise ValueError('%s seems to exist. Please provide a free path'%output_path)
        os.makedirs(self.output_path.resolve())
        # record parsers
        self.parser = parser
        self.max_db_size = max_db_size
        self.fragment_class = fragment_class
        self.features = features or []
        self.metadata = {'filters': filters, 'exclude': exclude}
        if check:
            print(f'Dataset path : {dataset_path}')
            print(f'Output path : {output_path}')
            print(f'filters: {filters};\texclude: {exclude}')
            print(f'features : {features}')
            print(f'Found {len(self._files)} files')
            out = None
            while out is None:
                out = input('proceed? (y/n): ')
                if out.lower() == "y":
                    out = True
                elif out.lower() == "n":
                    out = False
                else:
                    out = None
            if not out: 
                exit()


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

    def get_feature_name(self, f):
        return getattr(f, "feature_name", type(f).__name__.lower())

    def _init_feature_hash(self, features):
        feature_hash = {'original_path': {}}
        for f in features:
            if getattr(f, "has_hash", False): continue
            feature_name = self.get_feature_name(f)
            feature_hash[feature_name] = {}
        return feature_hash
        
    def _extract_features(self, fragment, current_key, feature_hash):
        for feature in self.features:
            feature.extract(fragment, current_key, feature_hash)

    def _close_features(self):
        for feature in self.features:
            feature.close()
        
    def _add_feature_hash_to_lmdb(self, txn, feature_hash):
        txn.put(
            "feature_hash".encode('utf-8'),
            dill.dumps(feature_hash)
        )

    def build(self):
        env = lmdb.open(str(self.output_path.resolve()), map_size=self.max_db_size * 1024 ** 3)
        audio_id = 0
        n_seconds = 0
        metadata = {}
        feature_hash = self._init_feature_hash(self.features)
        feature_keys = [self.get_feature_name(f) for f in self.features]

        for current_file in tqdm.tqdm(self._files):
            current_parser = self.parser(current_file, features=self.features)
            metadata = self._update_metadata(current_parser.get_metadata(), metadata)
            for load_fn in current_parser:
                try:
                    current_data = load_fn(self.fragment_class)
                    current_file = str(Path(current_file).relative_to(self.dataset_path.resolve()))
                    feature_hash['original_path'][current_file] = feature_hash['original_path'].get(current_file, [])
                    with env.begin(write=True) as txn:
                        for fragment in current_data:
                            current_key = f"{audio_id:09d}"
                            self._extract_features(fragment, current_key, feature_hash)
                            txn.put(
                                current_key.encode(), 
                                fragment.serialize()
                            )
                            audio_id += 1
                            n_seconds += metadata.get('chunk_length', 0) 
                            feature_hash['original_path'][str(current_file)].append(current_key)
                        self._add_feature_hash_to_lmdb(txn, feature_hash)
                except FileNotReadException: 
                        pass

        # write metadata
        metadata_path = self.output_path / "metadata.yaml"
        with open(metadata_path, "w+") as f:
            yaml.safe_dump({
                "n_seconds": n_seconds,
                "writer_class": type(self).__name__,
                "fragment_class": self.fragment_class.__name__, 
                "features": feature_keys,
                **self.metadata, 
                **metadata,
            }, f)
        extras_path = self.output_path / "code"
        os.makedirs(extras_path, exist_ok=True)
        fragment_class_name = self.fragment_class.__module__.split('.')[-1]

        proto_path = Path(__file__).parent / ".." / "fragments" / "interfaces" / f"{fragment_class_name}.proto"
        if os.path.exists(proto_path):
            shutil.copyfile(proto_path, extras_path / proto_path.name)
        compiled_path = Path(__file__).parent / ".." / "fragments" / "compiled" / f"{fragment_class_name}_pb2.py"
        if os.path.exists(compiled_path):
            shutil.copyfile(compiled_path, extras_path / compiled_path.name)

        env.close()
        self._close_features()

    @classmethod
    def open(cls, path):
        return lmdb.open(str(path), lock=False, readonly=True)

    @classmethod
    def get_feature_hash(cls, txn):
        return dill.loads(txn.get('feature_hash'.encode('utf-8'))) 

    @classmethod
    def get_file_ids(cls, txn):
        file_keys = list(txn.cursor().iternext(values=False))
        try:
            idx = file_keys.index(b'feature_hash')
            del file_keys[idx]
        except IndexError:
            pass
        return file_keys

    @classmethod
    def iter_files(cls, txn, fragment_class):
        file_keys = cls.get_file_ids(txn)        
        return {f: fragment_class(txn.get(f)) for f in file_keys}.items()

