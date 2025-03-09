import os
import logging
import collections
import re
import dill 
import shutil
import lmdb
import tqdm
from pathlib import Path
from .utils import audio_paths_from_dir, FeatureHash, KeyIterator
from .. import get_fragment_class, get_parser_class_from_path, get_metadata_from_path, get_fragment_class_from_path
from ..parsers import RawParser, FileNotReadException
from ..fragments import AudioFragment
from ..features import AcidsDatasetFeature
from typing import List, Callable
import gin
import yaml

@gin.configurable(module="writer")
class LMDBWriter(object):
    non_buffer_keys = ['feature_hash', 'features']
    def __init__(
        self, 
        dataset_path: str | Path, 
        output_path: str | Path,
        fragment_class: str | AudioFragment = "AcidsFragment", 
        features: List[AcidsDatasetFeature] | None = None, 
        parser: str | object = RawParser, 
        max_db_size: int | float | None = 100, 
        valid_exts: List[str] | None = None,
        dir_parser: Callable = audio_paths_from_dir,
        filters: List[str] = [], 
        exclude: List[str] = [],
        check: bool = False, 
        force: bool = False,
        waveform: bool = True,
    ):
        # parse dataset
        self.dataset_path = Path(dataset_path)
        self._parse_dataset(dir_parser, valid_exts, filters, exclude)
        # create output
        self.output_path = Path(output_path).resolve()
        if self.output_path.exists():
            if force:
                shutil.rmtree(self.output_path)
            else:
                raise ValueError('%s seems to exist. Please provide a free path'%output_path)
        os.makedirs(self.output_path)
        # record parsers
        self.parser = parser
        self.max_db_size = max_db_size
        if isinstance(fragment_class, str): fragment_class = get_fragment_class(fragment_class)
        self.fragment_class = fragment_class
        self.features = features or []
        self.metadata = {'filters': filters, 'exclude': exclude}
        self.waveform = waveform
        if check:
            print(f'Dataset path : {dataset_path}')
            print(f'Output path : {output_path}')
            print(f'filters: {filters};\texclude: {exclude}')
            print(f'features : {features}')
            print(f'Found {len(self._files)} files')
            if not waveform:
                print('No waveform import')
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

    def _parse_dataset(self, dir_parser, valid_exts = None, filters=None, exclude=None, dataset_path=None):
        dataset_path = dataset_path or self.dataset_path
        self._files = dir_parser(dataset_path, valid_exts = valid_exts, flt=filters, exclude=exclude)
        if len(self._files) == 0:
            raise RuntimeError(f'no valid files were found in {dataset_path} with flt={filters} and exclude={exclude}.')
        self.valid_exts = valid_exts
        self.filters = filters
        self.exclude = exclude

    def _update_metadata(self, metadata, metadata_dict):
        for k, v in metadata.items():
            if k not in metadata_dict: metadata_dict[k] = v
        return metadata_dict

    @staticmethod
    def get_feature_name(f):
        return getattr(f, "feature_name", type(f).__name__.lower())

    def _init_feature_hash(self):
        feature_hash = FeatureHash(original_path={})
        return feature_hash
        
    @staticmethod
    def _extract_features(fragment, features, current_key, feature_hash, overwrite=False):
        for feature in features:
            feature_name = feature.feature_name
            if fragment.has_buffer(feature_name) or fragment.has_metadata(feature_name):
                if not overwrite:
                    logging.info(f"metadata {feature_name} already present in fragment {current_key} ; skipping")
                    continue
            feature.extract(fragment=fragment, 
                            current_key=current_key, 
                            feature_hash=feature_hash)

    @staticmethod
    def _close_features(features):
        for feature in features:
            feature.close()
        
    @staticmethod
    def _add_feature_hash_to_lmdb(txn, feature_hash):
        txn.put(
            "feature_hash".encode('utf-8'),
            dill.dumps(dict(feature_hash))
        )

    @staticmethod
    def _add_features_to_lmdb(txn, features):
        binarized_features = collections.OrderedDict()
        for f in features:
            binarized_features[f.feature_name] = f
        txn.put(
            "features".encode('utf-8'), 
            dill.dumps(binarized_features)
        )

    @classmethod
    def _add_file_to_lmdb(cls, 
                          txn, 
                          parser, 
                          features, 
                          feature_hash, 
                          key_generator, 
                          fragment_class, 
                          dataset_path):
        n_seconds = 0
        current_file = parser.audio_path
        for load_fn in parser:
            try:
                current_data = load_fn(fragment_class)
                current_file = str(Path(current_file).relative_to(dataset_path.resolve().parent))
                feature_hash['original_path'][current_file] = feature_hash['original_path'].get(current_file, [])
                for fragment in current_data:
                    current_key = next(key_generator)
                    cls._extract_features(fragment, features, current_key, feature_hash)
                    txn.put(
                        current_key.encode(), 
                        fragment.serialize()
                    )
                    feature_hash['original_path'][str(current_file)].append(current_key)
            except FileNotReadException: 
                pass
        return n_seconds

    def build(self):
        env = lmdb.open(str(self.output_path.resolve()), map_size=self.max_db_size * 1024 ** 3)
        n_seconds = 0
        metadata = {}
        feature_hash = self._init_feature_hash()
        dataset_path = self.dataset_path.resolve().absolute()
        key_iterator = iter(KeyIterator())
        with env.begin(write=True) as txn:
            for current_file in tqdm.tqdm(self._files):
                parser = self.parser(current_file, features=self.features, dataset_path = dataset_path, waveform=self.waveform) 
                if len(metadata) == 0:
                    metadata = self._update_metadata(parser.get_metadata(), metadata)
                parsed_time = self._add_file_to_lmdb(txn, parser, self.features, feature_hash, key_iterator, self.fragment_class, dataset_path)
                n_seconds += parsed_time
            type(self)._add_feature_hash_to_lmdb(txn, feature_hash)
            self._close_features(self.features)
            type(self)._add_features_to_lmdb(txn, self.features)


        # write metadata
        metadata_path = self.output_path / "metadata.yaml"
        n_chunks = key_iterator.current_idx
        with open(metadata_path, "w+") as f:
            yaml.safe_dump({
                "n_seconds": n_seconds,
                "n_chunks": n_chunks,
                "writer_class": type(self).__name__,
                "fragment_class": self.fragment_class.__name__, 
                "features": {self.get_feature_name(f): str(f) for f in self.features},
                "parser_class": self.parser.__name__,
                **self.metadata, 
                **metadata,
            }, f)
        extras_path = self.output_path / "code"
        os.makedirs(extras_path, exist_ok=True)
        fragment_class_name = self.fragment_class.__module__.split('.')[-1]

        # write additional files
        proto_path = Path(__file__).parent / ".." / "fragments" / "interfaces" / f"{fragment_class_name}.proto"
        if os.path.exists(proto_path):
            shutil.copyfile(proto_path, extras_path / proto_path.name)
        compiled_path = Path(__file__).parent / ".." / "fragments" / "compiled" / f"{fragment_class_name}_pb2.py"
        if os.path.exists(compiled_path):
            shutil.copyfile(compiled_path, extras_path / compiled_path.name)
        with open(self.output_path / "config.gin", "w+") as f:
            f.write(gin.config_str())

        env.close()
        

    @classmethod
    def open(cls, path, readonly=True, lock=False):
        return lmdb.open(str(path), lock=lock, readonly=readonly)

    @classmethod
    def get_feature_hash(cls, txn):
        return dill.loads(txn.get('feature_hash'.encode('utf-8'))) 

    @classmethod
    def iter_fragment_keys(cls, txn):
        for key in txn.cursor().iternext(values=False):
            if key in cls.non_buffer_keys:
                yield key
            
        # file_keys = txn.cursor().iternext(values=False)
        # try:
        #     idx = file_keys.index(b'feature_hash')
        #     idx = file_keys.index(b'features')
        #     del file_keys[idx]
        # except IndexError:
        #     pass
        # return file_keys

    @classmethod
    def iter_fragments(cls, txn, fragment_class):
        for key in txn.cursor().iternext(values=False):
            if key in cls.non_buffer_keys:
                yield key, fragment_class(txn.get(key))

    @classmethod
    def parse_from_path(cls, path):
        env = cls.open(path)
        fragment_class = get_fragment_class_from_path(path)
        with env.begin() as txn:
            feature_hash = cls.get_feature_hash(txn)
            iterator = cls.iter_fragments(txn, fragment_class)
        return iterator, feature_hash

    @classmethod
    def update(cls, 
        path: str, 
        features: List[AcidsDatasetFeature] | None = None,
        data: List[str | Path] | None = None, 
        check: bool = True, 
        overwrite: bool = False, 
        filters: List[str] | None = None,
        exclude: List[str] | None = None,
        dir_parser: Callable = audio_paths_from_dir,
        valid_exts: List[str] | None = None, 
        max_db_size: int = 100,
    ):
        path = Path(path).resolve()
        assert path.exists(), f"preprocessed dataset {path} does not seem to exist."
        features = features or []
        data = data or []
        assert len(features) + len(data) > 0, "update needs at least one operation to perform; here both features and data are empty"
        for i, d in enumerate(data):
            data[i] = Path(d).resolve()
            if not data[i].exists(): raise FileNotFoundError(f'folder {data[i]} does not seem to exist')

        # create env
        env = lmdb.open(str(path), map_size=max_db_size * 1024 ** 3)
        metadata = get_metadata_from_path(path)
        n_chunks = metadata['n_chunks']
        n_seconds = metadata['n_seconds']

        # parse features
        with env.begin(write=True) as txn:
            # parse features
            features = list(dill.loads(txn.get('features'.encode('utf-8'))).values()) + features
            fragment_class = get_fragment_class_from_path(path)
            feature_hash = cls.get_feature_hash(txn)

            if len(features) > 0:
                for key in cls.iter_fragment_keys(txn):
                    fragment = fragment_class(txn.get(key))
                    cls._extract_features(fragment, features, key, feature_hash, overwrite=overwrite)

            # then, add additional data if needed
            parser_class = get_parser_class_from_path(path)
            files = []
            filters = filters or []
            exclude = exclude or []
            if len(data) > 0:
                for data_path in data:
                    files.append((data_path, 
                                dir_parser(data_path, valid_exts = valid_exts, flt=filters, exclude=exclude)))
                
                new_file_list = set(sum([d[1] for d in files], []))
                original_file_list = set(feature_hash['original_path'].keys())
                common_files = new_file_list.intersection(original_file_list)

                if not overwrite: 
                    new_file_list.difference_update(common_files)
                
                key_iterator = iter(KeyIterator(n_chunks))
                status_bar = tqdm.tqdm(total=len(new_file_list), desc="parsing additional files...")
                status_bar.display()
                
                for data_path, current_files in files:
                    data_path = data_path.resolve().absolute()
                    for current_file in current_files:
                        status_bar.update(1)
                        parser = parser_class(current_file, 
                                              features=features, 
                                              dataset_path = data_path)
                        parsed_time = cls._add_file_to_lmdb(txn, parser, features, feature_hash, key_iterator, fragment_class, data_path)
                        n_seconds += parsed_time
                n_chunks = key_iterator.current_idx
                metadata.update(n_seconds=n_seconds, n_chunks=n_chunks, features={cls.get_feature_name(f): str(f) for f in features})

            cls._add_feature_hash_to_lmdb(txn, feature_hash)
            cls._close_features(features)
            cls._add_features_to_lmdb(txn, features)

            # write metadata
            metadata_path = path / "metadata.yaml"
            with open(metadata_path, "w+") as f:
                yaml.safe_dump(metadata, f)

        env.close()
        cls._close_features(features)


