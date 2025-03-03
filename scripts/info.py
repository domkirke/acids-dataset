import sys
import re
import yaml
import os
from absl import flags, app
from pathlib import Path
try:
    import acids_dataset
except ImportError as e: 
    sys.path.append(str((Path(__file__).parent / '..').resolve()))
    import acids_dataset
from acids_dataset import get_fragment_class

FLAGS = flags.FLAGS
flags.DEFINE_string('path', None, 'dataset path', required=True)
flags.DEFINE_boolean('files', False, 'list files within dataset')
flags.DEFINE_boolean('check_metadata', False, 'check metadata discrepencies in dataset')


def main(argv):
    dataset_path = Path(FLAGS.path)
    metadata_path = dataset_path / "metadata.yaml"
    with open(metadata_path, "r") as f:
        metadata = yaml.safe_load(f)
    print('dataset at path %s :'%dataset_path)
    for k, v in metadata.items():
        print(f"{k}: {v}")
    if FLAGS.files:
        writer_class = getattr(acids_dataset.datasets, metadata.get('wrtier_class', 'LMDBWriter'))
        env = writer_class.open(str(dataset_path))
        with env.begin() as txn:
            feature_hash = writer_class.get_feature_hash(txn)
            print('files : ')
            metadata_keys = metadata['features']
            for i, (f, idx) in enumerate(feature_hash['original_path'].items()):
                print(f'{re.sub(f"^{dataset_path}", "", f)}: {len(idx)} chunks')
            if FLAGS.check_metadata:
                file_keys = writer_class.get_file_ids(txn)
                fragment_class = get_fragment_class(metadata['fragment_class'])
                missing_metadata = {}
                for key in file_keys:
                    ae = fragment_class(txn.get(key))
                    for k in metadata_keys:
                        try:
                            ae.get_buffer(k)
                        except KeyError:
                            audio_path = getattr(ae, "audio_path", None)
                            if audio_path is None:
                                missing_metadata[key] = missing_metadata.get(key, []).append(k)
                            else:
                                missing_metadata[audio_path] = missing_metadata.get(audio_path, []).append(k)
        if len(missing_metadata) == 0:
            print('-------\nNo metadata discrepencies.')
        else:
            print('-------\n[WARNING] Found metadata discrepencies :')
            for k, v in missing_metadata.items():
                print(f"{k}: {v} missing")

    
if __name__ == "__main__":
    app.run(main)