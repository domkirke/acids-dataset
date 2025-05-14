import sys
import re
import yaml
import os
from absl import flags, app
from pathlib import Path

import sys
import sys; sys.path.append(str(Path(__file__).parent.parent))
from acids_dataset import get_writer_class_from_path, get_metadata_from_path, get_fragment_class_from_path

def import_flags():
    flags.DEFINE_string('path', None, 'dataset path', required=True)
    flags.DEFINE_boolean('files', False, 'list files within dataset')
    flags.DEFINE_boolean('check_metadata', False, 'check metadata discrepencies in dataset')
    flags.DEFINE_multi_integer('idx', [], 'parses fragment with idx %s')
    flags.DEFINE_multi_string('key', [], 'parses fragment with key %s')

def main(argv):
    FLAGS = flags.FLAGS
    dataset_path = Path(FLAGS.path)
    writer_class = get_writer_class_from_path(FLAGS.path)
    metadata = get_metadata_from_path(FLAGS.path)
    for k, v in metadata.items():
        print(f"{k}: {v}")
    if FLAGS.files:
        env = writer_class.open(dataset_path)
        with env.begin() as txn:
            feature_hash = writer_class.get_feature_hash(txn)
            print('files : ')
            metadata_keys = metadata['features']
            for i, (f, idx) in enumerate(feature_hash['original_path'].items()):
                print(f'{re.sub(f"^{dataset_path}", "", f)}: {len(idx)} chunks')
            if FLAGS.check_metadata:
                missing_metadata = {}
                fragment_class = get_fragment_class_from_path(dataset_path)
                for key, ae in writer_class.iter_fragments(txn, fragment_class):
                    for k in metadata_keys:
                        try:
                            ae.get_buffer(k)
                        except KeyError:
                            audio_path = getattr(ae, "audio_path", None)
                            if audio_path is None:
                                missing_metadata[key] = (missing_metadata.get(key) or []).append(k)
                            else:
                                missing_metadata[audio_path] = (missing_metadata.get(audio_path) or []).append(k)
                if len(missing_metadata) == 0:
                    print('-------\nNo metadata discrepencies.')
                else:
                    print('-------\n[WARNING] Found metadata discrepencies :')
                    for k, v in missing_metadata.items():
                        print(f"{k}: {v} missing")
    if len(FLAGS.idx) > 0 or len(FLAGS.key) > 0:
        loader = writer_class.loader(FLAGS.path)
        for i in FLAGS.idx : 
            print('\n--INDEX %d :'%i)
            try: 
                fg = loader[i]
                print(fg.description)
            except IndexError:
                print('index %d not in dataset')
        for k in FLAGS.key:
            print('\n--KEY %s : '%k)
            try:
                fg = loader[k]
                print(fg.description)
            except IndexError:
                print('key %s not in dataset')

    
if __name__ == "__main__":
    import_flags()
    app.run(main)