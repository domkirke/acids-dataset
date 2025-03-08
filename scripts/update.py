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
from acids_dataset import update_dataset

FLAGS = flags.FLAGS
flags.DEFINE_string('path', None, 'dataset path', required=True)
flags.DEFINE_multi_string('data', [], help='add audio files to the target dataset.')
flags.DEFINE_multi_string('feature', [], help='add features to the target dataset.')
flags.DEFINE_multi_string('filter', [], help="wildcard to filter target files")
flags.DEFINE_multi_string('exclude', [], help="wildcard to exclude target files")
flags.DEFINE_multi_string('meta_regexp', [], 'parses additional blob wildcards as features.')
flags.DEFINE_multi_string('override', [], 'add audio files to the target dataset.')
flags.DEFINE_boolean('overwrite', False, help="recomputes the feature if already present in the dataset, and overwrites existing files")
flags.DEFINE_boolean('check', True, help="recomputes the feature if already present in the dataset")


def main(argv):
    update_dataset(
        FLAGS.path, 
        features=FLAGS.feature,
        data = FLAGS.data, 
        overwrite = FLAGS.overwrite, 
        check = FLAGS.check
    )
        

if __name__ == "__main__":
    app.run(main)