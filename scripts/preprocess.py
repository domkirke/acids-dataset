import sys
import os
from absl import flags, app
from pathlib import Path
import gin 
try:
    import acids_dataset
except ImportError as e: 
    sys.path.append(str((Path(__file__).parent / '..').resolve()))
    import acids_dataset
from acids_dataset import preprocess_dataset

FLAGS = flags.FLAGS

flags.DEFINE_string('path', None, help="dataset path", required=True)
flags.DEFINE_multi_string('config', "default.gin", help="config files")
flags.DEFINE_string('out', None, help="parsed dataset location")
flags.DEFINE_integer('num_signal', 131072, help="chunk size of audio fragments")
flags.DEFINE_integer('sample_rate', 44100, help="sample rate")
flags.DEFINE_integer('channels', 1, help="number of audio channels")
flags.DEFINE_boolean('check', True, help="has interactive mode for data checking.")

def main(argv):
    preprocess_dataset(
        FLAGS.path, 
        out = FLAGS.out, 
        configs = FLAGS.config, 
        check = FLAGS.check, 
        sample_rate=FLAGS.sample_rate, 
        channels = FLAGS.channels
    )


if __name__ == "__main__":
    app.run(main)