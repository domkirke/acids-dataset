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
flags.DEFINE_string('out', None, help="parsed dataset location")
flags.DEFINE_string('config', "default.gin", help="dataset config")
flags.DEFINE_multi_string('feature', "default.gin", help="config files")
flags.DEFINE_multi_string('filter', [], help="wildcard to filter target files")
flags.DEFINE_multi_string('exclude', [], help="wildcard to exclude target files")
flags.DEFINE_multi_string('meta_regexp', [], help="additional regexp for metadata parsing")
flags.DEFINE_multi_string('override', [], help="additional overridings for configs")
flags.DEFINE_integer('num_signal', 131072, help="chunk size of audio fragments")
flags.DEFINE_integer('sample_rate', 44100, help="sample rate")
flags.DEFINE_integer('channels', 1, help="number of audio channels")
flags.DEFINE_boolean('check', True, help="has interactive mode for data checking.")
flags.DEFINE_boolean('force', False, help="force dataset preprocessing if folder already exists")
flags.DEFINE_boolean('waveform', True, help="no waveform parsing")



def main(argv):
    preprocess_dataset(
        FLAGS.path, 
        out = FLAGS.out, 
        config = FLAGS.config, 
        features = FLAGS.feature,
        check = FLAGS.check, 
        sample_rate=FLAGS.sample_rate, 
        channels = FLAGS.channels, 
        flt=FLAGS.filter, 
        exclude=FLAGS.exclude,
        meta_regexp=FLAGS.meta_regexp,
        force=FLAGS.force,
        waveform=FLAGS.waveform, 
        override=FLAGS.override,
    )


if __name__ == "__main__":
    app.run(main)