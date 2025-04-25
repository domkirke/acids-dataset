# acids-dataset

`acids-dataset` is a preprocessing package for audio data and metadata, mostly used by [RAVE](http://github.com/acids-ircam/RAVE) and [AFTER](http://github.com/acids-ircam/AFTER) but opened for custom use. Built open [lmdb](https://openldap.org/), it leverages the pre-processing step of data parsing required by audio generative models to extract metadata and audio features that can be accessed and used during training. It brings : 
- Conveninent pre-processing pipelines allowing to easily select / discard files, extract metadata with features of from filenames, and metadata hashing
- Data augmentations tailored for audio with probablities
- Powerful data loaders for supervised / self-supervised learning with additional features (balancing, positive / negative examples, curriculum learning)

# Installation
To install acids-dataset, just install it through pip.

```bash
pip install acids-dataset
```

# Usage

## Preprocessing

`acids-dataset` is available as a command-line tool to easily pre-process a dataset path. For example, simply parse a dataset for [after](http://github.com/acids-ircam/AFTER) with 
```bash
acids-dataset preprocess --path /path/to/your/dataset --out /target/path/for/preprocessed --config after
```
where here `after` is like a preset to automatically embed the metadata required by AFTER (see table below for a comprehensive list of available). You can also provide a list of filters / excluding filters to select the parsed files : 
```bash
acids-dataset preprocess --path /path/to/your/dataset --out /target/path/for/preprocessed --filter "**/*" --exclude "*.opus"
```
that would only retain audio files in subfolders of the root, and exclude all .opus files. All the preprocess options are available by executing the command with the `--help` tag : 

```bash
       USAGE: scripts/preprocess.py [flags]
flags:

scripts/preprocess.py:
  --channels: number of audio channels
    (default: '1')
    (an integer)
  --[no]check: has interactive mode for data checking.
    (default: 'true')
  --config: config files;
    repeat this option to specify a list of values
    (default: "['default.gin']")
  --exclude: wildcard to exclude target files;
    repeat this option to specify a list of values
    (default: '[]')
  --filter: wildcard to filter target files;
    repeat this option to specify a list of values
    (default: '[]')
  --num_signal: chunk size of audio fragments
    (default: '131072')
    (an integer)
  --out: parsed dataset location
  --path: dataset path
  --sample_rate: sample rate
    (default: '44100')
    (an integer)
```

You can also declaratively use this package in Python using the `preprocess_dataset` function of the `acids_dataset` package:
```python
import acids_dataset

acids_dataset.preprocess_dataset(
        dataset_path
        out = out_path, 
        configs = config_list, 
        check = False,
        sample_rate = 44100,
        channels = 1
        flt = [], 
        exclude=[]
    )
```


### Embedding features
`acids_dataset` has specific configuration files to embed metadata (like audio descriptors) in the database, to make them accessible during training. For example, to add loudness and mel profiles for each data chunk, you may add the `mel` and `loudness` configs: 
```bash
acids-dataset preprocess --path /path/to/your/dataset --out /target/path/for/preprocessed --config features/loudness --config features/mel
```
of course, you can customize these features ; please see the section [customize](#customize) below. Features can be conveniently extracted from the file names by providing `--meta_regexp` patterns, that can be understood as glob patterns with placeholders contained withing double curvy braces. For example, let's imagine (like Slakh) that your dataset is organised as folows :
```
dataset/
  Track00001/
    mix.flac
    stems/
      S01.flac
      S02.flac
      ...
  Track00002/
    mix.flac
    stems/
      S01.flac
      S02.flac
```
you can extract respectively the track ids and the instrument ids into `id` and `inst` features by running :
```bash
acids-dataset preprocess --path /path/to/your/dataset --exclude "**/mix.flac" --meta_regexp "Track{{id}}/stems/S{{inst}}.flac" 
```

### Updating a dataset 
Even if a dataset is preprocessed, it can be embedded additional features and additional data with the `update` command : 
```bash 
acids-dataset update --help                                                 15:51

       USAGE: scripts.update [flags]
flags:

scripts.update:
  --[no]check: recomputes the feature if already present in the dataset
    (default: 'true')
  --data: add audio files to the target dataset.;
    repeat this option to specify a list of values
    (default: '[]')
  --exclude: wildcard to exclude target files;
    repeat this option to specify a list of values
    (default: '[]')
  --feature: add features to the target dataset.;
    repeat this option to specify a list of values
    (default: '[]')
  --filter: wildcard to filter target files;
    repeat this option to specify a list of values
    (default: '[]')
  --meta_regexp: parses additional blob wildcards as features.;
    repeat this option to specify a list of values
    (default: '[]')
  --override: add audio files to the target dataset.;
    repeat this option to specify a list of values
    (default: '[]')
  --[no]overwrite: recomputes the feature if already present in the dataset, and overwrites existing files
    (default: 'false')
  --path: dataset path
```

For example, if you want to update a pre-processed dataset with the `loudness` feature and add some more data, you can use
```bash
acids-dataset update --path path/to/preprocessed --data path/to/additional/data --feature features/loudness
```

or, alternatively, use the python high-level function `update`
```python
from acids_dataset import update
from acids_dataset.features import Loudness

update_dataset(
  "path/to/preprocessed", 
  data=["path/to/additional/data"],
  features=[Loudness()]
)
```


### Get information
You can quickly monitor the content of a parsed metadata using the `info` metadata : 
```bash
acids-dataset info --path /path/to/preprocessed
```
You can also add the `--files` to list all the parsed audio files, or the `--check_metadata` flag to check missing metadata. See the full available options by running `acids-dataset info --help`.

### Available configurations

| Name                    | Content                                                                                                  |
| ----------------------- | -------------------------------------------------------------------------------------------------------- |
| `default.gin`           | base configuration for dataset.                                                                          |
| `rave.gin`              | base configuration for [RAVE](github.com/acids-ircam/RAVE)                                               |
| `after.gin`             | base configuration for [AFTER](github.com/acids-ircam/RAVE), adding `AfterMIDI` feature for MIDI parsing |
| `features/mel.gin`      | adding mel spectrogram to buffer.                                                                        |
| `features/loudness.gin` | adding loudness to buffer.                                                                               |
| `features/midi.gin`     | adding midi information to buffer.                                                                       |

<a href="#customize"></a>

## Data augmentations

`acids_dataset` brings adaptive and convenient data augmentations pipelines for audio compatible with gin and datasets. Data augmentations can work with both `np.ndarray` and `torch.Tensor`, and may be composed through the `transforms.Compose` that override usual list methods (`append` , `extend`, etc...).

| Name             | Config name            | Description                                  |
| ---------------- | ---------------------- | -------------------------------------------- |
| BasicPitch       | `basicpitch.gin`       | Pitch extraction based on BasicPitch         |
| Compess          | `compress.gin`         | Compression with random parameters.          |
| Crop             | ` crop.gin`            | Random cropping to fixed chunk size.         |
| Dequantize       | `dequantize.gin`       | Dequantize incoming signal.                  |
| Derivator        | `derivator.gin`        | Derivates incoming signal.                   |
| FrequencyMasking | `frequencymasking.gin` | Randomly masks frequency bands of input.     |
| Gain             | `gain.gin`             | Applies random gain to the input.            |
| Integrator       | `integrator.gin`       | Integrates incoming signal.                  |
| Mute             | `mute.gin`             | Mutes incoming incoming signal.              |
| Normalize        | `normalize.gin`        | Normalizes incoming signal.                  |
| PhaseMangle      | `phasemangle.gin`      | shuffle phase with an all-pass filter        |
| PreEmphasis      | `preemphasis.gin`      | adds pre-emphasis to a signal.               |
| RandomDelay      | `randomdelay.gin`      | random short comb-filtered delays            |
| RandomDistort    | `randomdistort.gin`    | random Eqing and distortion of signal        |
| RandomEQ         | `randomeq.gin`         | random EQing of the signal                   |
| Resample         | `resample.gin`         | resampling of the signal.                    |
| Strech           | `stretch.gin`          | Pitch alteration using polyphase resampling` |

All augmentations derived from a base class `Transform`, that is initialised with the following arguments : 

- `sr (int)` : a sample rate (mandatory for most transforms)
- `p (float)` : augmentation probability between 0 and 1
- `batchwise (bool)`: if available, are probabilities applied for each batch, or for a whole batch (default : True)
- `name (str)`: an optional name (see below)

Stochastic options of the base class are very powerful to develop versatile augmentation pipelines. For example : 

```
from acids_dataset import transforms as atf

pipeline = atf.Compose([
  atf.Stretch(131072, p=0.2), # Stretch is always non-batchwise, as it crops to a target size
  atf.Gain(p=0.8), 
  atf.Compress(p=0.5), 
  atf.Mute(p=0.1), 
])
```
performs random pitch shifting with probability 20%, random gain with probability 80%, random compression with probability 50%, and mutes random batches with probability 0.10% (very useful to force models to learn silence.) 


## Data loaders

# Extending `acids-dataset`

### Implementing features
You can add your own features by subclassing the `AcidsDatasetFeature` object. 

```python
class AcidsDatasetFeature(object):
    denylist = []
    has_hash = False # (1) <---- see below to see how datasets are hashed
    def __init__(
            self, 
            name: Optional[str] = None,
            hash_from_feature: Optional[Callable] = None, 
        ):
        super(CustomFeature, self).__init__(name, hash_from_feature)

    @property
    def default_feature_name(self):
        """defines the default feature name, is not provided"""
        return type(self).__name__.lower()

    def pre_chunk_hook(self, path, audio, sr) -> None:
      """this is a hook to perform some optional operations before audio chunking."""
      pass

    def close(self):
        """if some features has side effects like buffering, empty buffers and delete files"""
        pass

    def from_file(self, path, start_pos, chunk_length):
      # extraction code here....
      return feature

    def from_audio(self, audio):
      # extraction code here...
      return feature

    def from_fragment(self, fragment, write: bool = None):
        """extract the data from fragment, and writes into it if write is True"""
        audio_data = fragment.get_audio("waveform")
        meta = self.from_audio(audio_data)
        # you can also access the original file
        metadata = fragment.get_metadata()
        meta = self.from_file(metadata['audio_path'], metadata['start_pos'], metadata['chunk_length'])
        if write: 
          fragment.put_array(self.feature_name, meta)
        return meta
```

The `AcidsDatasetFeature` is the base object for all audio features. One instance is created for one feature, and the object is called when : 
- An audio file is gonna be chunked, through a call to `pre_chunk_hook` (for some buffering, or some analysis that would require the whole file)
- During writing, when the audio chunks are written in the lmdb database. 

`AcidsDatasetFeature` also automatically fills up a hash during writing, allowing to get all the audio indexes belonging to a given hash. However, this is not automatic: this hash process is performed if :  
- the `has_hash` attribute is `True` (the feature has then to be hashable ; arrays, tensors, or lists, are typically non hashable.)
- a `hash_from_feature(self, ft)` callback is defined in the class, or provided at initialization (allowing to be gin configurable). If a callback is provided at initialization, it erases in any case the one defined (or not) in the class.


### Implementing transforms

A new transform can be done as follows : 

```python 
@gin.configurable(module="transforms")
class NewTransform(Transform):
    takes_as_input = Transform.input_types.torch
    allow_random = True

    def __init__(self, custom_param: int | None = None, **kwargs):
      super().__init__()

    def apply(self, x):
      # operations to x....
```

and... that's it! `apply` provides the main operating function for data processing, and is directly called by `Transform` if `p=None` or randomly called if `p>0`. 
The `allow_random` attributes if transform can be randomized. You can also overload the method `apply_random(self, x)` to override how the transform behaves when `p>0`.

