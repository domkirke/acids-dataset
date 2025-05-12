# acids-dataset

**Warning** `acids-dataset` is still experimental, do not hesitate to add issues! 

`acids-dataset` is a preprocessing package for audio data and metadata, mostly used by [RAVE](http://github.com/acids-ircam/RAVE) and [AFTER](http://github.com/acids-ircam/AFTER) but opened for custom use. Built open [lmdb](https://openldap.org/), it leverages the pre-processing step of data parsing required by audio generative models to extract metadata and audio features that can be accessed and used during training. It brings : 
- Conveninent pre-processing pipelines allowing to easily select / discard files, extract metadata with features of from filenames, and metadata hashing
- Data augmentations tailored for audio with probablities
- Powerful data loaders for supervised / self-supervised learning with additional features (balancing, positive / negative examples, curriculum learning)

# Installation
To install acids-dataset, just install it through pip.

```bash
pip install acids-dataset
```

Some dependencies also have to be installed manually for some features. 
- `F0` and `Pitch` has `yin` and `pyin` methods installed  through `librosa`, but can also have `crepe`, `pesto`, `world`, or `praat` by installing libraries (respectively) [torchcrepe](https://github.com/maxrmorrison/torchcrepe), [pesto-pitch](https://github.com/SonyCSLParis/pesto), [pyworld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder), or [praat-parselmouth](https://parselmouth.readthedocs.io/en/stable/index.html#).

# Usage

## Preprocessing

### Simple parsing
`acids-dataset` is available as a command-line tool to easily pre-process a dataset path. For example, you can parse a dataset in 2048-sized chunks with the following command :

```bash
acids-dataset preprocess --path /path/to/your/dataset --out /target/path/for/preprocessed --chunk_length 2048
```

the `--config` argument allows you to provide additional configurations for dataset parsing. Some configurations are provided in `acids_dataset/configs`, but you can also link your own by placing them into a folder `ad_configs` in the current execution directory. For exemple, you can parse a dataset for [RAVE](http://github.com/acids-ircam/RAVE) with :

```bash
acids-dataset preprocess --path /path/to/your/dataset --out /target/path/for/preprocessed --config rave 
```

or, similarly, for [AFTER](http://github.com/acids-ircam/RAVE) :

```bash
acids-dataset preprocess --path /path/to/your/dataset --out /target/path/for/preprocessed --config after 
```


| Name          | Content                                                                                                  |
| ------------- | -------------------------------------------------------------------------------------------------------- |
| `default.gin` | base configuration for dataset. Do not forget to give `chunk_length`, as this is not provided!           |
| `rave.gin`    | base configuration for [RAVE](github.com/acids-ircam/RAVE)                                               |
| `after.gin`   | base configuration for [AFTER](github.com/acids-ircam/RAVE), adding `AfterMIDI` feature for MIDI parsing |


You can also parse several data folders to join them in a single dataset by providing several `--path` keywords :

```bash
acids-dataset preprocess --path /path/to/your/dataset1 --path /path/to/your/dataset2 --out /target/path/for/preprocessed 
```

`acids-dataset` will keep track of the original folder for each data chunk, so no information will be lost along compilation!

##### Important : the `max_db_size` is very important, as it allows you to set the maximum size of the compiled library. This is imposed by LMDB, and may be a little tricky to deal with. For little datasets (like <20Gb), you may want to add the `--compact` option that will first make a "full" compiled version of the database in a temporary folder, and then compact it in the target directory. For big datasets, be careful on how much data you allocate!


### Data selection

`acids-dataset` also allows to very easily select and exclude files inside a given folder. For exemple, here `--filter` allows to only select the files contained in a sub-dataset of the folder provided with `path`, and `--exclude` to exclude all .opus file.

```bash
acids-dataset preprocess --path /path/to/your/dataset --out /target/path/for/preprocessed --filter "**/*" --exclude "*.opus"
```

that would only retain audio files in subfolders of the root, and exclude all .opus files. All the preprocess options are available by executing the command with the `--help` tag :

```bash
USAGE: acids_dataset/preprocess.py [flags]
flags:

acids_dataset/preprocess.py:
  --channels: number of audio channels
    (default: '1')
    (an integer)
  --[no]check: has interactive mode for data checking.
    (default: 'true')
  --chunk_length: number of samples per chunk
    (an integer)
  --[no]compact: create compact version of lmdb database
    (default: 'false')
  --config: dataset config
    (default: 'default.gin')
  --device: device for feature computation
    (default: 'cpu')
  --exclude: wildcard to exclude target files;
    repeat this option to specify a list of values
    (default: '[]')
  --feature: config files;
    repeat this option to specify a list of values
    (default: '[]')
  --filter: wildcard to filter target files;
    repeat this option to specify a list of values
    (default: '[]')
  --[no]force: force dataset preprocessing if folder already exists
    (default: 'false')
  --log: log file
  --max_db_size: maximum database size
    (default: '100.0')
    (a number)
  --meta_regexp: additional regexp for metadata parsing;
    repeat this option to specify a list of values
    (default: '[]')
  --out: parsed dataset location
  --override: additional overridings for configs;
    repeat this option to specify a list of values
    (default: '[]')
  --path: dataset path;
    repeat this option to specify a list of values
  --sample_rate: sample rate
    (default: '44100')
    (an integer)
  --[no]waveform: no waveform parsing
    (default: 'true')
```

You can also declaratively use this package in Python using the `preprocess_dataset` function of the `acids_dataset` package:

```python
import acids_dataset

acids_dataset.preprocess_dataset(
        dataset_path
        out = out_path, 
        configs = config_list, 
        check = False,
        chunk_length = 131072,
        sample_rate = 44100,
        channels = 1
        flt = [], 
        exclude=[]
    )
```

### Embedding features

`acids_dataset` has specific configuration files to embed metadata (like audio descriptors) in the database, to make them accessible during training. For example, to add loudness and mel profiles for each data chunk, you may add the `mel` and `loudness` configs, that are contained in `acids_dataset/configs/features`

```bash
acids-dataset preprocess --path /path/to/your/dataset --out /target/path/for/preprocessed --feature loudness --config mel
```

of course, you can customize these features ; please see the section [customize](#customize) below.

Among all the available features, the `RegexpFeature` is a special feature that is accessible in command-line. By providing `--meta_regexp` patterns, that can be understood as glob patterns with placeholders contained withing double curvy braces. For example, let's imagine (like Slakh) that your dataset is organised as folows :

```text
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

## Updating a compiled database

The `update` commamnd allows to add features / files to a compiled dataset.

```bash
acids-dataset update --help                                                 15:51

      USAGE: acids_dataset/update.py [flags]
flags:

acids_dataset/update.py:
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

For example, if you want to update a pre-processed dataset with the `loudness` feature and add some more data contained in `path/to/additional/data`, you can use

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

##### By default, `update` will open the database with 1.1 * size of the database. If you plan to add havy audio / features, do not forget to also update `--max_db_size`!!


### Get information

You can quickly monitor the content of a parsed metadata using the `info` metadata : 
```bash
    USAGE: acids_dataset/info.py [flags]
flags:

acids_dataset/info.py:
  --[no]check_metadata: check metadata discrepencies in dataset
    (default: 'false')
  --[no]files: list files within dataset
    (default: 'false')
  --path: dataset path
```

You can also add the `--files` to list all the parsed audio files, or the `--check_metadata` flag to check missing metadata. See the full available options by running `acids-dataset info --help`. For example, it should provide you informations as follows :

```bash
channels: 1
chunk_length: 1.0
exclude: []
features: {'f0': 'F0(mode=yin, sr=44100)', 'pitch': 'Pitch(mode=yin, sr=44100)'}
filters: []
format: int16
fragment_class: AcidsFragment
hop_length: 2.0
import_backend: 0
loudness_threshold: None
max_db_size: 5.1000000000000005
n_chunks: 40
n_seconds: 40.0
pad_mode: 0
parser_class: RawParser
sr: 44100
writer_class: LMDBWriter
```

# DataLoaders, partitioning, augmentations

`acids_dataset` brings adaptive and convenient data augmentations pipelines for audio compatible with `gin` and `torch.utils.Dataset`. 

## Indexing, partitioning, and sampling interface

`acids_dataset` provides very powerful features to index / transform complex datasets, and provide easy way to perform multi-view indexing in a very simple way.

```python
from acids_dataset import SimpleDataset, transforms as adt

data_path = "..."
dataset = SimpleDataset(data_path)
dataset[0] # will provide automatically the "waveform" field as a single tensor
dataset.get(0) # __getitem__ is a proxy for .get function, that's actually much more powerful
dataset.get(0, "(waveform,mel)") # => returns both waveform and mel fields as a tuple
dataset.get(0, "{waveform,mel}") # => returns both waveform and mel fields as a dictionary with waveform and mel keys
dataset.get(0, "{waveform,mel->z}") # => same, but put mel vector in field "z"
dataset.get(0, "{waveform[0]->x_l, waveform[1]->x_r}") # => puts first channel in x_l field, second in x_r field
dataset.get(0, "(waveform,pitch)", [[], adt.OneHot(12)]) # => you can add also transforms (see below)
dataset.get(0, "{waveform,pitch}", {'pitch': dt.OneHot(12)}) # => same with dict outputs

# you can also specify during initialisation
dataset = SimpleDataset(data_path, output_pattern="{waveform->x,pitch->y}, transforms={"x": [], "y": adt.OneHot(12)})
out = dataset[0] 
```

it also provides handy methods to retrive data from a given label :
```
note_a_data = dataset.query_from_features(..., pitch='A') # retrieve all the data with 0 pitch class
note_a_data = dataset.query_from_features(10, pitch='A') # retrieve 10 exemples with 0 pitch class
note_a_data = dataset.(10, pitch='A', original_path=['file1.wav', 'file2.wav]) # retrieve 10 exemples with 0 pitch class in given data files
```

## Data augmentation

Data augmentations can work with both `np.ndarray` and `torch.Tensor`, and may be composed through the `transforms.Compose` that override usual list methods (`append` , `extend`, etc...). `acids-dataset` implements its own augmentations, and has a backend for the [audiomentations](https://github.com/iver56/audiomentations) package.


Stochastic options of the base class are very powerful to develop versatile augmentation pipelines. For example :

```python
from acids_dataset import transforms as atf

pipeline = atf.Compose([
  atf.Stretch(131072, p=0.2), # Stretch is always non-batchwise, as it crops to a target size
  atf.Gain(p=0.8), 
  atf.Compress(p=0.5), 
  atf.Mute(p=0.1), 
])
```

performs random pitch shifting with probability 20%, random gain with probability 80%, random compression with probability 50%, and mutes random batches with probability 0.10% (very useful to force models to learn silence). 

For a full table of available transfroms, see at the end of `README.md`. 

|
<a href="#customize"></a>

# Extending `acids-dataset`

## Customizing features

If you want to customize a feature, you can copy a feature configuration file (see `acids_dataset/configs/features/`) and place it a `ad_dataset` subfolder of your current working directory. It is recommended to give it a custom name to avoid naming problems. For examples, if you want a custom CQT with 257 bins (instead of the default 84), copy `acids_dataset/configs/features/cqt.gin` to `$(pwd)/ad_configs/cqt_rene.gin` (if your name is René), and modify the amount of bins :

```yaml
CQT:
  name=cqt_rene
  n_bins=257
  device=%DEVICE
  sr = %SAMPLE_RATE

features.parse_features:
  features = @features.CQT
```

and update your database using `acids-dataset update --path ... --feature cqt_rene`. The feature will be added as `cqt_rene`, as the name in the configuration as been specified. 


## Implementing features
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

## Customizing augmentations

Similarly to features, you can copy the `compress.gin` configuration as `thierry_compress.gin` (if your name is Thierry), and customize the inner parameters :

```yaml
Compress:
  name = "thierry_compress"
  threshold = -30
  amp_range = [-20, 10]
  attack = 0.1
  release = 0.3
  prob = 1.0
  limit = False
  sr = %SAMPLE_RATE

transforms.parse_transform:
    transform = @transforms.Compress
```

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

# Full augmentation table 

| Name                  | Config name                 | Description                                                                 |
| --------------------- | --------------------------- | --------------------------------------------------------------------------- |
| AddColorNoise         | `addcolornoise.gin`         | Pitch extraction based on BasicPitch                                        |
| AddGaussianNoise      | `addgaussiannoise.gin`      | Add gaussian noise to audio signal                                          |
| AddNoise              | `addnoise.gin`              | Add uniform noise to audio signal                                           |
| AddShortNoises        | `addshortnoises.gin`        | Add short bursts of noise to the signal                                     |
| AirAbsorption         | `airabsorption.gin`         | Add air absorption.                                                         |
| Aliasing              | `aliasing.gin`              | Aliases audio signal.                                                       |
| ApplyImpulseResponse  | `applyimpulseresponse.gin`  | Applies impulse response (needs audio files as first argument).             |
| BandPassFilter        | `bandpassfilter.gin`        | Random band-pass filter.                                                    |
| BasicPitch            | `basicpitch.gin`            | Pitch extraction based on BasicPitch                                        |
| Bitcrush              | `bitcrush.gin`              | Adds random bit-crushing (bit & sr reduction)                               |
| Clip                  | `clip.gin`                  | Clips incomping signal.                                                     |
| ClippingDistortion    | `clippingdistortion.gin`    | Adds clipping distortion.                                                   |
| Compress              | `compress.gin`              | Compression with random parameters.                                         |
| Crop                  | `crop.gin`                  | Random cropping to fixed chunk size.                                        |
| Dequantize            | `dequantize.gin`            | Dequantize incoming signal.                                                 |
| Derivator             | `derivator.gin`             | Derivates incoming signal.                                                  |
| FrequencyMasking      | `frequencymasking.gin`      | Randomly masks frequency bands of input.                                    |
| Gain                  | `gain.gin`                  | Applies random gain to the input.                                           |
| GainTransition        | `gaintransition.gin`        | Applies random gain ramps to the signal.                                    |
| HighPassFilter        | `highpassfilter.gin`        | Applies random high-pass filtering.                                         |
| HighShelfFilter       | `highshelffilter.gin`       | Applies random high-shelf filtering.                                        |
| Integrator            | `integrator.gin`            | Integrates incoming signal.                                                 |
| Lambda                | `lambda.gin`                | Performs a lambda function to the signal (needes lambda as first argument.) |
| LoudnessNormalization | `loudnessnormalization.gin` | Normalizes loundess according to IFRS standard.                             |
| LowPassFilter         | `lowpassfilter.gin`         | Applies random low-pass filtering.                                          |
| LowShelfFilter        | `lowshelffilter.gin`        | Applies random low-shelf filtering.                                         |
| Mp3Compression        | `mp3compression.gin`        | Applies random mp3 compression.                                             |
| Mute                  | `mute.gin`                  | Mutes incoming incoming signal.                                             |
| Normalize             | `normalize.gin`             | Normalizes incoming signal.                                                 |
| PeakingFilter         | `peakingfilter.gin`         | Applies random peak filtering, superposed to data.                          |
| PhaseMangle           | `phasemangle.gin`           | shuffle phase with an all-pass filter                                       |
| PolarityInversion     | `polarityinversion.gin`     | Randomly inverses the waveform (*-1)                                        |
| PreEmphasis           | `preemphasis.gin`           | adds pre-emphasis to a signal.                                              |
| RandomDelay           | `randomdelay.gin`           | random short comb-filtered delays                                           |
| RandomDistort         | `randomdistort.gin`         | random Eqing and distortion of signal                                       |
| RandomEQ              | `randomeq.gin`              | random EQing of the signal                                                  |
| Resample              | `resample.gin`              | resampling of the signal.                                                   |
| Reverse               | `reverse.gin`               | randomly reverses temporally the signal.                                    |
| RoomSimulator         | `roomsimulator.gin`         | performs room reverberation simulation.                                     |
| SevenBandParametricEQ | `sevenbandparametriceq.gin` | Random seven-band parametric EQ.                                            |
| Shift                 | `shift.gin`                 | Random pitch-shifting of the signal.                                        |
| TanhDistortion        | `tanhdistortion.gin`        | Performs random tanh distortion to the signal.                              |
| TimeMask              | `timemask.gin`              | Randomly masks temporal zones of the signal with silence.                   |
| TimeStretch           | `timestretch.gin`           | Randomly time-stretches the incoming signal, and crop.                      |
| Stretch               | `stretch.gin`               | Time-shifting using polyphase-filtering .                                   |
