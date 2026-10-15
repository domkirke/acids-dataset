# acids-dataset

**Warning** `acids-dataset` is still experimental, do not hesitate to add issues! 

`acids-dataset` is a preprocessing package for audio data and metadata, mostly used by [RAVE](http://github.com/acids-ircam/RAVE) and [AFTER](http://github.com/acids-ircam/AFTER) but opened for custom use. Built on [lmdb](https://openldap.org/), it leverages the pre-processing step of data parsing required by audio generative models to extract metadata and audio features that can be accessed and used during training. It brings : 
- Convenient pre-processing pipelines allowing to easily select / discard files, extract metadata with features of from filenames, and metadata hashing
- Data augmentations tailored for audio with probabilities
- Powerful data loaders for supervised / self-supervised learning with additional features (partitioning, multi-class indexing)
- An *online* mode, that loads raw audio folders directly (without pre-processing) with the same interface

## Installation
To install acids-dataset, just install it through pip.

```bash
pip install acids-dataset
```

**Some dependencies also have to be installed manually for some features.**

- `F0` and `Pitch` has `yin` and `pyin` methods installed  through `librosa`, but can also have `crepe`, `pesto`, `world`, or `praat` by installing libraries (respectively) [torchcrepe](https://github.com/maxrmorrison/torchcrepe), [pesto-pitch](https://github.com/SonyCSLParis/pesto), [pyworld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder), or [praat-parselmouth](https://parselmouth.readthedocs.io/en/stable/index.html#).

Audio loading in online mode relies on [soundfile](https://github.com/bastibe/python-soundfile) (installed with `acids-dataset`), that allows sample-accurate partial loading of `wav`, `flac`, `aiff`, `ogg`, `opus` and `mp3` files. Other formats (`m4a`, `aac`...) are loaded through [torchcodec](https://github.com/pytorch/torchcodec) (FFmpeg-based, installed with recent versions of `torchaudio`).

## Usage

`acids-dataset` is designed to easily:

- parse an audio dataset, automatically dealing with sampling rate, channels, metadata...
- computed before-hand and dynamically a wide variety of features (pitch, loudness, audio descriptors, custom classes...)
- custom class balancing, partitioning, multi-dataset weighting
- handy data augmentation classes

There is two ways to use `acids-dataset` in your pipeline : either in *preprocessed* mode, where all the data is pre-processed in a compressed format with accompnying metadata, or in *online* mode, where the audio is minimally parsed in execution time and dynamically loaded in workers. Both modes share the same dataset interface, presented in [Datasets](#datasets) ; available features, augmentations and output formats are presented in [Features, augmentations and output format](#features-augmentations-and-output-format), and pre-processing in [Pre-processing](#pre-processing). Once you are familiar with both modes, [Presets](#presets) allow to share parsing settings between them.

# Datasets

`acids_dataset` brings adaptive and convenient data loading pipelines for audio compatible with `gin` and `torch.utils.Dataset`. 

## Loading a dataset

`AudioDataset` is the common interface of datasets : instantiating it with a path returns a `PreprocessedAudioDataset` if the path is a pre-processed dataset folder (see [Pre-processing](#pre-processing)), or an `OnlineAudioDataset` otherwise. Both share the same indexing, partitioning and loading interface.

```python
from acids_dataset.datasets import AudioDataset

dataset = AudioDataset("path/to/preprocessed")   # -> PreprocessedAudioDataset
dataset = AudioDataset("path/to/raw/audio")      # -> OnlineAudioDataset
dataset[0] # will provide automatically the "waveform" field as a single tensor
```

### Online mode

In online mode, audio files are parsed on the fly : there is no compilation step, and features are computed dynamically when data is loaded. Parsing parameters can be given directly, or with a [preset](#presets) (explicit arguments override the preset).

```python
from acids_dataset.datasets import AudioDataset
from acids_dataset.features import Loudness

dataset = AudioDataset(
  "path/to/your/dataset",            # or a list of folders
  sr = 44100,                        # target sampling rate (None to keep original ones)
  channels = 1,                      # target number of channels (None to keep original ones)
  chunk_length = 131072,             # in samples (int) or seconds (float)
  batch_by = "auto",                 # "files", "dataset" or "auto" (see Data loaders)
  pad_mode = "zero",                 # padding for chunks exceeding files ("zero", "repeat", "reflect", "replicate")
  features = [Loudness(), "mel"],    # computed dynamically ; objects or feature config names
  meta_regexp = ["**/{{instrument}}_*.wav"], # automatically create a "instrument" feature if audio file matches the regexp
  filters = [], 
  exclude = ["**/*.opus"]
)
loader = dataset.get_loader(batch_size=16, num_workers=4)
```

At initialization, audio files are scanned to retrieve their length, sampling rate, number of channels and codec, without decoding them ; path-based metadata (`meta_regexp`, see [Features](#features)) is also extracted. This information is cached in a `.ad` index file written in each dataset folder, such that next initializations only scan new or modified files (deleted files are removed from the index). Unreadable files are listed in `dataset.invalid_files` and are not scanned again until they are modified.

| Option        | Effect                                                                                     |
| ------------- | ------------------------------------------------------------------------------------------ |
| `index_dir`   | write index files as `<folder_name>.ad` in this folder (useful for read-only datasets)     |
| `rescan`      | ignore existing index files, and scan every file                                           |
| `write_index` | if `False`, do not write index files                                                       |

Several dataset folders can be given, as long as they have different names. Audio files are partially loaded (only the requested chunk is decoded), resampled to `sr`, and adapted to `channels`.

### Loading audio from files

Both modes give access to the audio files of the dataset, independently of dataset items : `dataset.files` lists the files (with their `audio_path`, `duration`, `sr` and `channels`), and audio or fragments can be loaded from a given file :

```python
wav = dataset.load_audio(file_idx, start=1.0, length=2.0)        # tensor (channels, samples), at the sampling rate of the dataset
fragment = dataset.get_fragment(file_idx, start=1.0, length=2.0)  # AcidsFragment, with waveform, metadata and features
fragment.get_array("loudness")
```

Features of fragments are computed on the loaded audio (the features of the dataset, or the ones given with `features=[...]`). For pre-processed datasets, audio is rebuilt from the chunks stored in the database : parts of files that were not stored (e.g. discarded chunks) are filled with zeros, and datasets pre-processed without waveform cannot load audio.

## Metadata

Dataset metadata is accessible through `dataset.metadata` (for pre-processed datasets, see also [Get information](#get-information)). In online mode, the list of scanned files with their header information and path-based metadata is available through `dataset.files`.

`acids_dataset` also provides handy methods to retrive data from a given label :
```python
note_a_data = dataset.query_from_features(..., pitch='A') # retrieve all the data with 0 pitch class
note_a_data = dataset.query_from_features(10, pitch='A') # retrieve 10 exemples with 0 pitch class
note_a_data = dataset.query_from_features(10, pitch='A', original_path=['file1.wav', 'file2.wav']) # retrieve 10 exemples with 0 pitch class in given data files
```

Features available for querying are hashed in `dataset.feature_hash` (`{feature: {value: indices}}`). In online mode, only path-based features (`original_path` and `meta_regexp` features) can be used for search and partitioning, as they are known without loading audio. `index_for_features` then returns file indices, and `query_from_features` returns chunks of the matching files.

## Data loaders

`get_loader` (available in both modes) returns a `torch.utils.data.DataLoader` with a random sampler (or a sequential one with `shuffle=False`). Arguments of `DataLoader` (`num_workers`, `drop_last`...) are given to the loader, other ones (`chunk_length`, `batch_by`, `pad_mode` in online mode) configure the dataset.

```python
loader = dataset.get_loader(batch_size=16, num_workers=4, drop_last=True)
loader = dataset.get_loader(batch_size=16, max_samples=10000, replacement=True) # fixed number of samples per epoch
```

The chunk length of pre-processed datasets is fixed at compilation : giving a different `chunk_length` to `get_loader` raises an error.

### Chunking strategy (online mode)

In online mode, dataset items are chunks of `chunk_length` samples, that can be given at initialization, or later with `configure` or `get_loader` (that return configured copies of the dataset). The way chunks are distributed in the dataset is set by `batch_by` :

| `batch_by`  | Items                                                                                                                                                                                                                                  |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `"files"`   | one item per file, taken at a random position of the file. Files shorter than `chunk_length` are padded with `pad_mode`. Each file has the same weight, whatever its duration.                                                         |
| `"dataset"` | each file is split in `round(duration / chunk_length)` slots (at least 1), and each item is taken at a random position in its slot. The dataset length is then about `total_duration / chunk_length`, and chunks never overlap two files. |
| `"auto"`    | `"dataset"` if the median file duration is longer than `auto_ratio * chunk_length` (a few long files, default `auto_ratio=2`), `"files"` otherwise (a lot of short files).                                                              |

As items are taken at random positions, the same index gives different chunks across epochs (random positions are seeded independently in each `DataLoader` worker). `dataset.chunk_position(index, random_offset=False)` gives the deterministic (file, start) position of an item.

```python
dataset = AudioDataset("path/to/your/dataset", sr=44100)
train_loader = dataset.get_loader(batch_size=16, chunk_length=1., batch_by="dataset", num_workers=4)
len(train_loader.dataset)                    # ~ total duration / 1 second
dataset.configure(chunk_length=2.).batch_by  # "auto" is resolved once chunk_length is known
```

If files have different sampling rates, `sr` must be given so that all the items of a batch have the same size.

## Partitioning

Datasets can be split into partitions, randomly or according to features (such that each feature value belongs to a single partition) :

```python
partitions = dataset.split({"train": 0.8, "valid": 0.2})                                   # random split
partitions = dataset.split({"train": 0.8, "valid": 0.2}, features=["original_path"])       # chunks of a file stay in the same partition
partitions = dataset.split({"train": 0.8, "valid": 0.2}, features=["instrument"], 
                           balance_cardinality=True, tolerance=0.1)                         # balance partition sizes
partitions = dataset.split({"train": 0.8, "valid": 0.2}, seed=0)                          # reproducible split
partitions = dataset.split({"train": 0.8, "valid": 0.2}, write="my_split")                 # write partition
partitions = dataset.load_partition("my_split")                                             # load it back
train_loader = partitions["train"].get_loader(batch_size=16)
```

- Partitions are drawn with the global random state, unless `seed` is given : partitions with the same seed are identical, including the order of their items. **For distributed training, give the same seed in every process** (otherwise each process gets different partitions, mixing training and validation data), or split and write partitions in one process and load them in the others.
- With `balance_cardinality=True`, feature values are distributed such that partition sizes follow the given ratios (with a warning if a partition exceeds `tolerance`). If ratios sum to 1, every item belongs to a partition ; otherwise, the remaining part is left out.
- If a `partition` feature exists (e.g. `--meta_regexp "**/{{partition}}/*.wav"` for `train/` and `valid/` subfolders), its values are used as partitions (disable with `use_meta_if_available=False`).
- Pre-processed datasets write partitions in `<dataset>/partitions/`. Online datasets partition files (chunks of a file always belong to the same partition), and write partitions as json files referring to audio paths in `.ad_partitions/` of the first dataset folder (or in the `partitions` subfolder of `index_dir`, or in `partition_dir` if given) ; partitions are robust to added / removed files.

## Multi-dataset

Several datasets can be combined with `CombinedAudioDataset`, with optional sampling weights for each dataset. Combined datasets must share the same output pattern and number of channels.

```python
import torch
from acids_dataset.datasets import AudioDataset, CombinedAudioDataset

dataset = CombinedAudioDataset([AudioDataset("path/to/dataset1"), AudioDataset("path/to/dataset2")], weights=[0.7, 0.3])
loader = torch.utils.data.DataLoader(dataset, batch_size=16, sampler=dataset.get_sampler())
```

Combined datasets can be partitioned with `split` and `load_partition` (each dataset is split with the same arguments, and partitions are combined datasets of the partitions of each dataset). You can also parse several data folders into a single pre-processed dataset (see [Simple parsing](#simple-parsing)), or give several folders to an online dataset.

## Class balancing and curriculum

Class balancing is currently available through balanced partitions (`balance_cardinality=True`, see [Partitioning](#partitioning)), dataset weights of `CombinedAudioDataset`, and feature queries (`query_from_features`). Class-balanced batch sampling and curriculum learning are work in progress.

# Features, augmentations and output format

## Output format

Outputs of datasets are set by an output pattern, that allows to index / transform complex datasets, and provide easy way to perform multi-view indexing in a very simple way.

```python
import torch
from acids_dataset import transforms as adt
from acids_dataset.datasets import AudioDataset

data_path = "..."
one_hot = lambda p: torch.nn.functional.one_hot(p.long() % 12, 12) # transforms can be any callable
dataset = AudioDataset(data_path)
dataset[0] # will provide automatically the "waveform" field as a single tensor
dataset.get(0) # __getitem__ is a proxy for .get function, that's actually much more powerful
dataset.get(0, "(waveform,mel)") # => returns both waveform and mel fields as a tuple
dataset.get(0, "{waveform,mel}") # => returns both waveform and mel fields as a dictionary with waveform and mel keys
dataset.get(0, "{waveform,mel->z}") # => same, but put mel vector in field "z"
dataset.get(0, "{waveform[0]->x_l, waveform[1]->x_r}") # => puts first channel in x_l field, second in x_r field
dataset.get(0, "(waveform,pitch)", [adt.Gain(), one_hot]) # => you can add also transforms (one per output, see below)
dataset.get(0, "{waveform,pitch}", {'pitch': one_hot}) # => same with dict outputs, only for given keys

# you can also specify during initialisation
dataset = AudioDataset(data_path, output_pattern="{waveform->x,pitch->y}", transforms={"y": one_hot})
out = dataset[0] 
```

Datasets can also be indexed with slices or sequences of indices, that return batches of items (collated as with a `torch.utils.data.DataLoader`, or as a list if items have different shapes) :

```python
dataset[0:4]                                    # => tensor of shape (4, channels, chunk_length)
dataset[-2:]                                    # => last two items
dataset[[0, 5, 7]]                              # => lists, tuples, numpy arrays or tensors of indices
dataset.get(slice(0, 4), "{waveform,mel}")      # => dictionary of batched waveform and mel
```

Output patterns access the buffers of data fragments (waveform and array features). Path-based metadata extracted with `meta_regexp` is stored as fragment metadata, and is not accessible through output patterns.

## Features

Features (like audio descriptors) can be embedded in pre-processed datasets to make them accessible during training (see [Embedding features](#embedding-features)), or computed dynamically in online mode by giving them to the dataset (`features=[...]`). Features can also be defined in [presets](#presets), to be used in both modes. Feature configurations are contained in `acids_dataset/configs/features`, and can be customized (see [Customizing features](#customizing-features)).

Among all the available features, the `RegexpFeature` is a special feature that extracts metadata from file paths. By providing `meta_regexp` patterns, that can be understood as glob patterns with placeholders contained withing double curvy braces. For example, let's imagine (like Slakh) that your dataset is organised as folows :

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

you can extract respectively the track ids and the instrument ids into `id` and `inst` features with the pattern `"Track{{id}}/stems/S{{inst}}.flac"` (see [Embedding features](#embedding-features) for the command line, or give `meta_regexp=[...]` to an online dataset).

### Pitch features

`yin` and `pyin` pitch estimations need at least one period of the lowest frequency `fmin` in their analysis frames. If `frame_length` is too short for `fmin` (for example, `frame_length=512` with `fmin="C2"` at 44.1kHz), frames are analyzed with a larger window (two periods of `fmin`), while keeping the same `hop_length` : the number of output frames is unchanged. Inputs shorter than `frame_length` are zero-padded to a single frame (zero padding biases pitch estimation much less than reflection or repetition, that create phase discontinuities).

## Data augmentation

Data augmentations can work with both `np.ndarray` and `torch.Tensor`, and may be composed through the `transforms.Compose` that override usual list methods (`append` , `extend`, etc...). `acids-dataset` implements its own augmentations, and has a backend for the [audiomentations](https://github.com/iver56/audiomentations) package.


Stochastic options of the base class are very powerful to develop versatile augmentation pipelines. For example :

```python
from acids_dataset import transforms as atf

pipeline = atf.Compose([
  atf.Stretch(131072, p=0.2), # Stretch is always non-batchwise, as output lengths depend on the random factor
  atf.Gain(p=0.8), 
  atf.Compress(p=0.5), 
  atf.Mute(p=0.1), 
])
```

performs random resampling (speed and pitch change) with probability 20%, random gain with probability 80%, random compression with probability 50%, and mutes random batches with probability 10% (very useful to force models to learn silence). 

For a full table of available transfroms, see at the [end](#full-augmentation-table) of `README.md`. 

## Extending `acids-dataset`

### Customizing features

If you want to customize a feature, you can copy a feature configuration file (see `acids_dataset/configs/features/`) and place it a `ad_configs` subfolder of your current working directory. It is recommended to give it a custom name to avoid naming problems. For examples, if you want a custom CQT with 257 bins (instead of the default 84), copy `acids_dataset/configs/features/cqt.gin` to `$(pwd)/ad_configs/cqt_rene.gin` (if your name is René), and modify the amount of bins :

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

The `AcidsDatasetFeature` is the base object for all audio features. One instance is created for one feature, and the object is called when : 
- An audio file is gonna be chunked, through a call to `pre_chunk_hook` (for some buffering, or some analysis that would require the whole file)
- During writing, when the audio chunks are written in the lmdb database. 
- In online mode, when an audio chunk is loaded.

`AcidsDatasetFeature` also automatically fills up a hash during writing, allowing to get all the audio indexes belonging to a given hash. However, this is not automatic: this hash process is performed if :  
- the `has_hash` attribute is `True` (the feature has then to be hashable ; arrays, tensors, or lists, are typically non hashable.)
- a `hash_from_feature(self, ft)` callback is defined in the class, or provided at initialization (allowing to be gin configurable). If a callback is provided at initialization, it erases in any case the one defined (or not) in the class.

### Customizing augmentations

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
      # operations to x....
```

and... that's it! `apply` provides the main operating function for data processing, and is directly called by `Transform` if `p=None` or randomly called if `p>0`.
The `allow_random` attributes if transform can be randomized. You can also overload the method `apply_random(self, x)` to override how the transform behaves when `p>0`.

# Pre-processing

## Simple parsing
`acids-dataset` is available as a command-line tool to easily pre-process a dataset path. For example, you can parse a dataset in 2048-sized chunks with the following command :

```bash
acids-dataset preprocess --path /path/to/your/dataset --out /target/path/for/preprocessed --chunk_length 2048
```

the `--preset` argument (or its alias `--config`) allows you to provide a [preset](#presets) for dataset parsing. Some presets are provided in `acids_dataset/configs`, but you can also link your own by placing them into a folder `ad_configs` in the current execution directory. For exemple, you can parse a dataset for [RAVE](http://github.com/acids-ircam/RAVE) with :

```bash
acids-dataset preprocess --path /path/to/your/dataset --out /target/path/for/preprocessed --preset rave 
```

or, similarly, for [AFTER](http://github.com/acids-ircam/AFTER) :

```bash
acids-dataset preprocess --path /path/to/your/dataset --out /target/path/for/preprocessed --preset after 
```


Options given in command line (`--chunk_length`, `--sample_rate`, `--channels`...) override the values of the preset (see [Presets](#presets)).

You can also parse several data folders to join them in a single dataset by providing several `--path` keywords :

```bash
acids-dataset preprocess --path /path/to/your/dataset1 --path /path/to/your/dataset2 --out /target/path/for/preprocessed 
```

`acids-dataset` will keep track of the original folder for each data chunk, so no information will be lost along compilation!

##### Important : the `max_db_size` is very important, as it allows you to set the maximum size of the compiled library. This is imposed by LMDB, and may be a little tricky to deal with. For little datasets (like <20Gb), you may want to add the `--compact` option that will first make a "full" compiled version of the database in a temporary folder, and then compact it in the target directory. For big datasets, be careful on how much data you allocate!

## Data selection

`acids-dataset` also allows to very easily select and exclude files inside a given folder. For exemple, here `--filter` allows to only select the files contained in a sub-dataset of the folder provided with `path`, and `--exclude` to exclude all .opus file.

```bash
acids-dataset preprocess --path /path/to/your/dataset --out /target/path/for/preprocessed --filter "**/*" --exclude "*.opus"
```

that would only retain audio files in subfolders of the root, and exclude all .opus files. All the preprocess options are available by executing the command with the `--help` tag :

```bash
USAGE: acids_dataset/preprocess.py [flags]
flags:

acids_dataset/preprocess.py:
  --channels: number of audio channels (overrides preset)
    (an integer)
  --[no]check: has interactive mode for data checking.
    (default: 'true')
  --chunk_length: number of samples per chunk (overrides preset)
    (an integer)
  --[no]compact: create compact version of lmdb database
    (default: 'false')
  --config: dataset preset (name or path to a .gin file)
    (default: 'default')
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
  --preset: an alias for --config
  --sample_rate: sample rate (overrides preset)
    (an integer)
  --[no]waveform: store waveform in the database (--nowaveform to only store features)
    (default: 'true')
```

You can also declaratively use this package in Python using the `preprocess_dataset` function of the `acids_dataset` package:

```python
import acids_dataset

acids_dataset.preprocess_dataset(
        dataset_path,
        out = out_path, 
        preset = "default", 
        check = False,
        chunk_length = 131072,
        sample_rate = 44100,
        channels = 1,
        flt = [], 
        exclude = [],
        waveform = True
    )
```

### Unreadable files

Files that cannot be read (corrupted, empty...) are skipped with a warning, and do not stop the parsing. Skipped files are listed in the `skipped_files` entry of the dataset metadata (see [Get information](#get-information)). If you prefer the parsing to stop at the first unreadable file, set the `ACIDSTRANSFORMS_RAISE_EXC_IF_WRITER_ERROR` environment variable to `1` (or set `acids_dataset.writers.RAISE_EXC_IF_WRITER_ERROR = True` in Python).

## Embedding features

`acids_dataset` has specific configuration files to embed metadata (like audio descriptors) in the database, to make them accessible during training. For example, to add loudness and mel profiles for each data chunk, you may add the `mel` and `loudness` configs, that are contained in `acids_dataset/configs/features`

```bash
acids-dataset preprocess --path /path/to/your/dataset --out /target/path/for/preprocessed --feature loudness --feature mel
```

of course, you can customize these features ; please see the section [Customizing features](#customizing-features).

The `RegexpFeature` (see [Features](#features)) is accessible in command-line with `--meta_regexp`. With the Slakh-like example, you can extract respectively the track ids and the instrument ids into `id` and `inst` features by running :

```bash
acids-dataset preprocess --path /path/to/your/dataset --exclude "**/mix.flac" --meta_regexp "Track{{id}}/stems/S{{inst}}.flac" 
```

### Storing features only

If you only need the features during training, you can avoid storing the audio with `--nowaveform` (or `waveform=False` in `preprocess_dataset`). Features are still computed from the audio during parsing, but the waveform is not written in the database, which can drastically reduce its size. This choice is recorded in the dataset metadata (`waveform` entry), such that audio files added later with `update` follow the same setting (datasets compiled before this option are considered to contain waveforms). Such datasets only give access to their features : use a feature as `output_pattern` when loading them (see [Output format](#output-format)).

```bash
acids-dataset preprocess --path /path/to/your/dataset --out /target/path/for/preprocessed --feature loudness --feature mel --nowaveform
```

## Updating a compiled database

The `update` commamnd allows to add features / files to a compiled dataset.

```bash
acids-dataset update --help

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

or, alternatively, use the python high-level function `update_dataset`

```python
from acids_dataset import update_dataset
from acids_dataset.features import Loudness

update_dataset(
  "path/to/preprocessed", 
  data=["path/to/additional/data"],
  features=[Loudness()]
)
```

##### By default, `update` will open the database with 1.1 * size of the database. If you plan to add havy audio / features, do not forget to also update `--max_db_size`!!

Audio files added with `--data` follow the `waveform` setting of the compiled dataset (see [Storing features only](#storing-features-only)), and unreadable files are skipped and appended to `skipped_files`.

## Computing embedding from a TorchScript model

A special script, `add_embedding`, provides a very handy way to compute embeddings from a scripted PyTorch model. For example, if you have a RAVE .ts file, you can populate the latent embeddings and  in your database through cuda (replace by `cpu` or `mps` in case) using:

```bash
acids-dataset add_embedding --path replace/path/to/db --module replace/path/to/ts --method encode --device cuda:0
```

An interesting feature of `add_embedding` is also the ability to compute "augmented“ embeddings, batching original data with transforms that can augment the data. This is typically very useful for self-supervised learning.  

```bash
acids-dataset add_embedding --path replace/path/to/db --module replace/path/to/ts --transform pitchshift --transform pitchshift --transform pitchshift --method encode --device cuda:0
```

Similarly, a python backend is provided

```python
from acids_dataset import add_embedding_to_dataset, transforms as adt

# load your model here...
model = Model()

# you can also directly populate with additional transforms. 
# different transforms will be batched during import. 

transforms = [adt.PitchShift() for i in range(4)]

add_embedding_to_dataset(
  path="path/to/dataset", 
  module=model, 
  module_path=None, # use path to directly load a .ts
  transforms=transforms, 
  module_sr=None, # acids_dataset will try to retrieve the sr attribute ; if not, provide it to avoid problems
  device="cpu"
)
```

## Get information

You can quickly monitor the content of a parsed metadata using the `info` metadata : 
```bash
    USAGE: acids_dataset/info.py [flags]
flags:

acids_dataset/info.py:
  --[no]check_metadata: check metadata discrepencies in dataset
    (default: 'false')
  --exclude: raw audio folders : wildcard to exclude target files;
    repeat this option to specify a list of values
    (default: '[]')
  --[no]files: list files within dataset
    (default: 'false')
  --filter: raw audio folders : wildcard to filter target files;
    repeat this option to specify a list of values
    (default: '[]')
  --idx: parses fragment with idx %s (file index for raw audio folders);
    repeat this option to specify a list of values
    (default: '[]')
    (an integer)
  --index_dir: raw audio folders : folder of index files (default : in the dataset folder)
  --key: parses fragment with key %s (audio path for raw audio folders);
    repeat this option to specify a list of values
    (default: '[]')
  --meta_regexp: raw audio folders : regexp for metadata parsing;
    repeat this option to specify a list of values
    (default: '[]')
  --path: dataset path (pre-processed dataset, or raw audio folder)
  --[no]rescan: raw audio folders : ignore existing index and re-scan every file
    (default: 'false')
  --[no]write_index: raw audio folders : write the .ad index file (asked if not given)
```

You can also add the `--files` to list all the parsed audio files, `--check_metadata` flag to check missing metadata, or either `--idx` or `--key` to get information for a specific sub-item. See the full available options by running `acids-dataset info --help`. For example, it should provide you informations as follows :

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
skipped_files: []
sr: 44100
waveform: true
writer_class: LMDBWriter
```

`info` also works on raw (not pre-processed) audio folders, that are then scanned as [online datasets](#online-mode) : it gives the number of files, total duration, sampling rates, channels, codecs, file durations and unreadable files of the folder (and `meta_regexp` values with `--meta_regexp`). `--files` lists the scanned files, and `--idx` / `--key` describe a file from its index or its audio path. If the folder has no `.ad` index yet, `info` asks whether to write it (making next scans faster) or to only scan the folder in memory ; use `--write_index` or `--nowrite_index` to answer in advance (the index is not written in non-interactive sessions). Existing indexes are updated.

```bash
acids-dataset info --path /path/to/raw/audio --meta_regexp "**/{{instrument}}_*.wav" --nowrite_index
```

# Presets

Presets are an advanced way to share parsing settings between pre-processing and online datasets. They are gin configuration files describing how audio is parsed : sampling rate, channels, chunking, padding, and features. The same preset can be used to pre-process a dataset or to load it online, giving the same data in both modes.

```bash
acids-dataset preprocess --path /path/to/your/dataset --out /target/path/for/preprocessed --preset rave
```

```python
from acids_dataset import preprocess_dataset
from acids_dataset.datasets import AudioDataset

preprocess_dataset("path/to/your/dataset", out="path/to/preprocessed", preset="rave")
dataset = AudioDataset("path/to/your/dataset", preset="rave")      # online, with the same parsing
dataset = AudioDataset("path/to/preprocessed", preset="rave")      # warns if the dataset does not match the preset
```

| Name          | Content                                                                                                                       |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `default.gin` | base preset, included by other presets. Do not forget to give `chunk_length`, as this is not provided!                        |
| `rave.gin`    | base preset for [RAVE](http://github.com/acids-ircam/RAVE)                                                                    |
| `after.gin`   | base preset for [AFTER](http://github.com/acids-ircam/AFTER) ; MIDI can be parsed by adding the `aftermidi` feature           |
| `ddsp.gin`    | base preset for DDSP models                                                                                                   |

Presets define the following values, that are overridden by the arguments given by the user (`--chunk_length`, `--sample_rate`, `--channels`, `--device` in command line ; `chunk_length`, `sample_rate`, `channels`, `device` in `preprocess_dataset` ; `chunk_length`, `sr`, `channels` for online datasets) : 

| Value          | Description                                                                         |
| -------------- | ----------------------------------------------------------------------------------- |
| `SAMPLE_RATE`  | target sampling rate                                                                |
| `CHANNELS`     | target number of channels (`None` to keep original channels)                        |
| `CHUNK_LENGTH` | chunk length, in samples (int) or seconds (float)                                   |
| `HOP_LENGTH`   | hop length between pre-processed chunks (`None` for `CHUNK_LENGTH`)                 |
| `DEVICE`       | device for feature computation                                                      |

Available presets and their values can be listed with `acids-dataset presets` (and `acids-dataset presets --preset rave` prints a given preset). In Python, see `acids_dataset.presets.list_presets` and `get_preset_values`.

## Writing presets

Custom presets can be placed in an `ad_configs` folder of the current working directory (local presets take precedence over the ones of `acids_dataset/configs`), or given as paths to `.gin` files. Presets usually include `default.gin`, that binds these values to pre-processing (`RawParser`) and online datasets (`OnlineAudioDataset`), and redefine them. Features of `features.parse_features` are computed in both modes ; parameters specific to one mode can be bound to `RawParser` (pre-processing, e.g. `loudness_threshold`, `discard_if_lower_than`) or to `OnlineAudioDataset` (online, e.g. `batch_by`, `pad_mode`). For example, `$(pwd)/ad_configs/rave_loudness.gin` : 

```yaml
include 'rave.gin'

SAMPLE_RATE = 48000

Loudness:
    sr = %SAMPLE_RATE
    device = %DEVICE

features.parse_features:
    features = [@features.Loudness]

RawParser:
    loudness_threshold = -60

OnlineAudioDataset:
    batch_by = "dataset"
```

Additional gin bindings can be given with `--override` in command line, `override` in `preprocess_dataset`, and `preset_overrides` for datasets :

```python
dataset = AudioDataset("path/to/your/dataset", preset="rave", preset_overrides=["CHUNK_LENGTH = 131072", "OnlineAudioDataset.batch_by = 'files'"])
```

# Full augmentation table 

| Name                  | Config name                 | Description                                                                 |
| --------------------- | --------------------------- | --------------------------------------------------------------------------- |
| AddColorNoise         | `addcolornoise.gin`         | Add colored noise to audio signal                                           |
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
| PitchShift            | `pitchshift.gin`            | Random pitch-shifting of the signal (duration preserved).                   |
| PolarityInversion     | `polarityinversion.gin`     | Randomly inverses the waveform (*-1)                                        |
| PreEmphasis           | `preemphasis.gin`           | adds pre-emphasis to a signal.                                              |
| RandomDelay           | `randomdelay.gin`           | random short comb-filtered delays                                           |
| RandomDistort         | `randomdistort.gin`         | random Eqing and distortion of signal                                       |
| RandomEQ              | `randomeq.gin`              | random EQing of the signal                                                  |
| Resample              | `resample.gin`              | resampling of the signal.                                                   |
| Reverse               | `reverse.gin`               | randomly reverses temporally the signal.                                    |
| RoomSimulator         | `roomsimulator.gin`         | performs room reverberation simulation.                                     |
| SevenBandParametricEQ | `sevenbandparametriceq.gin` | Random seven-band parametric EQ.                                            |
| Shift                 | `shift.gin`                 | Random time shift of the signal.                                            |
| TanhDistortion        | `tanhdistortion.gin`        | Performs random tanh distortion to the signal.                              |
| TimeMask              | `timemask.gin`              | Randomly masks temporal zones of the signal with silence.                   |
| TimeStretch           | `timestretch.gin`           | Randomly time-stretches the incoming signal, and crop.                      |
| Stretch               | `stretch.gin`               | Random resampling with polyphase filtering (changes speed and pitch).       |
