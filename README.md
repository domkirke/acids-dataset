# acids-dataset

`acids-dataset` is a preprocessing package for audio data and metadata, mostly used by [RAVE](http://github.com/acids-ircam/RAVE) and [AFTER](http://github.com/acids-ircam/AFTER) but opened for custom use. Built open [lmdb](https://openldap.org/), it leverages the pre-processing step of data parsing required by audio generative models to extract metadata and audio features that can be accessed and used during training. 

## Installation
To install acids-dataset, just install it through pip.

```bash
pip install acids-dataset
```

## Usage

### Preprocessing
`acids-dataset` is available as a command-line tool to easily pre-process a dataset path. For example, simply parse a dataset for [after](http://github.com/acids-ircam/AFTER) with 
```bash
acids-dataset preprocess --path /path/to/your/dataset --out /target/path/for/preprocessed --config after
```
where here `after` is like a preset to automatically embed the metadata required by AFTER (see table below for a comprehensive list of available ). You can also provide a list of filters / excluding filters to select the parsed files : 
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

### Embedding features
`acids_dataset` has specific configuration files to embed metadata (like audio descriptors) in the database, to make them accessible during training. For example, to add loudness and mel profiles for each data chunk, you may add the `mel` and `loudness` configs: 
```bash
acids-dataset preprocess --path /path/to/your/dataset --out /target/path/for/preprocessed --config features/loudness --config features/mel
```
of course, you can customize these features ; please see the section [customize](#customize) below.

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