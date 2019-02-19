SUREAL - Subjective Recovery Analysis
===================
[![Build Status](https://travis-ci.org/Netflix/sureal.svg?branch=master)](https://travis-ci.org/Netflix/sureal)

SUREAL is a toolbox developed by Netflix for recovering quality scores from noisy measurements obtained by subjective tests. Read [this](resource/doc/dcc17v3.pdf) paper for some background. SUREAL is being imported by the [VMAF](https://github.com/Netflix/vmaf) package.

## Requirements

- Python 2.7 or 3

## Install through `pip`

SUREAL is now available on [PyPI](https://pypi.org/project/sureal/), and can be installed through:

```
pip install sureal
```

To install locally, download the source and run:

```
pip install .
```

## Prerequisites & Installation

To use SUREAL from source, a number of packages are required:

  - [`numpy`](http://www.numpy.org/) (>=1.12.0)
  - [`scipy`](http://www.scipy.org/) (>=0.17.1)
  - [`matplotlib`](http://matplotlib.org/1.3.1/index.html) (>=2.0.0)
  - [`pandas`](http://pandas.pydata.org/) (>=0.19.2)

Under Ubuntu, you may also need to install the `python-tk` or `python3-tk` packages via `apt`.

First, upgrade `pip` to the newest version; then install the required Python packages:

```
sudo -H pip install --upgrade pip
pip install --user -r requirements.txt
```

For Python 3, use:

```
sudo -H pip3 install --upgrade pip
pip3 install --user -r requirements.txt
```

## Testing

The package has thus far been tested on Ubuntu 16.04 LTS and macOS 10.13. After installation, run:

```
./unittest
```

## Usage in Command Line

If you installed via Pip, run:

```
sureal
```

If you have not installed via Pip, under root directory, just run:

```
python3 -m sureal
```

to run the module.

This will print usage information:

```
usage: subjective_model dataset_filepath [--output-dir output_dir] [--print]
```

If `--output-dir` is given, plots will be written to the output directory. If `--print` is enabled, output statistics will be printed on the command-line and / or the output directory.

Below are two example usages:

```
sureal MLE resource/dataset/NFLX_dataset_public_raw_last4outliers.py
sureal MLE_CO resource/dataset/VQEGHD3_dataset_raw.py
```

Here `subjective_model` are the available subjective models offered in the package, including:
  - MOS - Standard mean opinion score
  - MLE - Full maximum likelihood estimation (MLE) model that takes into account both subjects and contents
  - MLE_CO - MLE model that takes into account only subjects ("Content-Oblivious")
  - DMOS - Differential MOS, as defined in [ITU-T P.910](https://www.itu.int/rec/T-REC-P.910)
  - DMOS_MLE - apply MLE on DMOS
  - DMOS_MLE_CO - apply MLE_CO on DMOS
  - SR_MOS - Apply subject rejection (SR), as defined in [ITU-R BT.500](https://www.itu.int/rec/R-REC-BT.500), before calculating MOS
  - ZS_SR_MOS - Apply z-score transformation, followed by SR, before calculating MOS
  - SR_DMOS - Apply SR, before calculating DMOS
  - ZS_SR_DMOS - Apply z-score transformation, followed by SR, before calculating DMOS

### Dataset Files

`dataset_filepath` is the path to a dataset file. Dataset files may be `.py` or `.json` files. The following examples use `.py` files, but JSON-formatted files can be constructed in a similar fashion.

There are two ways to construct a dataset file. The first way is only useful when the subjective test is full sampling, i.e. every subject views every distorted video. For example:

```
ref_videos = [
    {
      'content_id': 0, 'content_name': 'checkerboard',
      'path': 'checkerboard_1920_1080_10_3_0_0.yuv'
    },
    {
      'content_id': 1, 'content_name': 'flat',
      'path': 'flat_1920_1080_0.yuv'
    },
]
dis_videos = [
    {
      'content_id': 0, 'asset_id': 0,
      'os': [100, 100, 100, 100, 100],
      'path': 'checkerboard_1920_1080_10_3_0_0.yuv'
    },
    {
      'content_id': 0, 'asset_id': 1,
      'os': [40, 45, 50, 55, 60],
      'path': 'checkerboard_1920_1080_10_3_1_0.yuv'
    },
    {
      'content_id': 1, 'asset_id': 2,
      'os': [90, 90, 90, 90, 90],
      'path': 'flat_1920_1080_0.yuv'
    },
    {
      'content_id': 1, 'asset_id': 3,
      'os': [70, 75, 80, 85, 90],
      'path': 'flat_1920_1080_10.yuv'
    },
]
ref_score = 100
```

In this example, `ref_videos` is a list of reference videos. Each entry is a dictionary, and must have keys `content_id`, `content_name` and `path` (the path to the reference video file). `dis_videos` is a list of distorted videos. Each entry is a dictionary, and must have keys `content_id` (the same content ID as the distorted video's corresponding reference video), `asset_id`, `os` (stands for "opinion score"), and `path` (the path to the distorted video file). The value of `os` is a list of scores, reach voted by a subject, and must have the same length for all distorted videos (since it is full sampling). `ref_score` is the score assigned to a reference video, and is required when differential score is calculated, for example, in DMOS.

The second way is more general, and can be used when the test is full sampling or partial sampling (i.e. not every subject views every distorted video). The only difference from the first way is that, the value of `os` is now a dictionary, with the key being a subject ID, and the value being his/her voted score for particular distorted video. For example:

```
'os': {'Alice': 40, 'Bob': 45, 'Charlie': 50, 'David': 55, 'Elvis': 60}
```

Since partial sampling is allowed, it is not required that every subject ID is present in every `os` dictionary.

## Example Script

See [here](https://colab.research.google.com/drive/1hG6ARc8-rihyJPxIXZysi-sAe0e7xxB8#scrollTo=onasQ091O3sn) for example script to use SUREAL in Google Colab notebook.
