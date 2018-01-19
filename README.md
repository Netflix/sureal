SUREAL - Subjective Recovery Analysis
===================

SUREAL is a toolbox developed by Netflix for quality score recovery from noisy subjective measurements obtained by lab tests. Read [this](resource/doc/dcc17v3.pdf) paper for some background.

## Prerequisites & Installation

SUREAL requires a number of Python packages:

  - [`numpy`](http://www.numpy.org/) (>=1.12.0)
  - [`scipy`](http://www.scipy.org/) (>=0.17.1)
  - [`matplotlib`](http://matplotlib.org/1.3.1/index.html) (>=2.0.0)
  - [`pandas`](http://pandas.pydata.org/) (>=0.19.2)
  - [`scikit-learn`](http://scikit-learn.org/stable/) (>=0.18.1)

Upgrade `pip` to the newest version:

```
sudo -H pip install --upgrade pip
```

Then install the required Python packages:

```
pip install --user numpy scipy matplotlib pandas scikit-learn
```

Add the `python/src` subdirectories to the environment variable `PYTHONPATH`:

```
export PYTHONPATH="$(pwd)/python/src:$PYTHONPATH"
```

You can also add it to the environment permanently, by appending to `~/.bashrc`:

```
echo export PYTHONPATH="$(pwd)/python/src:$PYTHONPATH" >> ~/.bashrc
source ~/.bashrc
```

Under macOS, use `~/.bash_profile` instead.

# Command Line Usage

Under root directory, run `./run_subj` to print usage information:

```
usage: run_subj subjective_model dataset_filepath
```

`subjective_model` are the available subjective models offered in the package:
  - `MLE` - Full maximum likelihood estimation model that takes into account both subjects and content
  - `MOS` - Standard

```
./run_subj MLE resource/dataset/NFLX_dataset_public_raw_last4outliers.py
./run_subj MLE resource/dataset/VQEGHD3_dataset_raw.py
```

More ways to use the subjective models can be found in `/python/script/run_subjective_models.py`.