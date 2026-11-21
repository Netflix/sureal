SUREAL - Subjective Recovery Analysis
=====================================

.. image:: https://img.shields.io/pypi/v/sureal.svg
    :target: https://pypi.org/project/sureal/
    :alt: Version on pypi

.. image:: https://github.com/Netflix/sureal/actions/workflows/sureal.yml/badge.svg?branch=master
    :target: https://github.com/Netflix/sureal/actions/workflows/sureal.yml
    :alt: Build Status

SUREAL is a toolbox developed by Netflix that includes a number of models for the recovery of mean opinion scores (MOS) from noisy measurements obtained in psychovisual subjective experiments.
Read `this <resource/doc/dcc17v3.pdf>`_ paper and `this latest <resource/doc/hvei2020.pdf>`_ paper for some background.

SUREAL also includes models to recover MOS from paired comparison (PC) subjective data, such as `Thurstone (Case V) <https://en.wikipedia.org/wiki/Thurstonian_model>`_ and `Bradley-Terry <https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model>`_.

Installation
============
SUREAL can be either installed through ``pip`` (available via PyPI_), or locally.

Installation through ``pip``
----------------------------

To install SUREAL via ``pip``, run::

    pip install sureal

Local installation
------------------

To install locally, first, download the source. Under the root directory, (preferably in a virtualenv_), install the requirements::

    pip install -r requirements.txt

Under Ubuntu, you may also need to install the ``python-tk`` (Python 2) or ``python3-tk`` (Python 3) packages via ``apt``.

To test the source code before installing, run::

    python -m unittest discover --start test --pattern '*_test.py' --verbose --buffer


Lastly, install SUREAL by::

    pip install .

If you want to edit the source, use ``pip install --editable .`` or ``pip install -e .`` instead. Having ``--editable`` allows the changes made in the source to be picked up immediately without re-running ``pip install .``

.. _PyPI: https://pypi.org/project/sureal/
.. _virtualenv: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/


Usage in command line
=====================

Run::

    sureal --help

This will print usage information::

    usage: sureal [-h] --dataset DATASET --models MODELS [MODELS ...] [--output-dir OUTPUT_DIR]
    [--plot-raw-data] [--plot-dis-videos] [--plot-observers]

    optional arguments:
      -h, --help            show this help message and exit
      --dataset DATASET     Path to the dataset file.
      --models MODELS [MODELS ...]
                            Subjective models to use (can specify more than one),
                            choosing from: MOS, P910, P913, BT500.
      --output-dir OUTPUT_DIR
                            Path to the output directory (will force create is not existed).
                            If not specified, plots will be displayed and output will be printed.
      --plot-raw-data       Plot the raw data. This includes the raw opinion scores presented
                            in a video-subject matrix, counts per video and counts per subject.
      --plot-dis-videos     Plot the subjective scores of the distorted videos.
      --plot-observers      Plot the scores of the observers.

Below are two example usages::

    sureal --dataset resource/dataset/NFLX_dataset_public_raw_last4outliers.json --models MOS P910 \
        --plot-raw-data --plot-dis-videos --plot-observers --output-dir ./output/NFLX_dataset_public_raw_last4outliers
    sureal --dataset resource/dataset/VQEGHD3_dataset_raw.json --models MOS P910 \
        --plot-raw-data --plot-dis-videos --plot-observers --output-dir ./output/VQEGHD3_dataset_raw

Here ``--models`` are the available subjective models offered in the package, including:

  - MOS - Standard mean opinion score.

  - P910 - Model based on subject bias/inconsistency modeling and maximum likelihood estimation (MLE), newly standardized in `ITU-T P.910 (11/21) Annex E <https://www.itu.int/rec/T-REC-P.910>`_ (also in `ITU-T P.913 (06/21) 12.6 <https://www.itu.int/rec/T-REC-P.913>`_). The details of the algorithm is covered by the two papers aforementioned (`paper 1 <resource/doc/dcc17v3.pdf>`_ and `paper 2 <resource/doc/hvei2020.pdf>`_).

  - P913 - Model based on subject bias removal, standardized in `ITU-T P.913 (06/21) 12.4 <https://www.itu.int/rec/T-REC-P.913>`_.

  - BT500 - Model based on subject rejection, standardized in `ITU-R BT.500-14 (10/2019) A1-2.3.1 <https://www.itu.int/rec/R-REC-BT.500>`_.

The `sureal` command can also invoke subjective models for paired comparison (PC) subjective data. Below is one example::

    sureal --dataset resource/dataset/lukas_pc_dataset.json --models THURSTONE_MLE BT_MLE \
    --plot-raw-data --plot-dis-videos --output-dir ./output/lukas_pc_dataset

Here ``--models`` are the available PC subjective models offered in the package:

  - THURSTONE_MLE - `Thurstone (Case V) <https://en.wikipedia.org/wiki/Thurstonian_model>`__ model, with a MLE solver.

  - BT_MLE - `Bradley-Terry <https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model>`__ model, with a MLE solver.

Both models leverage MLE-based solvers. For the mathematics behind the implementation, refer to `this document <resource/doc/pc.pdf>`_.

Dataset files
-------------

``--dataset`` is the path to a dataset file.

SUREAL supports three dataset file formats:

- JSON (``.json``) - Recommended. Easy to generate programmatically.
- YAML (``.yaml``, ``.yml``) - More human-readable alternative to JSON.
- Python (``.py``) - Legacy format, still fully supported.

The format is auto-detected based on file extension.

Dataset structure
~~~~~~~~~~~~~~~~~

A dataset contains two required fields:

- ``ref_videos`` - List of reference (source) videos
- ``dis_videos`` - List of distorted (test) videos with opinion scores

Optional fields include ``dataset_name``, ``ref_score``, ``yuv_fmt``, ``width``, ``height``, etc.

Example: JSON format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

    {
      "dataset_name": "my_experiment",
      "ref_score": 5.0,
      "ref_videos": [
        {"content_id": 0, "content_name": "checkerboard", "path": "ref/checkerboard.yuv"},
        {"content_id": 1, "content_name": "flat", "path": "ref/flat.yuv"}
      ],
      "dis_videos": [
        {
          "content_id": 0, "asset_id": 0,
          "os": {"Alice": 5, "Bob": 4, "Charlie": 5},
          "path": "ref/checkerboard.yuv"
        },
        {
          "content_id": 0, "asset_id": 1,
          "os": {"Alice": 2, "Bob": 3, "Charlie": 2},
          "path": "dis/checkerboard_q1.yuv"
        },
        {
          "content_id": 1, "asset_id": 2,
          "os": {"Alice": 4, "Bob": 5, "Charlie": 4},
          "path": "dis/flat_q1.yuv"
        }
      ]
    }

Example: YAML format
~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    dataset_name: my_experiment
    ref_score: 5.0

    ref_videos:
      - content_id: 0
        content_name: checkerboard
        path: ref/checkerboard.yuv
      - content_id: 1
        content_name: flat
        path: ref/flat.yuv

    dis_videos:
      - content_id: 0
        asset_id: 0
        path: ref/checkerboard.yuv
        os:
          Alice: 5
          Bob: 4
          Charlie: 5
      - content_id: 0
        asset_id: 1
        path: dis/checkerboard_q1.yuv
        os:
          Alice: 2
          Bob: 3
          Charlie: 2

Field descriptions
~~~~~~~~~~~~~~~~~~

``ref_videos`` entries must each have:

- ``content_id`` - Unique integer ID for the source content (0 to N-1, no gaps)
- ``content_name`` - Human-readable name for the content
- ``path`` - Path to the reference video file

``dis_videos`` entries must each have:

- ``content_id`` - Must match a ``content_id`` from ``ref_videos``
- ``asset_id`` - Unique integer ID for this distorted video
- ``path`` - Path to the distorted video file
- ``os`` - Opinion scores (see formats below)

``ref_score`` is the score assigned to a reference video, required when calculating differential scores (e.g., DMOS).

Opinion score formats
~~~~~~~~~~~~~~~~~~~~~

There are multiple ways to represent opinion scores in the ``os`` field of each distorted video.

**Full sampling (list format)** - When every subject views every video, ``os`` can be a list.
All distorted videos must have the same number of scores::

    "os": [5, 4, 5, 3, 4]

**Partial sampling (dictionary format)** - More flexible. Subject IDs as keys, scores as values.
Not every subject needs to appear in every ``os`` dictionary::

    "os": {"Alice": 5, "Bob": 4, "Charlie": 5}

**With repetitions** - When a subject votes multiple times, use a list for their scores::

    "os": {"Alice": 5, "Bob": [4, 4], "Charlie": [5, 4, 5]}

**Paired comparison (PC) format** - For PC datasets, the key is a tuple of subject name and the ``asset_id`` of the compared video::

    "os": {["Alice", 1]: 1, ["Bob", 3]: 0}

where 1 and 3 are the ``asset_id`` of the videos compared against, and the values indicate the comparison result.
For an example PC dataset, refer to `lukas_pc_dataset.json <resource/dataset/lukas_pc_dataset.json>`_.

Note that for PC models, we currently do not yet support repetitions.

Deprecated command line
================================

The deprecated version of the command line can still be invoked by::

    PYTHONPATH=. python ./sureal/cmd_deprecated.py

Usage in Python code
====================

See `here <https://colab.research.google.com/drive/1hG6ARc8-rihyJPxIXZysi-sAe0e7xxB8#scrollTo=onasQ091O3sn>`_ for an example script to use SUREAL in Google Collab notebook.


For developers
==============

SUREAL uses tox_ to manage automatic testing and continuous integration with `GitHub Actions`_, and setupmeta_ for new version release, packaging and publishing. Refer to `DEVELOPER.md <DEVELOPER.md>`_ for more details.

.. _tox: https://tox.readthedocs.io/en/latest/
.. _GitHub Actions: https://github.com/Netflix/sureal/actions
.. _setupmeta: https://github.com/zsimic/setupmeta
