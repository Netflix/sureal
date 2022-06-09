SUREAL - Subjective Recovery Analysis
=====================================

.. image:: https://img.shields.io/pypi/v/sureal.svg
    :target: https://pypi.org/project/sureal/
    :alt: Version on pypi

.. image:: https://travis-ci.com/Netflix/sureal.svg?branch=master
    :target: https://travis-ci.com/Netflix/sureal
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

    sureal --dataset resource/dataset/NFLX_dataset_public_raw_last4outliers.py --models MOS P910 \
        --plot-raw-data --plot-dis-videos --plot-observers --output-dir ./output/NFLX_dataset_public_raw_last4outliers
    sureal --dataset resource/dataset/VQEGHD3_dataset_raw.py --models MOS P910 \
        --plot-raw-data --plot-dis-videos --plot-observers --output-dir ./output/VQEGHD3_dataset_raw

Here ``--models`` are the available subjective models offered in the package, including:

  - MOS - Standard mean opinion score.

  - P910 - Model based on subject bias/inconsistency modeling and maximum likelihood estimation (MLE), newly standardized in `ITU-T P.910 (11/21) Annex E <https://www.itu.int/rec/T-REC-P.910>`_ (also in `ITU-T P.913 (06/21) 12.6 <https://www.itu.int/rec/T-REC-P.913>`_). The details of the algorithm is covered by the two papers aforementioned (`paper 1 <resource/doc/dcc17v3.pdf>`_ and `paper 2 <resource/doc/hvei2020.pdf>`_).

  - P913 - Model based on subject bias removal, standardized in `ITU-T P.913 (06/21) 12.4 <https://www.itu.int/rec/T-REC-P.913>`_.

  - BT500 - Model based on subject rejection, standardized in `ITU-R BT.500-14 (10/2019) A1-2.3.1 <https://www.itu.int/rec/R-REC-BT.500>`_.

The `sureal` command can also invoke subjective models for paired comparison (PC) subjective data. Below is one example::

    sureal --dataset resource/dataset/lukas_pc_dataset.py --models THURSTONE_MLE BT_MLE \
    --plot-raw-data --plot-dis-videos --output-dir ./output/lukas_pc_dataset

Here ``--models`` are the available PC subjective models offered in the package:

  - THURSTONE_MLE - `Thurstone (Case V) <https://en.wikipedia.org/wiki/Thurstonian_model>`_ model, with a MLE solver.

  - BT_MLE - `Bradley-Terry <https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model>`_ model, with a MLE solver.

Both models leverage MLE-based solvers. For the mathematics behind the implementation, refer to `this document <resource/doc/pc.pdf>`_.

Dataset files
-------------

``--dataset`` is the path to a dataset file.
Dataset files may be ``.py`` or ``.json`` files.
The following examples use ``.py`` files, but JSON-formatted files can be constructed in a similar fashion.

There are two ways to construct a dataset file.
The first way is only useful when the subjective test is full sampling,
i.e. every subject views every distorted video. For example::

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


In this example, ``ref_videos`` is a list of reference videos.
Each entry is a dictionary, and must have keys ``content_id``, ``content_name`` and ``path`` (the path to the reference video file).
``dis_videos`` is a list of distorted videos.
Each entry is a dictionary, and must have keys ``content_id`` (the same content ID as the distorted video's corresponding reference video),
``asset_id``, ``os`` (stands for "opinion score"), and ``path`` (the path to the distorted video file).
The value of ``os`` is a list of scores, reach voted by a subject, and must have the same length for all distorted videos
(since it is full sampling).
``ref_score`` is the score assigned to a reference video, and is required when differential score is calculated,
for example, in DMOS.

The second way is more general, and can be used when the test is full sampling or partial sampling
(i.e. not every subject views every distorted video). The only difference from the first way is that, the value of ``os`` is now a dictionary, with the key being a subject ID,
and the value being his/her voted score for particular distorted video. For example::

    'os': {'Alice': 40, 'Bob': 45, 'Charlie': 50, 'David': 55, 'Elvis': 60}


Since partial sampling is allowed, it is not required that every subject ID is present in every ``os`` dictionary.

In case a subject has voted a distorted video twice or more (repetitions), the votes can be logged by having a list in lieu of single vote. For example::

    'os': {'Alice': 40, 'Bob': [45, 45], 'Charlie': [50, 60], 'David': 55, 'Elvis': 60}


In case of a PC dataset, a distorted video is compared against another distorted video, and a vote is recorded. In this case, the key is a tuple of the subject name and the `asset_id` of the distorted video compared against. For example::

    'os': {('Alice', 1): 40, ('Bob', 3): 45}

where 1 and 3 are the `asset_id` of the distorted videos compared against. For an example PC dataset, refer to `lukas_pc_dataset.py <resource/dataset/lukas_pc_dataset.py>`_.

Note that for PC models, we current do not yet support repetitions.

Deprecated command line
================================

The deprecated version of the command line can still be invoked by::

    PYTHONPATH=. python ./sureal/cmd_deprecated.py

Usage in Python code
====================

See `here <https://colab.research.google.com/drive/1hG6ARc8-rihyJPxIXZysi-sAe0e7xxB8#scrollTo=onasQ091O3sn>`_ for an example script to use SUREAL in Google Collab notebook.


For developers
==============

SUREAL uses tox_ to manage automatic testing and continuous integration with `Travis CI`_ on Github, and setupmeta_ for new version release, packaging and publishing. Refer to `DEVELOPER.md <DEVELOPER.md>`_ for more details.

.. _tox: https://tox.readthedocs.io/en/latest/
.. _Travis CI: https://travis-ci.org/Netflix/sureal
.. _setupmeta: https://github.com/zsimic/setupmeta
