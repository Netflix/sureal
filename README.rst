SUREAL - Subjective Recovery Analysis
=====================================

.. image:: https://img.shields.io/pypi/v/sureal.svg
    :target: https://pypi.org/project/sureal/
    :alt: Version on pypi

.. image:: https://travis-ci.org/Netflix/sureal.svg?branch=master
    :target: https://travis-ci.org/Netflix/sureal
    :alt: Build Status

SUREAL is a toolbox developed by Netflix for recovering quality scores from noisy measurements obtained by subjective tests.
Read `this <resource/doc/dcc17v3.pdf>`_ paper for some background. SUREAL is being imported by the VMAF_ package.

Currently, SUREAL supports Python 2.7 and 3.7.

.. _VMAF: https://github.com/Netflix/vmaf


Installation
============
SUREAL can be either installed through ``pip`` (available via PyPI_), or locally.

Installation through ``pip``
----------------------------

To install SUREAL via ``pip``, run::

    pip install sureal

Local installation
------------------

To install locally, first, download the source. Under the root directory, (perferrably in a virtualenv_), install the requirements::

    pip install -r requirements.txt

Under Ubuntu, you may also need to install the ``python-tk`` (Python 2) or ``python3-tk`` (Python 3) packages via ``apt``.

To test the source code before installing, run::

    python -m unittest discover -s test -p '*_test.py'

The code thus far has been tested on Ubuntu 16.04 LTS and macOS 10.13.

Lastly, install SUREAL by::

    pip install .

If you want to edit the source, use ``pip install --editable .`` or ``pip install -e .`` instead. Having ``--editable`` allows the changes made in the source to be picked up immediately without re-running ``pip install .``

.. _PyPI: https://pypi.org/project/sureal/
.. _virtualenv: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/


Usage in command line
=====================

Run::

    sureal

This will print usage information::

    usage: subjective_model dataset_filepath [--output-dir output_dir] [--print]

If ``--output-dir`` is given, plots will be written to the output directory.

If ``--print`` is enabled, output statistics will be printed on the command-line and / or the output directory.

Below are two example usages::

    sureal MLE_CO_AP2 resource/dataset/NFLX_dataset_public_raw_last4outliers.py --print \
        --output-dir ./output/NFLX_dataset_public_raw_last4outliers
    sureal MLE_CO_AP2 resource/dataset/VQEGHD3_dataset_raw.py --print \
        --output-dir \./output/VQEGHD3_dataset_raw


Here ``subjective_model`` are the available subjective models offered in the package, including:

  - MOS - Standard mean opinion score

  - MLE - Full maximum likelihood estimation (MLE) model that takes into account both subjects and contents

  - MLE_CO - MLE model that takes into account only subjects ("Content-Oblivious")

  - MLE_CO_AP - Alternative implementation of MLE_CO based on Alternate Projection (AP)

  - MLE_CO_AP2 - Alternative implementation of MLE_CO based on Alternate Projection and per-stimuli confidence interval calculation (AP2)

  - DMOS - Differential MOS, as defined in [ITU-T P.910](https://www.itu.int/rec/T-REC-P.910)

  - DMOS_MLE - apply MLE on DMOS

  - DMOS_MLE_CO - apply MLE_CO on DMOS

  - SR_MOS - Apply subject rejection (SR), as defined in [ITU-R BT.500](https://www.itu.int/rec/R-REC-BT.500), before calculating MOS

  - ZS_SR_MOS - Apply z-score transformation, followed by SR, before calculating MOS

  - SR_DMOS - Apply SR, before calculating DMOS

  - ZS_SR_DMOS - Apply z-score transformation, followed by SR, before calculating DMOS


Dataset files
-------------

``dataset_filepath`` is the path to a dataset file.
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
(i.e. not every subject views every distorted video).
The only difference from the first way is that, the value of ``os`` is now a dictionary, with the key being a subject ID,
and the value being his/her voted score for particular distorted video. For example::

    'os': {'Alice': 40, 'Bob': 45, 'Charlie': 50, 'David': 55, 'Elvis': 60}


Since partial sampling is allowed, it is not required that every subject ID is present in every ``os`` dictionary.


Usage in Python code
====================

See `here <https://colab.research.google.com/drive/1hG6ARc8-rihyJPxIXZysi-sAe0e7xxB8#scrollTo=onasQ091O3sn>`_ for an example script to use SUREAL in Google Collab notebook.


For developers
==============

SUREAL uses tox_ to manage automatic testing and continuous integration with `Travis CI`_ on Github, and setupmeta_ for new version release, packaging and publishing. Refer to `DEVELOPER.md <DEVELOPER.md>`_ for more details.

.. _tox: https://tox.readthedocs.io/en/latest/
.. _Travis CI: https://travis-ci.org/Netflix/sureal
.. _setupmeta: https://github.com/zsimic/setupmeta
