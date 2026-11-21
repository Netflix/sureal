# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SUREAL (Subjective Recovery Analysis) is a Netflix Python library and CLI for recovering quality scores (MOS) from noisy raw opinion scores in subjective experiments. It covers both absolute-category-rating models (MOS, MLE-based subject bias/inconsistency models, ITU-T P.910 / P.913 and ITU-R BT.500 procedures) and paired-comparison (PC) models (Thurstone Case V, Bradley-Terry). The math is described in `resource/doc/dcc17v3.pdf`, `resource/doc/hvei2020.pdf`, and `resource/doc/pc.pdf`.

## Commands

```bash
# Dev setup (editable install; tests rely on the package being importable)
pip install -r requirements.txt -r test/requirements.txt
pip install -e .
# or: tox -e venv && source .venv/bin/activate

# Tests (pytest is the day-to-day runner)
python -m pytest test/ -v
python -m pytest test/dataset_loader_test.py -v               # one file
python -m pytest test/ -k test_btnr_subjective_model -v       # one test
python -m pytest test/ -x                                     # stop on first failure

# Full CI-equivalent run (pytest + coverage + --doctest-modules, plus a README rst check)
tox
tox -e py37 -- -k test_btnr_subjective_model

# Alternative unittest runner (from README)
python -m unittest discover --start test --pattern '*_test.py' --verbose --buffer

# CLI
sureal --dataset resource/dataset/NFLX_dataset_public_raw_last4outliers.json --models MOS P910 \
    --plot-raw-data --plot-dis-videos --plot-observers --output-dir ./output/nflx
sureal --dataset resource/dataset/lukas_pc_dataset.json --models THURSTONE_MLE BT_MLE --plot-raw-data --output-dir ./output/pc
```

There is no linter configured. Release is via setupmeta (`python setup.py version --bump patch --push`); the version lives in `sureal/__init__.py` and `CHANGELOG.md` gets an entry per release (see `DEVELOPER.md`).

## Architecture

**Pipeline:** dataset file → `load_dataset` (`sureal/dataset_loader.py`) → `DatasetReader` (`sureal/dataset_reader.py`) → `SubjectiveModel.run_modeling()` → result dict. `sureal/routine.py` (`run_subjective_models`, `format_output_of_run_subjective_models`) orchestrates this for the CLI (`sureal/__main__.py`) and handles all plotting.

**Datasets.** A dataset is a module-like object (a `Namespace` for JSON/YAML, an imported module for legacy `.py`) with `ref_videos` and `dis_videos` lists, plus optional `ref_score`, `dataset_name`, etc. Each `dis_video['os']` can be a list (full sampling), a dict `{subject: score}` (partial sampling), a dict with list values (repetitions), or for PC data a dict keyed by `(subject, compared_asset_id)`. `content_id`s must be contiguous `0..N-1`. Format details are in `README.rst`. Writers (`DatasetReader.write_out_dataset`, `to_*_dataset_file`) emit the `.py` format.

**DatasetReader hierarchy.** `RawDatasetReader` normalizes every `os` format into `opinion_score_3darray` with shape **(E videos, S subjects, R repetitions)**, NaN-padded for missing votes. Models consume only this array, so new input formats belong in the reader, not the models. `PairedCompDatasetReader` produces a (E, E, S) comparison array instead. The `Mocked*`/`Synthetic*`/`Corrupt*`/`Select*`/`MissingData*` readers take an `input_dict` and override `opinion_score_3darray` to synthesize or perturb data for validation experiments and tests.

**SubjectiveModel hierarchy** (`sureal/subjective_model.py`, `sureal/pc_subjective_model.py`, `sureal/experimental.py`):
- Every model inherits `TypeVersionEnabled` (`sureal/core/mixin.py`) and must define class attributes `TYPE` and `VERSION`. The CLI's `--models` strings are resolved with `SubjectiveModel.find_subclass(TYPE)`, which walks subclasses recursively and asserts exactly one match. TYPEs must be globally unique, and a model is only discoverable if its module has been imported.
- Subclasses implement the classmethod `_run_modeling(dataset_reader, **kwargs)`, which returns a dict that must include `quality_scores`, and often also `observer_bias`, `observer_inconsistency`, `quality_scores_std`, `reconstructions`, and similar keys.
- Common preprocessing happens in `_get_opinion_score_3darray_with_preprocessing` and is driven by kwargs: `dscore_mode`, `zscore_mode`, `bias_offset`, `subject_rejection` / `subject_rejection_type` (kurtosis/pearson/spearman). Many models are thin subclasses that only override `run_modeling` to inject these kwargs (e.g. `SubjrejMosModel`, `ZscoringMosModel`, `BiasremvSubjrejMosModel`, the DMOS variants). Postprocessing (`normalize_final`, `transform_final`) happens in `_postprocess_model_result`.
- The standards-named models users see (`BT500`, `P913`, `P910`) are aliases defined in `sureal/__init__.py` as subclasses of `SubjrejMosModel`, `BiasremvSubjrejMosModel`, and `SubjectMLEModelProjectionSolver2`.
- `PairedCompSubjectiveModel` subclasses `SubjectiveModel` but requires a `PairedCompDatasetReader`. The CLI refuses to mix PC and non-PC models in one run.

**Paths.** `SurealConfig` (`sureal/config.py`) resolves paths relative to the repo root: `resource/` (public datasets), `test/resource/` (test fixtures), and `workspace/workdir/` (scratch output, used by tests). `DisplayConfig.show(write_to_dir=...)` saves every open matplotlib figure as PNGs; without it, figures are shown interactively.

## Testing notes

- Tests subclass `sureal.tools.misc.MyTestCase`, whose `assertAlmostEqual` collects failures and reports them all in `tearDown` instead of stopping at the first one. Most tests check numeric outputs from models on fixtures in `test/resource/` against hard-coded values (`places=4` and similar). An intentional numeric change to a model means updating those expected values.
- `test/cmd_test.py` runs `sureal/__main__.py` as a subprocess, so the package must be installed (editable is fine).
- `test/doctest_test.py` runs the doctests in `sureal/tools/misc.py`. tox also passes `--doctest-modules`.

## Other directories

`itut_p910_demo/`, `itut_p913_demo/`, `itur_bt500_demo/` are standalone reference scripts (numpy/scipy only, CSV input) that accompany the ITU recommendations. They do not import `sureal`. `resource/util/mos_to_sureal.py` converts a wide CSV table of scores (one row per PVS, one column per subject) into SUREAL dataset format.

## Contribution conventions

From `DEVELOPER.md`: follow the existing style. Contributions are Apache 2.0 licensed. Source files carry `__copyright__` / `__license__` module attributes.
