import os

import numpy as np
import matplotlib.pyplot as plt

from sureal.subjective_model import MaximumLikelihoodEstimationModelContentOblivious, MosModel
from sureal.config import DisplayConfig
from sureal.dataset_reader import SyntheticRawDatasetReader
from sureal.routine import validate_with_synthetic_dataset

THIS_DIR = os.path.dirname(os.path.realpath(__file__))


color_dict = {
    'Subject/Content-Aware': 'red',
    'Subject-Aware': 'blue',
    'DMOS': 'green',
    'MOS': 'cyan',
    'LIVE DMOS': 'magenta',
    'BR_SR_MOS': 'blue',
    'Proposed': 'black',
    'MLE_CO': 'blue',
}


def validate_with_synthetic_dataset_quality_wrapper(subjective_model_classes, **more):

    ret = {}

    do_plot = more['do_plot'] if 'do_plot' in more else True
    assert isinstance(do_plot, bool)

    do_errorbar = more['do_errorbar'] if 'do_errorbar' in more else False
    assert isinstance(do_errorbar, bool)

    dataset_filepath = os.path.join(THIS_DIR, '..', 'dataset', 'NFLX_dataset_public_raw.py')

    seed = more['seed'] if 'seed' in more else 0

    force_synth_subjbias_zeromean = more['force_synth_subjbias_zeromean'] if 'force_synth_subjbias_zeromean' in more else True
    assert isinstance(force_synth_subjbias_zeromean, bool)

    np.random.seed(seed)

    synthetic_result_dict = more['synthetic_result_dict'] if 'synthetic_result_dict' in more else None

    n_bootstrap = more['n_bootstrap'] if 'n_bootstrap' in more else None
    if n_bootstrap is not None:
        for subjective_model_class in subjective_model_classes:
            assert 'bootstrap' in subjective_model_class.__name__.lower()

    if synthetic_result_dict is None:
        synthetic_result_dict = 'quality_ambiguity'

    if isinstance(synthetic_result_dict, dict):

        pass

        if do_plot:
            fig, axs = plt.subplots(figsize=(12, 8), nrows=2, ncols=3)
            ax_dict = {
                'quality_scores': axs[0][0],
                'quality_scores_std': axs[0][1],
                'quality_ambiguity': axs[0][2],
                'observer_bias': axs[1][0],
                'observer_inconsistency': axs[1][1],
            }
        else:
            ax_dict = None

    elif isinstance(synthetic_result_dict, str):
        if synthetic_result_dict == 'quality_ambiguity':
            synthetic_result_dict = {
                'quality_scores': np.random.uniform(1., 5., 79),
                'observer_bias': np.random.normal(0.0, 0.0, 30),
                'observer_inconsistency': np.abs(np.random.uniform(0.0, 0.0, 30)),
                'content_bias': np.random.normal(0.0, 0.0, 9),
                'content_ambiguity': np.abs(np.random.uniform(0.0, 0.0, 9)),
                'quality_ambiguity': np.abs(np.random.uniform(0.0, 1.0, 79)),
            }
            # derived
            synthetic_result_dict['quality_scores_std'] = synthetic_result_dict['quality_ambiguity'] / \
                                                          np.sqrt(len(synthetic_result_dict['observer_bias']))

            if do_plot:
                fig, axs = plt.subplots(figsize=(12, 4), nrows=1, ncols=3)
                ax_dict = {
                    'quality_scores': axs[0],
                    'quality_scores_std': axs[1],
                    'quality_ambiguity': axs[2],
                }
            else:
                ax_dict = None

        elif synthetic_result_dict == 'observer_only':
            synthetic_result_dict = {
                'quality_scores': np.random.uniform(1., 5., 79),
                'observer_bias': np.random.normal(0, 1, 30),
                'observer_inconsistency': np.abs(np.random.uniform(0.0, 1.0, 30)),
                'content_bias': np.random.normal(0.0, 0.0, 9),
                'content_ambiguity': np.abs(np.random.uniform(0.0, 0.0, 9)),
            }

            if do_plot:
                fig, axs = plt.subplots(figsize=(12, 4), nrows=1, ncols=3)
                ax_dict = {
                    'quality_scores': axs[0],
                    'observer_bias': axs[1],
                    'observer_inconsistency': axs[2],
                }
            else:
                ax_dict = None

        else:
            assert False, f'unknown string for synthetic_result_dict: {synthetic_result_dict}'
    else:
        assert False

    if force_synth_subjbias_zeromean:
        # pass
        synthetic_result_dict['observer_bias'] -= np.mean(synthetic_result_dict['observer_bias'])

    ret['synthetic_result'] = synthetic_result_dict

    synthetic_dataset_reader_class = SyntheticRawDatasetReader

    ret_ = validate_with_synthetic_dataset(
        synthetic_dataset_reader_class=synthetic_dataset_reader_class,
        subjective_model_classes=subjective_model_classes,
        dataset_filepath=dataset_filepath,
        synthetic_result=synthetic_result_dict,
        ax_dict=ax_dict,
        delta_thr=4e-3,
        color_dict=color_dict,
        do_errorbar=do_errorbar,
        n_bootstrap=n_bootstrap,
    )
    ret.update(ret_)

    return ret


def run_synthetic_validate_mos():

    rets = []
    seeds = range(10)
    for seed in seeds:
        ret = validate_with_synthetic_dataset_quality_wrapper(
            [
                MosModel,
            ],
            synthetic_result_dict='quality_ambiguity',
            do_plot=True,
            do_errorbar=True,
            seed=seed)
        rets.append(ret)

    qs_ci_percs = []
    for ret, seed in zip(rets, seeds):
        qs_ci_perc = ret['results']['MOS']['quality_scores_ci_perc']
        print(f"Seed {seed} CI%: quality_scores: {qs_ci_perc:.1f}")
        qs_ci_percs.append(qs_ci_perc)

    print(f'-- Averge CI%: quality_scores: {np.mean(qs_ci_percs):.1f} --')


def run_synthetic_validate_mleco():

    rets = []
    seeds = range(10)
    for seed in seeds:
        ret = validate_with_synthetic_dataset_quality_wrapper(
            [
                MaximumLikelihoodEstimationModelContentOblivious,
            ],
            synthetic_result_dict='observer_only',
            do_plot=True,
            do_errorbar=True,
            seed=seed)
        rets.append(ret)

    qs_ci_percs = []
    ob_ci_percs = []
    oi_ci_percs = []
    for ret, seed in zip(rets, seeds):
        qs_ci_perc = ret['results']['MLE_CO']['quality_scores_ci_perc']
        ob_ci_perc = ret['results']['MLE_CO']['observer_bias_ci_perc']
        oi_ci_perc = ret['results']['MLE_CO']['observer_inconsistency_ci_perc']
        print(f"Seed {seed} CI%: quality_scores: {qs_ci_perc:.1f} observer_bias {ob_ci_perc:.1f} observer_inconsistency {oi_ci_perc:.1f}")
        qs_ci_percs.append(qs_ci_perc)
        ob_ci_percs.append(ob_ci_perc)
        oi_ci_percs.append(oi_ci_perc)

    print(f'-- Averge CI%: quality_scores: {np.mean(qs_ci_percs):.1f} observer_bias {np.mean(ob_ci_percs):.1f} observer_inconsistency {np.mean(oi_ci_percs):.1f} --')


if __name__ == '__main__':

    run_synthetic_validate_mos()
    run_synthetic_validate_mleco()

    DisplayConfig.show(write_to_dir=None)

    print('Done.')
