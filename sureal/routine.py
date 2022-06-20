import copy
import math
import os
from typing import Union

import numpy as np
import scipy.stats

from sureal.config import SurealConfig
from sureal.perf_metric import PccPerfMetric, SrccPerfMetric, RmsePerfMetric
from sureal.subjective_model import SubjectiveModel
from sureal.tools.decorator import persist_to_dir
from sureal.tools.stats import histc

try:
    from matplotlib import pyplot as plt

except (ImportError, RuntimeError):
    # This file is sometimes imported too early by __main__.py, before the venv (with matplotlib) is installed
    # OSX system python comes with an ancient matplotlib that triggers RuntimeError when imported in this way
    plt = None

from sureal.dataset_reader import RawDatasetReader, PairedCompDatasetReader, \
    MissingDataRawDatasetReader, SelectSubjectRawDatasetReader
from sureal.tools.misc import import_python_file, import_json_file, Timer, \
    cmap_factory

__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"


def run_subjective_models(dataset_filepath, subjective_model_classes, do_plot=None, **kwargs):

    def _get_reconstruction_stats(raw_scores, rec_scores):
        assert raw_scores.shape == rec_scores.shape
        rec_scores, raw_scores = zip(*[(rec, raw) for rec, raw in zip(rec_scores.ravel(), raw_scores.ravel())
                                 if (not np.isnan(rec) and not np.isnan(raw))])
        rmse = RmsePerfMetric(raw_scores, rec_scores).evaluate(enable_mapping=False)['score']
        cc = PccPerfMetric(raw_scores, rec_scores).evaluate(enable_mapping=False)['score']
        srocc = SrccPerfMetric(raw_scores, rec_scores).evaluate(enable_mapping=False)['score']
        return {
            'rmse': rmse,
            'cc': cc,
            'srocc': srocc,
        }

    if do_plot is None:
        do_plot = []

    if 'dataset_reader_class' in kwargs:
        dataset_reader_class = kwargs['dataset_reader_class']
    else:
        dataset_reader_class = RawDatasetReader

    if 'dataset_reader_info_dict' in kwargs:
        dataset_reader_info_dict = kwargs['dataset_reader_info_dict']
    else:
        dataset_reader_info_dict = {}

    if 'plot_type' in kwargs:
        plot_type = kwargs['plot_type']
    else:
        plot_type = 'errorbar'

    if 'ax_dict' in kwargs:
        ax_dict = kwargs['ax_dict']
        del kwargs['ax_dict']
    else:
        ax_dict = {}

    raw_score_cmap = kwargs['raw_score_cmap'] if 'raw_score_cmap' in kwargs else cmap_factory('red2green')

    raw_score_residue_range = kwargs['raw_score_residue_range'] if 'raw_score_residue_range' in kwargs else [None, None]

    sort_quality_scores_in_plot = kwargs['sort_quality_scores_in_plot'] if 'sort_quality_scores_in_plot' in kwargs \
        else False
    assert isinstance(sort_quality_scores_in_plot, bool), 'sort_quality_scores_in_plot need to be True or False'

    assert len(raw_score_residue_range) == 2

    colors = ['black', 'gray', 'blue', 'red'] * 2

    if dataset_filepath.endswith('.py'):
        dataset = import_python_file(dataset_filepath)
    elif dataset_filepath.endswith('.json'):
        dataset = import_json_file(dataset_filepath)
    else:
        raise AssertionError("Unknown input type, must be .py or .json")
    dataset_reader = dataset_reader_class(dataset, input_dict=dataset_reader_info_dict)

    subjective_models = [
        s(dataset_reader) for s in subjective_model_classes
    ]

    results = [
        s.run_modeling(**kwargs) for s in subjective_models
    ]

    for result in results:
        dis_video_names = [dis_video['path'] for dis_video in dataset_reader.dis_videos]
        result['dis_video_names'] = dis_video_names

    for subjective_model, result in zip(subjective_models, results):
        if 'raw_scores' in result and 'reconstructions' in result:
            result['reconstruction_stats'] = _get_reconstruction_stats(result['raw_scores'], result['reconstructions'])

    if do_plot == 'all' or 'raw_scores' in do_plot:

        # TODO: visualize repetitions - currently taking mean over repetitions before plotting
        mtx = np.nanmean(dataset_reader.opinion_score_3darray, axis=2).T

        if 'ax_raw_scores' in ax_dict:
            ax_rawscores = ax_dict['ax_raw_scores']
            fig = None
        else:
            n_subj, n_stim = mtx.shape
            w, h = _get_imshow_width_and_height(n_subj, n_stim)
            fig, ax_rawscores = plt.subplots(figsize=(w, h))

        im = ax_rawscores.imshow(mtx, interpolation='nearest', cmap=raw_score_cmap)
        ax_rawscores.set_title(r'Raw Opinion Scores ($u_{ij}$)')
        ax_rawscores.set_xlabel(r'Video Stimuli ($j$)')
        ax_rawscores.set_ylabel(r'Test Subjects ($i$)')
        plt.colorbar(im, ax=ax_rawscores)
        if fig is not None:
            fig.tight_layout()

    if do_plot == 'all' or 'raw_counts' in do_plot:

        # TODO: visualize repetitions - currently taking mean over repetitions before plotting
        mtx = np.nanmean(dataset_reader.opinion_score_3darray, axis=2).T

        if 'ax_raw_counts' in ax_dict:
            ax_rawcounts = ax_dict['ax_raw_counts']
            fig = None
        else:
            n_subj, n_stim = mtx.shape
            w, h = _get_plot_width_and_height(n_stim)
            fig, ax_rawcounts = plt.subplots(figsize=(w, h))

        histcs = np.apply_along_axis(func1d=histc, axis=0, arr=mtx)
        datas = list()
        for ih, h in enumerate(histcs):
            data = [(ih, k, h[k]) for k in sorted(h.keys())]
            datas += data
        xs, ys, vs = zip(*datas)
        ax_rawcounts.scatter(xs, ys, s=np.array(vs) * 2, alpha=0.4)
        ax_rawcounts.set_xlabel(r'Video Stimuli ($j$)')
        ax_rawcounts.set_title(r'Raw Opinion Scores ($u_{ij}$) Counts')
        ax_rawcounts.grid()
        ax_rawcounts.set_xlim(0, len(histcs))
        if fig is not None:
            fig.tight_layout()

    if do_plot == 'all' or 'raw_counts_per_subject' in do_plot:

        # TODO: visualize repetitions - currently taking mean over repetitions before plotting
        mtx = np.nanmean(dataset_reader.opinion_score_3darray, axis=2).T

        if 'raw_counts_per_subject' in ax_dict:
            ax_rawcounts_per_subject = ax_dict['raw_counts_per_subject']
            fig = None
        else:
            n_subj, n_stim = mtx.shape
            w, h = _get_plot_width_and_height(n_subj)
            fig, ax_rawcounts_per_subject = plt.subplots(figsize=(w, h))

        histcs = np.apply_along_axis(func1d=histc, axis=1, arr=mtx)
        datas = list()
        for ih, h in enumerate(histcs):
            data = [(ih, k, h[k]) for k in sorted(h.keys())]
            datas += data
        xs, ys, vs = zip(*datas)
        ax_rawcounts_per_subject.scatter(xs, ys, s=np.array(vs) * 2, alpha=0.4)
        ax_rawcounts_per_subject.set_xlabel(r'Test Subjects ($i$)')
        ax_rawcounts_per_subject.set_title(r'Raw Opinion Scores ($u_{ij}$) Counts')
        ax_rawcounts_per_subject.grid()
        ax_rawcounts_per_subject.set_xlim(0, len(histcs))
        if fig is not None:
            fig.tight_layout()

    if do_plot == 'all' or 'raw_scores_minus_quality_scores' in do_plot:

        for subjective_model, result in zip(subjective_models, results):
            if 'quality_scores' in result:
                quality_scores = result['quality_scores']
                label = subjective_model.TYPE

                # TODO: visualize repetitions - currently taking mean over repetitions before plotting
                mtx = dataset_reader.opinion_score_3darray
                mtx = mtx - np.array(quality_scores)[:, None, None]
                mtx = np.nanmean(mtx, axis=2).T

                if 'ax_raw_scores_minus_quality_scores' in ax_dict:
                    ax_raw_scores_minus_quality_scores = ax_dict['ax_raw_scores_minus_quality_scores']
                    fig = None
                else:
                    w, h = _get_imshow_width_and_height(*mtx.shape)
                    fig, ax_raw_scores_minus_quality_scores = plt.subplots(figsize=(w, h))

                im = ax_raw_scores_minus_quality_scores.imshow(mtx, interpolation='nearest',
                                                               vmin=raw_score_residue_range[0], vmax=raw_score_residue_range[1],
                                                               cmap=raw_score_cmap)
                ax_raw_scores_minus_quality_scores.set_title(r'$u_{ij} - \psi_j$' + ', {}'.format(label))
                ax_raw_scores_minus_quality_scores.set_xlabel(r'Video Stimuli ($j$)')
                ax_raw_scores_minus_quality_scores.set_ylabel(r'Test Subjects ($i$)')
                plt.colorbar(im, ax=ax_raw_scores_minus_quality_scores)
                if fig is not None:
                    fig.tight_layout()

    if do_plot == 'all' or 'raw_scores_minus_quality_scores_and_observer_bias' in do_plot:

        for subjective_model, result in zip(subjective_models, results):
            if 'quality_scores' in result and 'observer_bias' in result:
                observer_bias = result['observer_bias']
                quality_scores = result['quality_scores']
                label = subjective_model.TYPE

                # TODO: visualize repetitions - currently taking mean over repetitions before plotting
                mtx = dataset_reader.opinion_score_3darray
                mtx = mtx - np.array(quality_scores)[:, None, None]
                mtx = mtx - np.array(observer_bias)[None, :, None]
                mtx = np.nanmean(mtx, axis=2).T

                if 'ax_raw_scores_minus_quality_scores_and_observer_bias' in ax_dict:
                    ax_raw_scores_minus_quality_scores_and_observer_bias = ax_dict['ax_raw_scores_minus_quality_scores_and_observer_bias']
                    fig = None
                else:
                    w, h = _get_imshow_width_and_height(*mtx.shape)
                    fig, ax_raw_scores_minus_quality_scores_and_observer_bias = plt.subplots(figsize=(w, h))

                im = ax_raw_scores_minus_quality_scores_and_observer_bias.imshow(mtx, interpolation='nearest',
                                                                                 vmin=raw_score_residue_range[0], vmax=raw_score_residue_range[1],
                                                                                 cmap=raw_score_cmap)
                ax_raw_scores_minus_quality_scores_and_observer_bias.set_title(r'$u_{ij} - \psi_j - \Delta_i$' + ', {}'.format(label))
                ax_raw_scores_minus_quality_scores_and_observer_bias.set_xlabel(r'Video Stimuli ($j$)')
                ax_raw_scores_minus_quality_scores_and_observer_bias.set_ylabel(r'Test Subjects ($i$)')
                plt.colorbar(im, ax=ax_raw_scores_minus_quality_scores_and_observer_bias)
                if fig is not None:
                    fig.tight_layout()

    if do_plot == 'all' or 'quality_scores_vs_raw_scores' in do_plot:

        # TODO: visualize repetitions - currently taking mean over repetitions before plotting
        mtx = np.nanmean(dataset_reader.opinion_score_3darray, axis=2).T
        num_obs = mtx.shape[0]
        assert num_obs > 1, 'need snum_subj > 1 for subplots to work'

        min_lim = np.nanmin(mtx)
        max_lim = np.nanmax(mtx)

        nrows = int(math.floor(math.sqrt(num_obs)))
        ncols = int(math.ceil(num_obs / float(nrows)))

        for subjective_model, result in zip(subjective_models, results):
            if 'quality_scores' in result:
                fig, axs = plt.subplots(figsize=(ncols * 4, nrows * 4), ncols=ncols, nrows=nrows)
                quality_scores = result['quality_scores']
                label = subjective_model.TYPE
                for i_obs in range(num_obs):
                    assert num_obs > 1
                    ax = axs.flatten()[i_obs]
                    ax.set_title(f"#{i_obs + 1}")
                    raw_scores = mtx[i_obs, :]
                    ax.scatter(raw_scores, quality_scores, label=label)
                    ax.set_xlim([min_lim, max_lim])
                    ax.set_ylim([min_lim, max_lim])
                    ax.plot([min_lim, max_lim], [min_lim, max_lim], '-r')
                    ax.set_xlabel('Raw Score ($u_{ij}$)')
                    ax.set_ylabel('Recovered Quality Score ($\psi_j$)')
                    ax.legend()
                    ax.grid()
                fig.tight_layout()

    if do_plot == 'all' or 'quality_scores' in do_plot:
        # ===== plot quality scores =====
        bar_width = 0.4

        if 'ax_quality_scores' in ax_dict:
            ax_quality = ax_dict['ax_quality_scores']
            fig = None
        else:
            cols = None
            for result in results:
                if 'quality_scores' in result:
                    cols = len(result['quality_scores'])
                    break
            w, h = _get_plot_width_and_height(cols)
            fig, ax_quality = plt.subplots(figsize=(w, h), nrows=1)

        if sort_quality_scores_in_plot:
            order = np.argsort(results[0]['quality_scores'])
        else:
            order = np.arange(len(results[0]['quality_scores']))

        shift_count = 0
        for subjective_model, result in zip(subjective_models, results):
            if 'quality_scores' in result:
                quality = np.array(result['quality_scores'])
                quality = list(quality[order])
                xs = range(len(quality))

                label = subjective_model.TYPE

                if 'bic' in result and 'reconstructions' in result:
                    label += ' [NBIC {:.2f}]'.format(result['bic'])

                if plot_type == 'bar':
                    ax_quality.bar(np.array(xs)+shift_count*bar_width, quality,
                                width=bar_width,
                                color=colors[shift_count],
                                label=label)

                elif plot_type == 'errorbar':
                    if 'quality_scores_ci95' in result:
                        try:
                            quality_error = result['quality_scores_ci95']
                            label += ' [avg CI {:.2f}]'.format(np.mean(np.array(quality_error[0]) +
                                                                       np.array(quality_error[1])))
                        except TypeError:
                            quality_error = None
                        ax_quality.errorbar(np.array(xs)+shift_count*bar_width+0.2, quality,
                                            yerr=quality_error, fmt='.', capsize=2,
                                            color=colors[shift_count],
                                            label=label)
                    else:
                        ax_quality.plot(np.array(xs)+shift_count*bar_width+0.2, quality, '.',
                                    color=colors[shift_count],
                                    label=label)
                else:
                    raise AssertionError("Unknown plot_type: {}".format(plot_type))

                ax_quality.set_xlabel(r'Video Stimuli ($j$)')
                ax_quality.set_title(r'Recovered Quality Score ($\psi_j$)')
                ax_quality.set_xlim([min(xs), max(xs)+1])
                shift_count += 1

        ax_quality.grid()
        ax_quality.legend(ncol=2, frameon=True)
        if fig is not None:
            fig.tight_layout()

    if do_plot == 'all' or 'subject_scores' in do_plot:

        # ===== plot subject bias and inconsistency =====
        bar_width = 0.4
        if 'ax_observer_bias' in ax_dict and 'ax_observer_inconsistency' in ax_dict:
            ax_bias = ax_dict['ax_observer_bias']
            ax_inconsty = ax_dict['ax_observer_inconsistency']
            fig = None
        else:
            cols = None
            for result in results:
                if 'observer_bias' in result:
                    cols = len(result['observer_bias'])
                    break
            w, h = _get_plot_width_and_height(cols)
            h = h * 2 + 2
            fig, (ax_bias, ax_inconsty) = plt.subplots(figsize=(w, h), nrows=2, ncols=1, sharex=True)

        if 'ax_rejected' in ax_dict:
            ax_rejected = ax_dict['ax_rejected']
        else:
            ax_rejected = None

        if 'ax_rejected_1st_stats' in ax_dict:
            ax_rejected_1st_stats = ax_dict['ax_rejected_1st_stats']
        else:
            ax_rejected_1st_stats = None

        if 'ax_rejected_2nd_stats' in ax_dict:
            ax_rejected_2nd_stats = ax_dict['ax_rejected_2nd_stats']
        else:
            ax_rejected_2nd_stats = None

        xs = None
        shift_count = 0
        my_xticks = None
        for subjective_model, result in zip(subjective_models, results):

            if 'observer_bias' in result:
                bias = result['observer_bias']
                xs = range(len(bias))

                if plot_type == 'bar':
                    ax_bias.bar(np.array(xs)+shift_count*bar_width, bias,
                                width=bar_width,
                                color=colors[shift_count],
                                label=subjective_model.TYPE)
                elif plot_type == 'errorbar':
                    if 'observer_bias_ci95' in result:
                        try:
                            bias_error = result['observer_bias_ci95']
                            label = '{} [avg CI {:.2f}]'.format(
                                subjective_model.TYPE, np.mean(np.array(bias_error[0]) +
                                                               np.array(bias_error[1])))
                        except TypeError:
                            bias_error = None
                            label = subjective_model.TYPE
                        ax_bias.errorbar(np.array(xs)+shift_count*bar_width+0.2, bias,
                                         yerr=bias_error, fmt='.', capsize=2,
                                         color=colors[shift_count],
                                         label=label)
                    else:
                        ax_bias.plot(np.array(xs)+shift_count*bar_width+0.2, bias, '.',
                                     color=colors[shift_count],
                                     label=subjective_model.TYPE)
                else:
                    raise AssertionError("Unknown plot_type: {}".format(plot_type))

                ax_inconsty.set_xlim([min(xs), max(xs)+1])
                ax_bias.set_title(r'Subject Bias ($\Delta_i$)')
                ax_bias.legend(ncol=1, frameon=True)

                if 'observers' in result:
                    observers = result['observers']
                    assert len(bias) == len(observers)
                    my_xticks = observers
                    plt.sca(ax_inconsty)
                    plt.xticks(np.array(xs) + 0.01, my_xticks, rotation=90)

            if 'observer_inconsistency' in result:
                inconsty = result['observer_inconsistency']

                xs = range(len(inconsty))

                if plot_type == 'bar':
                    ax_inconsty.bar(np.array(xs)+shift_count*bar_width, inconsty,
                                    width=bar_width,
                                    color=colors[shift_count],
                                    label=subjective_model.TYPE)
                elif plot_type == 'errorbar':
                    if 'observer_inconsistency_ci95' in result:
                        try:
                            inconsistency_error = result['observer_inconsistency_ci95']
                            label = '{} [avg CI {:.2f}]'.format(
                                subjective_model.TYPE, np.mean(np.array(inconsistency_error[0]) +
                                                               np.array(inconsistency_error[1])))
                        except TypeError:
                            inconsistency_error = None
                            label = subjective_model.TYPE
                        ax_inconsty.errorbar(np.array(xs)+shift_count*bar_width+0.2, inconsty,
                                             yerr=inconsistency_error, fmt='.', capsize=2,
                                             color=colors[shift_count],
                                             label=label)
                    else:
                        ax_inconsty.plot(np.array(xs)+shift_count*bar_width+0.2, inconsty, '.',
                                         color=colors[shift_count],
                                         label=subjective_model.TYPE)
                else:
                    raise AssertionError("Unknown plot_type: {}".format(plot_type))

                ax_inconsty.set_xlim([min(xs), max(xs)+1])
                ax_inconsty.set_title(r'Subject Inconsistency ($\upsilon_i$)')
                ax_inconsty.legend(ncol=1, frameon=True)

            if 'observer_rejected' in result and ax_rejected is not None:

                rejected = np.array(result['observer_rejected']).astype(int)

                xs = range(len(rejected))
                ax_rejected.bar(np.array(xs) + shift_count * bar_width, rejected,
                                width=bar_width,
                                color=colors[shift_count],
                                label=subjective_model.TYPE)
                ax_rejected.set_xlim([min(xs), max(xs)+1])
                ax_rejected.set_title(r'Subject Rejected')
                ax_rejected.legend(ncol=1, frameon=True)

            if 'observer_rejected_1st_stats' in result and ax_rejected_1st_stats is not None:

                rejected = result['observer_rejected_1st_stats']

                xs = range(len(rejected))
                ax_rejected_1st_stats.bar(np.array(xs) + shift_count * bar_width, rejected,
                                width=bar_width,
                                color=colors[shift_count],
                                label=subjective_model.TYPE)
                ax_rejected_1st_stats.set_xlim([min(xs), max(xs)+1])
                ax_rejected_1st_stats.set_title(r'Subject Rejected (1st stats)')
                ax_rejected_1st_stats.legend(ncol=1, frameon=True)

            if 'observer_rejected_2nd_stats' in result and ax_rejected_2nd_stats is not None:

                rejected = result['observer_rejected_2nd_stats']

                xs = range(len(rejected))
                ax_rejected_2nd_stats.bar(np.array(xs) + shift_count * bar_width, rejected,
                                width=bar_width,
                                color=colors[shift_count],
                                label=subjective_model.TYPE)
                ax_rejected_2nd_stats.set_xlim([min(xs), max(xs)+1])
                ax_rejected_2nd_stats.set_title(r'Subject Rejected (2nd stats)')
                ax_rejected_2nd_stats.legend(ncol=1, frameon=True)

            if 'observer_bias' in result or 'observer_inconsistency' in result or 'observer_rejected' in result:
                shift_count += 1

        if xs and my_xticks is None:
            my_xticks = list(map(lambda x: "#{}".format(x+1), xs))
            plt.sca(ax_inconsty)
            plt.xticks(np.array(xs) + 0.3, my_xticks, rotation=90)
            if ax_rejected is not None:
                plt.sca(ax_rejected)
                plt.xticks(np.array(xs) + 0.3, my_xticks, rotation=90)
            if ax_rejected_1st_stats is not None:
                plt.sca(ax_rejected_1st_stats)
                plt.xticks(np.array(xs) + 0.3, my_xticks, rotation=90)
            if ax_rejected_2nd_stats is not None:
                plt.sca(ax_rejected_2nd_stats)
                plt.xticks(np.array(xs) + 0.3, my_xticks, rotation=90)

        ax_bias.grid()
        ax_inconsty.grid()
        if ax_rejected is not None:
            ax_rejected.grid()
        if ax_rejected_1st_stats is not None:
            ax_rejected_1st_stats.grid()
        if ax_rejected_2nd_stats is not None:
            ax_rejected_2nd_stats.grid()
        if fig is not None:
            fig.tight_layout()

    if do_plot == 'all' or 'content_scores' in do_plot:

        # ===== plot content ambiguity =====
        bar_width = 0.4

        if 'ax_content_ambiguity' in ax_dict:
            ax_ambgty = ax_dict['ax_content_ambiguity']
            fig = None
        else:
            cols = None
            for result in results:
                if 'content_ambiguity' in result:
                    cols = len(result['content_ambiguity'])
                    break
            w, h = _get_plot_width_and_height(cols)
            fig, ax_ambgty = plt.subplots(figsize=(w, h), nrows=1)
        xs = None
        shift_count = 0
        for subjective_model, result in zip(subjective_models, results):
            if 'content_ambiguity' in result:
                ambgty = result['content_ambiguity']
                xs = range(len(ambgty))

                if plot_type == 'bar':
                    ax_ambgty.bar(np.array(xs)+shift_count*bar_width, ambgty,
                                  width=bar_width,
                                  color=colors[shift_count],
                                  label=subjective_model.TYPE)
                elif plot_type == 'errorbar':
                    if 'content_ambiguity_ci95' in result:
                        try:
                            ambiguity_error = result['content_ambiguity_ci95']
                            label = '{} [avg CI {:.2f}]'.format(
                                subjective_model.TYPE, np.mean(np.array(ambiguity_error[0]) +
                                                               np.array(ambiguity_error[1])))
                        except TypeError:
                            ambiguity_error = None
                            label = subjective_model.TYPE
                        ax_ambgty.errorbar(np.array(xs)+shift_count*bar_width+0.2, ambgty,
                                           yerr=ambiguity_error, fmt='.', capsize=2,
                                           color=colors[shift_count],
                                           label=label)
                    else:
                        ax_ambgty.plot(np.array(xs)+shift_count*bar_width+0.2, ambgty, '.',
                                       color=colors[shift_count],
                                       label=subjective_model.TYPE)
                else:
                    raise AssertionError("Unknown plot_type: {}".format(plot_type))

                shift_count += 1
                ax_ambgty.set_title(r'Content Ambiguity ($\rho_k$)')
                ax_ambgty.grid()
        if xs:
            my_xticks = ['' for _ in range(len(xs))]
            for ref_video in dataset_reader.dataset.ref_videos:
                my_xticks[ref_video['content_id']] = ref_video['content_name']
            rotation = 90
            plt.sca(ax_ambgty)
            plt.xticks(np.array(xs) + 0.01, my_xticks, rotation=rotation)
        ax_ambgty.legend(ncol=2, frameon=True)
        if fig is not None:
            fig.tight_layout()

    return dataset, subjective_models, results


def _get_imshow_width_and_height(rows, cols):
    w, h = cols / 30, rows / 30
    if w > 100:
        x = w // 100
        w, h = w / x, h / x
    if h > 100:
        x = h // 100
        w, h = w / x, h / x
    w, h = w + 2, h + 2
    return w, h


def _get_plot_width_and_height(num: Union[int, None]) -> [int, int]:
    if num is None:
        return 10, 3
    else:
        return max(np.sqrt(num) + 2, 10), 3


def format_output_of_run_subjective_models(dataset, subjective_models, results):

    assert len(subjective_models) == len(results)
    for result in results:
        assert 'quality_scores' in result
        assert 'dis_video_names' in result
        assert len(result['dis_video_names']) == len(result['quality_scores'])
        if 'quality_scores_std' in result:
            assert len(result['quality_scores']) == len(result['quality_scores_std'])
        if 'quality_scores_ci95' in result:
            assert len(result['quality_scores_ci95']) == 2
            assert len(result['quality_scores']) == len(list(zip(*result['quality_scores_ci95'])))

        if 'observer_bias' in result:
            if 'observers' in result:
                assert len(result['observers']) == len(result['observer_bias'])
            else:
                result['observers'] = [f'observer{o}' for o in range(len(result['observer_bias']))]
            if 'observer_bias_std' in result:
                assert len(result['observer_bias']) == len(result['observer_bias_std'])
            if 'observer_bias_ci95' in result:
                assert len(result['observer_bias_ci95']) == 2
                assert len(result['observer_bias']) == len(list(zip(*result['observer_bias_ci95'])))
            if 'observer_inconsistency' in result:
                assert len(result['observer_bias']) == len(result['observer_inconsistency'])
            if 'observer_inconsistency_std' in result:
                assert len(result['observer_bias']) == len(result['observer_inconsistency_std'])
            if 'observer_inconsistency_ci95' in result:
                assert len(result['observer_inconsistency_ci95']) == 2
                assert len(result['observer_bias']) == len(list(zip(*result['observer_inconsistency_ci95'])))
            if 'observer_scores_mean' in result:
                assert len(result['observer_bias']) == len(result['observer_scores_mean'])
            if 'observer_scores_std' in result:
                assert len(result['observer_bias']) == len(result['observer_scores_std'])

        if 'content_ambiguity' in result:
            dict_contentid_content = dict()
            for ref_video in dataset.ref_videos:
                dict_contentid_content[ref_video['content_id']] = \
                    f"{ref_video['content_name']}"
            result['contents'] = [dict_contentid_content[k] for k in sorted(dict_contentid_content.keys())]
            assert len(result['contents']) == len(result['content_ambiguity'])
            if 'content_ambiguity_std' in result:
                assert len(result['content_ambiguity']) == len(result['content_ambiguity_std'])
            if 'content_ambiguity_ci95' in result:
                assert len(result['content_ambiguity_ci95']) == 2
                assert len(result['content_ambiguity']) == len(list(zip(*result['content_ambiguity_ci95'])))

    output = dict()
    for subjective_model, result in zip(subjective_models, results):
        for idx, dis_video_name in enumerate(result['dis_video_names']):
            if 'dis_video_name' in output.setdefault('dis_videos', dict()).setdefault(idx, dict()):
                assert output.setdefault('dis_videos', dict()).setdefault(idx, dict())['dis_video_name'] == dis_video_name
            else:
                output.setdefault('dis_videos', dict()).setdefault(idx, dict())['dis_video_name'] = dis_video_name
        for idx, quality_score in enumerate(result['quality_scores']):
            output.setdefault('dis_videos', dict()).setdefault(idx, dict()).setdefault('models', dict()).setdefault(subjective_model.TYPE, dict())['quality_score'] = quality_score
        if 'quality_scores_std' in result:
            for idx, quality_score_std in enumerate(result['quality_scores_std']):
                output.setdefault('dis_videos', dict()).setdefault(idx, dict()).setdefault('models', dict()).setdefault(subjective_model.TYPE, dict())['quality_score_std'] = quality_score_std
        if 'quality_scores_ci95' in result:
            for idx, quality_score_ci95 in enumerate(list(zip(*result['quality_scores_ci95']))):
                output.setdefault('dis_videos', dict()).setdefault(idx, dict()).setdefault('models', dict()).setdefault(subjective_model.TYPE, dict())['quality_score_ci95'] = quality_score_ci95

        if 'observer_bias' in result:
            if 'observers' in result:
                for idx, observer in enumerate(result['observers']):
                    if 'observer' in output.setdefault('observers', dict()).setdefault(idx, dict()):
                        assert output.setdefault('observers', dict()).setdefault(idx, dict())['observer'] == observer
                    else:
                        output.setdefault('observers', dict()).setdefault(idx, dict())['observer'] = observer
            for idx, observer_bias in enumerate(result['observer_bias']):
                output.setdefault('observers', dict()).setdefault(idx, dict()).setdefault('models', dict()).setdefault(subjective_model.TYPE, dict())['observer_bias'] = observer_bias
            if 'observer_bias_std' in result:
                for idx, observer_bias_std in enumerate(result['observer_bias_std']):
                    output.setdefault('observers', dict()).setdefault(idx, dict()).setdefault('models', dict()).setdefault(subjective_model.TYPE, dict())['observer_bias_std'] = observer_bias_std
            if 'observer_bias_ci95' in result:
                for idx, observer_bias_ci95 in enumerate(list(zip(*result['observer_bias_ci95']))):
                    output.setdefault('observers', dict()).setdefault(idx, dict()).setdefault('models', dict()).setdefault(subjective_model.TYPE, dict())['observer_bias_ci95'] = observer_bias_ci95
            if 'observer_inconsistency' in result:
                for idx, observer_inconsistency in enumerate(result['observer_inconsistency']):
                    output.setdefault('observers', dict()).setdefault(idx, dict()).setdefault('models', dict()).setdefault(subjective_model.TYPE, dict())['observer_inconsistency'] = observer_inconsistency
            if 'observer_inconsistency_std' in result:
                for idx, observer_inconsistency_std in enumerate(result['observer_inconsistency_std']):
                    output.setdefault('observers', dict()).setdefault(idx, dict()).setdefault('models', dict()).setdefault(subjective_model.TYPE, dict())['observer_inconsistency_std'] = observer_inconsistency_std
            if 'observer_inconsistency_ci95' in result:
                for idx, observer_inconsistency_ci95 in enumerate(list(zip(*result['observer_inconsistency_ci95']))):
                    output.setdefault('observers', dict()).setdefault(idx, dict()).setdefault('models', dict()).setdefault(subjective_model.TYPE, dict())['observer_inconsistency_ci95'] = observer_inconsistency_ci95
            if 'observer_scores_mean' in result:
                for idx, observer_scores_mean in enumerate(result['observer_scores_mean']):
                    output.setdefault('observers', dict()).setdefault(idx, dict()).setdefault('models', dict()).setdefault(subjective_model.TYPE, dict())['observer_scores_mean'] = observer_scores_mean
            if 'observer_scores_std' in result:
                for idx, observer_scores_std in enumerate(result['observer_scores_std']):
                    output.setdefault('observers', dict()).setdefault(idx, dict()).setdefault('models', dict()).setdefault(subjective_model.TYPE, dict())['observer_scores_std'] = observer_scores_std

        if 'content_ambiguity' in result:
            if 'contents' in result:
                for idx, content in enumerate(result['contents']):
                    if 'content' in output.setdefault('contents', dict()).setdefault(idx, dict()):
                        assert output.setdefault('contents', dict()).setdefault(idx, dict())['content'] == content
                    else:
                        output.setdefault('contents', dict()).setdefault(idx, dict())['content'] = content
            for idx, content_ambiguity in enumerate(result['content_ambiguity']):
                output.setdefault('contents', dict()).setdefault(idx, dict()).setdefault('models', dict()).setdefault(subjective_model.TYPE, dict())['content_ambiguity'] = content_ambiguity
            if 'content_ambiguity_std' in result:
                for idx, content_ambiguity_std in enumerate(result['content_ambiguity_std']):
                    output.setdefault('contents', dict()).setdefault(idx, dict()).setdefault('models', dict()).setdefault(subjective_model.TYPE, dict())['content_ambiguity_std'] = content_ambiguity_std
            if 'content_ambiguity_ci95' in result:
                for idx, content_ambiguity_ci95 in enumerate(list(zip(*result['content_ambiguity_ci95']))):
                    output.setdefault('contents', dict()).setdefault(idx, dict()).setdefault('models', dict()).setdefault(subjective_model.TYPE, dict())['content_ambiguity_ci95'] = content_ambiguity_ci95

        if 'aic' in result:
            output.setdefault('stats', dict()).setdefault('models', dict()).setdefault(subjective_model.TYPE, dict())['aic'] = result['aic']
        if 'bic' in result:
            output.setdefault('stats', dict()).setdefault('models', dict()).setdefault(subjective_model.TYPE, dict())['bic'] = result['bic']

    if 'dis_videos' in output:
        output['dis_videos'] = list(output['dis_videos'].values())
    if 'observers' in output:
        output['observers'] = list(output['observers'].values())
    if 'contents' in output:
        output['contents'] = list(output['contents'].values())

    return output


def visualize_pc_dataset(dataset_filepath):

    dataset = import_python_file(dataset_filepath)
    dataset_reader = PairedCompDatasetReader(dataset)
    tensor_pvs_pvs_subject = dataset_reader.opinion_score_3darray

    plt.figure()
    mtx_pvs_pvs = np.nansum(tensor_pvs_pvs_subject, axis=2) \
                  / (np.nansum(tensor_pvs_pvs_subject, axis=2) +
                     np.nansum(tensor_pvs_pvs_subject, axis=2).transpose())
    plt.imshow(mtx_pvs_pvs, interpolation='nearest')
    plt.title(r'Paired Comparison Winning Rate')
    plt.ylabel(r"PVS ($j$)")
    plt.xlabel(r"PVS ($j'$) [Compared Against]")
    plt.set_cmap('jet')
    plt.colorbar()
    plt.tight_layout()


def validate_with_synthetic_dataset(synthetic_dataset_reader_class,
                                    subjective_model_classes,
                                    dataset_filepath,
                                    synthetic_result,
                                    ax_dict,
                                    **more):
        ret = {}

        color_dict = more['color_dict'] if 'color_dict' in more else {}

        marker_dict = more['marker_dict'] if 'marker_dict' in more else {}

        output_synthetic_dataset_filepath = more['output_synthetic_dataset_filepath'] \
            if 'output_synthetic_dataset_filepath' in more else None

        missing_probability = more['missing_probability'] if 'missing_probability' in more else None
        assert missing_probability is None or 0 <= missing_probability < 1

        measure_runtime = more['measure_runtime'] if 'measure_runtime' in more else False
        assert isinstance(measure_runtime, bool)

        dataset = import_python_file(dataset_filepath)
        dataset_reader = synthetic_dataset_reader_class(dataset, input_dict=synthetic_result)

        if missing_probability is not None:
            dataset = dataset_reader.to_dataset()
            synthetic_result2 = copy.deepcopy(synthetic_result)
            synthetic_result2.update({'missing_probability': missing_probability})
            dataset_reader = MissingDataRawDatasetReader(dataset, input_dict=synthetic_result2)

        if output_synthetic_dataset_filepath is not None:
            dataset_reader.write_out_dataset(dataset_reader.to_dataset(), output_synthetic_dataset_filepath)
            ret['output_synthetic_dataset_filepath'] = output_synthetic_dataset_filepath

        subjective_models = map(
            lambda subjective_model_class: subjective_model_class(dataset_reader),
            subjective_model_classes
        )

        def run_modeling(subjective_model):
            if measure_runtime:
                with Timer() as t:
                    ret = subjective_model.run_modeling(**more)
                ret['runtime'] = t.interval
            else:
                ret = subjective_model.run_modeling(**more)
            return ret

        results = list(map(
            lambda subjective_model: run_modeling(subjective_model),
            subjective_models
        ))

        if ax_dict is None:
            ax_dict = dict()

        do_errorbar = more['do_errorbar'] if 'do_errorbar' in more else False
        assert isinstance(do_errorbar, bool)

        ret['results'] = dict()
        for subjective_model_class, result in zip(subjective_model_classes, results):
            ret['results'][subjective_model_class.TYPE] = result

        for ax in ax_dict.values():
            ax.set_xlabel('Synthetic')
            ax.set_ylabel('Recovered')
            ax.grid()

        for subjective_model_class, result, idx in zip(subjective_model_classes, results, range(len(results))):

            model_name = subjective_model_class.TYPE

            if 'quality_scores' in result and 'quality_scores' in synthetic_result and 'quality_scores_ci95' in result:
                ci_perc = get_ci_percentage(synthetic_result, result, 'quality_scores', 'quality_scores_ci95')
                ret['results'][model_name]['quality_scores_ci_perc'] = ci_perc

            if 'observer_bias' in result and 'observer_bias' in synthetic_result and 'observer_bias_ci95' in result:
                ci_perc = get_ci_percentage(synthetic_result, result, 'observer_bias', 'observer_bias_ci95')
                ret['results'][model_name]['observer_bias_ci_perc'] = ci_perc

            if 'observer_inconsistency' in result and 'observer_inconsistency' in synthetic_result \
                    and 'observer_inconsistency_ci95' in result:
                ci_perc = get_ci_percentage(synthetic_result, result,
                                            'observer_inconsistency', 'observer_inconsistency_ci95')
                ret['results'][model_name]['observer_inconsistency_ci_perc'] = ci_perc

            if 'quality_scores' in ax_dict:
                ax = ax_dict['quality_scores']
                ax.set_title(r'Quality Score ($\psi_j$)')
                if 'quality_scores' in result and 'quality_scores' in synthetic_result:
                    color = color_dict[model_name] if model_name in color_dict else 'black'
                    marker = marker_dict[model_name] if model_name in marker_dict else '.'
                    x = synthetic_result['quality_scores']
                    y = result['quality_scores']
                    if 'quality_scores_ci95' in result:
                        yerr = result['quality_scores_ci95']
                        ci_perc = get_ci_percentage(synthetic_result, result,  'quality_scores', 'quality_scores_ci95')
                    else:
                        yerr = None
                        ci_perc=None
                    if do_errorbar is True and 'quality_scores_ci95' in result:
                        ax.errorbar(x, y, yerr=yerr, color=color, capsize=2, marker=marker, linestyle='None',
                                    label='{sm} (RMSE {rmse:.4f}, CI% {ci_perc:.1f})'.format(
                                        sm=model_name,
                                        rmse=RmsePerfMetric(x, y).evaluate(enable_mapping=False)['score'],
                                        ci_perc=ci_perc,
                                    ))
                    else:
                        ax.scatter(x, y, color=color, marker=marker,
                                   label='{sm} (RMSE {rmse:.4f})'.format(
                                       sm=model_name, rmse=RmsePerfMetric(x, y).evaluate(enable_mapping=False)['score']))
                    ax.legend()
                    diag_line = np.arange(min(x), max(x), step=0.01)
                    ax.plot(diag_line, diag_line, '-', color='gray', )

            if 'quality_scores_std' in ax_dict:
                ax = ax_dict['quality_scores_std']
                ax.set_title(r'Std of Quality Score ($\sigma(\psi_j)$)')
                if 'quality_scores_std' in result and 'quality_scores_std' in synthetic_result:
                    color = color_dict[model_name] if model_name in color_dict else 'black'
                    marker = marker_dict[model_name] if model_name in marker_dict else '.'
                    x = synthetic_result['quality_scores_std']
                    y = result['quality_scores_std']
                    ax.scatter(x, y, color=color, marker=marker,
                               label='{sm} (RMSE {rmse:.4f})'.format(
                                   sm=model_name, rmse=RmsePerfMetric(x, y).evaluate(enable_mapping=False)['score']))
                    ax.legend()
                    diag_line = np.arange(min(x), max(x), step=0.01)
                    ax.plot(diag_line, diag_line, '-', color='gray', )

            if 'observer_bias' in ax_dict:
                ax = ax_dict['observer_bias']
                ax.set_title(r'Subject Bias ($\Delta_i$)')
                if 'observer_bias' in result and 'observer_bias' in synthetic_result:
                    color = color_dict[model_name] if model_name in color_dict else 'black'
                    marker = marker_dict[model_name] if model_name in marker_dict else '.'
                    x = synthetic_result['observer_bias']
                    y = result['observer_bias']
                    if 'observer_bias_ci95' in result:
                        yerr = result['observer_bias_ci95']
                        ci_perc = get_ci_percentage(synthetic_result, result, 'observer_bias', 'observer_bias_ci95')
                    else:
                        yerr = None
                        ci_perc = None
                    min_xy = np.min([len(x),len(y)])
                    x = x[:min_xy]
                    y = y[:min_xy]
                    if do_errorbar is True and 'observer_bias_ci95' in result:
                        ax.errorbar(x, y, yerr=yerr, color=color, capsize=2, marker=marker, linestyle='None',
                                    label='{sm} (RMSE {rmse:.4f}, CI% {ci_perc:.1f})'.format(
                                        sm=model_name,
                                        rmse=RmsePerfMetric(x, y).evaluate(enable_mapping=False)['score'],
                                        ci_perc=ci_perc,
                                    ))
                    else:
                        ax.scatter(x, y, color=color, marker=marker,
                                    label='{sm} (RMSE {rmse:.4f})'.format(
                                        sm=model_name,
                                        rmse=RmsePerfMetric(x, y).evaluate(enable_mapping=False)['score']))
                    ax.legend()
                    diag_line = np.arange(min(x), max(x), step=0.01)
                    ax.plot(diag_line, diag_line, '-', color='gray', )

            if 'observer_inconsistency' in ax_dict:
                ax = ax_dict['observer_inconsistency']
                ax.set_title(r'Subject Inconsistency ($\upsilon_i$)')
                if 'observer_inconsistency' in result and 'observer_inconsistency' in synthetic_result:
                    color = color_dict[model_name] if model_name in color_dict else 'black'
                    marker = marker_dict[model_name] if model_name in marker_dict else '.'
                    x = synthetic_result['observer_inconsistency']
                    y = result['observer_inconsistency']
                    if 'observer_inconsistency_ci95' in result:
                        yerr = np.array(result['observer_inconsistency_ci95'])
                        ci_perc = get_ci_percentage(synthetic_result, result,
                                                    'observer_inconsistency', 'observer_inconsistency_ci95')
                    else:
                        yerr = None
                        ci_perc = None
                    min_xy = np.min([len(x),len(y)])
                    x = x[:min_xy]
                    y = y[:min_xy]
                    if do_errorbar is True and 'observer_inconsistency_ci95' in result:
                        ax.errorbar(x, y, yerr=yerr, color=color, capsize=2, marker=marker, linestyle='None',
                                    label='{sm} (RMSE {rmse:.4f}, CI% {ci_perc:.1f})'.format(
                                        sm=model_name,
                                        rmse=RmsePerfMetric(x, y).evaluate(enable_mapping=False)['score'],
                                        ci_perc=ci_perc,
                                    ))
                    else:
                        ax.scatter(x, y, color=color, marker=marker,
                                   label='{sm} (RMSE {rmse:.4f})'.format(
                                       sm=model_name, rmse=RmsePerfMetric(x, y).evaluate(enable_mapping=False)['score']))
                    ax.legend()
                    diag_line = np.arange(min(x), max(x), step=0.01)
                    ax.plot(diag_line, diag_line, '-', color='gray', )

            if 'quality_ambiguity' in ax_dict:
                ax = ax_dict['quality_ambiguity']
                ax.set_title(r'Quality Ambiguity ($\phi_j$)')
                if 'quality_ambiguity' in result and 'quality_ambiguity' in synthetic_result:
                    color = color_dict[model_name] if model_name in color_dict else 'black'
                    marker = marker_dict[model_name] if model_name in marker_dict else '.'
                    x = synthetic_result['quality_ambiguity']
                    y = result['quality_ambiguity']
                    min_xy = np.min([len(x),len(y)])
                    x = x[:min_xy]
                    y = y[:min_xy]
                    ax.scatter(x, y, color=color, marker=marker,
                               label='{sm} (RMSE {rmse:.4f})'.format(
                                   sm=model_name, rmse=RmsePerfMetric(x, y).evaluate(enable_mapping=False)['score']))
                    ax.legend()
                    diag_line = np.arange(min(x), max(x), step=0.01)
                    ax.plot(diag_line, diag_line, '-', color='gray', )

        return ret


def get_ci_percentage(synthetic_result, result, key, errkey):
    xs = synthetic_result[key]
    ys = result[key]
    ys_ci95 = np.array(result[errkey])
    ys_ci95_shape = ys_ci95.shape
    assert ys_ci95_shape[0] == 2
    assert len(xs) == len(ys) == ys_ci95_shape[1]
    ind_in_ci = [y - y_ci95_l <= x <= y + y_ci95_u for x, y, y_ci95_l, y_ci95_u in zip(xs, ys, ys_ci95[0], ys_ci95[1])]
    ci_perc = sum(ind_in_ci) / len(ind_in_ci) * 100
    return ci_perc


def get_sample_stats(datasets, subjective_model_classes, do_plot=False, plot_type='bar', subj_fraction=None, random_seed=None):

    resultss = []  # dataset x subjective_model
    for dataset in datasets:
        if do_plot:
            fig_raw_scores, ax_raw_scores = plt.subplots(figsize=(7, 3))
            fig_quality_scores, ax_quality_scores = plt.subplots(figsize=(12, 3.5), nrows=1)

            # fig_bias_inconsty, [ax_bias, ax_inconsty] = plt.subplots(figsize=(6, 5), nrows=2, ncols=1, sharex=True)
            # fig_rejected, ax_rejected = plt.subplots(figsize=(6, 3), nrows=1)
            fig_bias_inconsty, [ax_bias, ax_inconsty, ax_rejected] = plt.subplots(figsize=(6, 7), nrows=3, ncols=1, sharex=True)
            fig_rejected = None

            ax_dict = {
                'ax_raw_scores': ax_raw_scores,
                'ax_quality_scores': ax_quality_scores,
                'ax_observer_bias': ax_bias,
                'ax_observer_inconsistency': ax_inconsty,
                'ax_rejected': ax_rejected,
            }
            do_plot_list = [
                'raw_scores',
                'quality_scores',
                'subject_scores'
            ]
        else:
            fig_raw_scores = None
            fig_quality_scores = None
            fig_bias_inconsty = None
            fig_rejected = None
            ax_dict = {}
            do_plot_list = []

        dataset_filepath = dataset['path']
        dataset_reader_info_dict = {}
        if subj_fraction is None:
            dataset, subjective_models, results = run_subjective_models(
                dataset_filepath=dataset_filepath,
                subjective_model_classes=subjective_model_classes,
                normalize_final=False,  # True or False
                do_plot=do_plot_list,
                plot_type=plot_type,
                ax_dict=ax_dict,
                dataset_reader_info_dict=dataset_reader_info_dict
            )
            resultss.append(results)
        else:
            assert 0.0 < subj_fraction <= 1.0

            if dataset_filepath.endswith('.py'):
                dataset = import_python_file(dataset_filepath)
            elif dataset_filepath.endswith('.json'):
                dataset = import_json_file(dataset_filepath)
            else:
                raise AssertionError("Unknown input type, must be .py or .json")
            dataset_reader = RawDatasetReader(dataset, input_dict=dataset_reader_info_dict)
            num_subj = dataset_reader.num_observers
            frac_num_subj = int(num_subj * subj_fraction)
            if random_seed is not None:
                    np.random.seed(random_seed)
            selected_subjects = np.random.choice(list(range(num_subj)), size=frac_num_subj, replace=False)
            dataset_reader_info_dict_new = dataset_reader_info_dict.copy()
            dataset_reader_info_dict_new['selected_subjects'] = selected_subjects
            dataset, subjective_models, results = run_subjective_models(
                dataset_filepath=dataset_filepath,
                subjective_model_classes=subjective_model_classes,
                normalize_final=False,  # True or False
                do_plot=do_plot_list,
                plot_type=plot_type,
                ax_dict=ax_dict,
                dataset_reader_info_dict=dataset_reader_info_dict_new,
                dataset_reader_class=SelectSubjectRawDatasetReader,
            )
            resultss.append(results)

        if fig_raw_scores is not None:
            fig_raw_scores.tight_layout()
        if fig_quality_scores is not None:
            fig_quality_scores.tight_layout()
        if fig_bias_inconsty is not None:
            fig_bias_inconsty.tight_layout()
        if fig_rejected is not None:
            fig_rejected.tight_layout()

    return resultss


def plot_scatter_target_vs_compared_models(target_models, compared_models, datasets,
                                           target_subj_fraction=None,
                                           compared_subj_fraction=None,
                                           random_seed=None,
                                           get_sample_stats_wrapper_method=None
                                           ):

    if get_sample_stats_wrapper_method is None:
        get_sample_stats_wrapper_method = _get_sample_stats_wrapper

    for dataset in datasets:

        target_resultss = []
        for model in target_models:
            results_target = get_sample_stats_wrapper_method(
                dataset,
                target_subj_fraction,
                model, random_seed)
            target_resultss.append(results_target)

        compared_resultss = []
        for model in compared_models:
            results_compared = get_sample_stats_wrapper_method(
                dataset,
                compared_subj_fraction,
                model, random_seed)
            compared_resultss.append(results_compared)

        fig, axss = plt.subplots(ncols=len(compared_resultss), nrows=2,
                                 figsize=[5 * len(compared_resultss), 7],
                                 gridspec_kw={'height_ratios': [3, 1]})
        axss = axss.T
        if len(compared_resultss) == 1:
            axss = np.array([axss])

        for target_model, target_results, compapred_model, compared_results, axs in zip(target_models, target_resultss, compared_models, compared_resultss, axss):
            target_results = target_results[0]
            compared_results = compared_results[0]
            for target_result, compared_result \
                    in zip(target_results, compared_results):
                ax_scatter, ax_hist = axs
                xs = target_result['quality_scores']
                ys = compared_result['quality_scores']
                diffs = np.array(ys) - np.array(xs)
                mean_diff = np.mean(diffs)
                std_diff = np.std(diffs)
                plcc = scipy.stats.pearsonr(xs, ys)[0]
                srocc = scipy.stats.spearmanr(xs, ys)[0]
                label = os.path.splitext(os.path.basename(dataset['path']))[0][:30]
                compared_tag = compapred_model
                if compared_subj_fraction is not None:
                    compared_tag += f" {int(compared_subj_fraction * 100)}% data"
                target_tag = target_model
                if target_subj_fraction is not None:
                    target_tag += f" {int(target_subj_fraction * 100)}% data"
                ax_scatter.scatter(xs, ys, alpha=0.2, label=label)
                ax_scatter.set_xlabel(target_tag)
                ax_scatter.set_ylabel(compared_tag)
                ax_scatter.grid()
                ax_scatter.legend()
                ax_hist.hist(diffs, label=label)
                ax_hist.set_title(
                    f"plcc {plcc:.3f}, srocc {srocc:.3f}, diff: mean {mean_diff:.3f}, std {std_diff:.3f}")
                ax_hist.set_xlabel(
                    f"diff: {compared_tag} vs. {target_tag}")
                ax_hist.set_ylabel(f"No. ocurrences")
                ax_hist.grid()
                ax_hist.legend()
            plt.tight_layout()


def _get_sample_stats_wrapper(dataset, subj_fraction, model, random_seed):
    model_class = SubjectiveModel.find_subclass(model)
    results = get_sample_stats(
        [dataset],
        [model_class],
        do_plot=False,
        subj_fraction=subj_fraction,
        random_seed=random_seed,
    )  # dataset x subjective_model
    for result in results:
        if 'raw_scores' in result[0]:
            del result[0]['raw_scores']
        if 'reconstructions' in result[0]:
            del result[0]['reconstructions']

    return results


@persist_to_dir(SurealConfig.workdir_path('_get_sample_stats_wrapper'))
def _get_sample_stats_wrapper_persistent(dataset, subj_fraction, model, random_seed):
    return _get_sample_stats_wrapper(dataset, subj_fraction, model, random_seed)
