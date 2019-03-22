import numpy as np
from matplotlib import pyplot as plt

from sureal.dataset_reader import RawDatasetReader, PairedCompDatasetReader
from sureal.tools.misc import import_python_file, import_json_file
from vmaf.tools.misc import import_python_file

__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"


def run_subjective_models(dataset_filepath, subjective_model_classes, do_plot=None, **kwargs):

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

    if do_plot == 'all' or 'raw_scores' in do_plot:
        # ===== plot raw scores
        plt.figure(figsize=(5, 2.5))
        mtx = dataset_reader.opinion_score_2darray.T
        S, E = mtx.shape
        plt.imshow(mtx, interpolation='nearest')
        # xs = np.array(range(S)) + 1
        # my_xticks = map(lambda x: "#{}".format(x), xs)
        # plt.yticks(np.array(xs), my_xticks, rotation=0)
        plt.title(r'Raw Opinion Scores ($x_{es}$)')
        plt.xlabel(r'Impaired Video Encodes ($e$)')
        plt.ylabel(r'Test Subjects ($s$)')
        plt.set_cmap('gray')
        plt.tight_layout()

    if do_plot == 'all' or 'quality_scores' in do_plot:
        # ===== plot quality scores =====
        bar_width = 0.4
        fig, ax_quality = plt.subplots(figsize=(10, 2.5), nrows=1)
        xs = None
        shift_count = 0
        my_xticks = None
        for subjective_model, result in zip(subjective_models, results):
            if 'quality_scores' in result:
                quality = result['quality_scores']
                xs = range(len(quality))

                # plt.plot(result['quality_scores'], label=subjective_model.TYPE)

                if plot_type == 'bar':
                    ax_quality.bar(np.array(xs)+shift_count*bar_width, quality,
                                width=bar_width,
                                color=colors[shift_count],
                                label=subjective_model.TYPE)
                elif plot_type == 'errorbar':
                    if 'quality_scores_std' in result:
                        quality_error = np.array(result['quality_scores_std']) * 1.96 # 95% C.I.
                        ax_quality.errorbar(np.array(xs)+shift_count*bar_width+0.2, quality,
                                            yerr=quality_error, fmt='.',
                                            color=colors[shift_count],
                                            label=subjective_model.TYPE)
                    else:
                        ax_quality.plot(np.array(xs)+shift_count*bar_width+0.2, quality, '.',
                                    color=colors[shift_count],
                                    label=subjective_model.TYPE)
                else:
                    raise AssertionError("Unknown plot_type: {}".format(plot_type))

                ax_quality.set_xlabel(r'Impaired Video Encodes ($e$)')
                ax_quality.set_title(r'Recovered Quality Score ($x_e$)')
                ax_quality.set_xlim([min(xs), max(xs)+1])
                shift_count += 1
        ax_quality.grid()
        ax_quality.legend(loc=1, ncol=2, frameon=True)
        plt.tight_layout()

    if do_plot == 'all' or 'subject_scores' in do_plot:

        # ===== plot subject bias and inconsistency =====
        bar_width = 0.4
        figsize = (5, 3.5)
        # figsize = (7, 10)
        fig, (ax_bias, ax_inconsty) = plt.subplots(figsize=figsize, nrows=2, sharex=True)
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
                    if 'observer_bias_std' in result:
                        bias_error = np.array(result['observer_bias_std']) * 1.96 # 95% C.I.
                        ax_bias.errorbar(np.array(xs)+shift_count*bar_width+0.2, bias,
                                         yerr=bias_error, fmt='.',
                                         color=colors[shift_count],
                                         label=subjective_model.TYPE)
                    else:
                        ax_bias.plot(np.array(xs)+shift_count*bar_width+0.2, bias, '.',
                                     color=colors[shift_count],
                                     label=subjective_model.TYPE)
                else:
                    raise AssertionError("Unknown plot_type: {}".format(plot_type))

                ax_inconsty.set_xlim([min(xs), max(xs)+1])
                ax_bias.set_title(r'Subject Bias ($b_s$)')
                ax_bias.grid()

                if 'observers' in result:
                    observers = result['observers']
                    assert len(bias) == len(observers)
                    my_xticks = observers
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
                    if 'observer_inconsistency_std' in result:
                        inconsistency_error = np.array(result['observer_inconsistency_std']) * 1.96 # 95% C.I.
                        ax_inconsty.errorbar(np.array(xs)+shift_count*bar_width+0.2, inconsty,
                                             yerr=inconsistency_error, fmt='.',
                                             color=colors[shift_count],
                                             label=subjective_model.TYPE)
                    else:
                        ax_inconsty.plot(np.array(xs)+shift_count*bar_width+0.2, inconsty, '.',
                                         color=colors[shift_count],
                                         label=subjective_model.TYPE)
                else:
                    raise AssertionError("Unknown plot_type: {}".format(plot_type))

                ax_inconsty.set_xlim([min(xs), max(xs)+1])
                ax_inconsty.set_title(r'Subject Inconsisency ($v_s$)')
                ax_inconsty.legend(loc=2, ncol=2, frameon=True)
                ax_inconsty.grid()

            if 'observer_bias' in result:
                shift_count += 1

        if xs and my_xticks is None:
            my_xticks = map(lambda x: "#{}".format(x+1), xs)
            plt.xticks(np.array(xs) + 0.3, my_xticks, rotation=90)
        plt.tight_layout()

    if do_plot == 'all' or 'content_scores' in do_plot:

        # ===== plot content ambiguity =====
        bar_width = 0.4
        fig, ax_ambgty = plt.subplots(figsize=(5, 3.5), nrows=1)
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
                    if 'content_ambiguity_std' in result:
                        ambiguity_error = np.array(result['content_ambiguity_std']) * 1.96 # 95% C.I.
                        ax_ambgty.errorbar(np.array(xs)+shift_count*bar_width+0.2, ambgty,
                                           yerr=ambiguity_error, fmt='.',
                                           color=colors[shift_count],
                                           label=subjective_model.TYPE)
                    else:
                        ax_ambgty.plot(np.array(xs)+shift_count*bar_width+0.2, ambgty, '.',
                                       color=colors[shift_count],
                                       label=subjective_model.TYPE)
                else:
                    raise AssertionError("Unknown plot_type: {}".format(plot_type))

                shift_count += 1
                ax_ambgty.set_title(r'Content Ambiguity ($a_c$)')
                ax_ambgty.grid()
        if xs:
            my_xticks = ['' for _ in range(len(xs))]
            for ref_video in dataset_reader.dataset.ref_videos:
                my_xticks[ref_video['content_id']] = ref_video['content_name']
            # rotation = 75
            rotation = 90
            plt.xticks(np.array(xs) + 0.01, my_xticks, rotation=rotation)
        ax_ambgty.legend(loc=1, ncol=2, frameon=True)
        plt.tight_layout()

    return dataset, subjective_models, results


def visualize_pc_dataset(dataset_filepath):

    dataset = import_python_file(dataset_filepath)
    dataset_reader = PairedCompDatasetReader(dataset)
    tensor_pvs_pvs_subject = dataset_reader.opinion_score_3darray

    plt.figure()
    # plot the rate of winning x, 0 <= x <= 1.0, of one PVS compared against another PVS
    mtx_pvs_pvs = np.nansum(tensor_pvs_pvs_subject, axis=2) \
                  / (np.nansum(tensor_pvs_pvs_subject, axis=2) +
                     np.nansum(tensor_pvs_pvs_subject, axis=2).transpose())
    plt.imshow(mtx_pvs_pvs, interpolation='nearest')
    plt.title(r'Paired Comparison Winning Rate')
    plt.ylabel(r'PVS ($e$)')
    plt.xlabel(r'PVS ($f$) [Compared Against]')
    plt.set_cmap('jet')
    plt.colorbar()
    plt.tight_layout()