#!/usr/bin/env python3

import argparse
import json
import os

from sureal.config import DisplayConfig
from sureal.dataset_reader import RawDatasetReader, PairedCompDatasetReader
from sureal.pc_subjective_model import PairedCompSubjectiveModel
from sureal.routine import run_subjective_models, \
    format_output_of_run_subjective_models
from sureal.subjective_model import SubjectiveModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", dest="dataset", nargs=1, type=str,
        help="Path to the dataset file.",
        required=True)
    parser.add_argument(
        "--models", dest="models", nargs="+", type=str,
        help="Subjective models to use (can specify more than one), choosing "
             "from: MOS, P910, P913, BT500.",
        required=True)
    parser.add_argument(
        "--output-dir", dest="output_dir", nargs=1, type=str,
        help="Path to the output directory (will force create is not existed). "
             "If not specified, plots will be displayed and output will be printed.",
        required=False)
    parser.add_argument(
        "--plot-raw-data", dest="plot_raw_data", action='store_true',
        help="Plot the raw data. This includes the raw opinion scores presented "
             "in a video-subject matrix, counts per video and counts per subject.",
        required=False)
    parser.add_argument(
        "--plot-dis-videos", dest="plot_dis_videos", action='store_true',
        help="Plot the subjective scores of the distorted videos.",
        required=False)
    parser.add_argument(
        "--plot-observers", dest="plot_observers", action='store_true',
        help="Plot the scores of the observers.",
        required=False)
    args = parser.parse_args()
    dataset = args.dataset[0]
    models = args.models
    output_dir = args.output_dir[0] if args.output_dir else None
    plot_raw_data = args.plot_raw_data
    plot_dis_videos = args.plot_dis_videos
    plot_observers = args.plot_observers

    ModelClasses = list()
    for model in models:
        ModelClass = SubjectiveModel.find_subclass(model)
        ModelClasses.append(ModelClass)

    def is_subj_model_class(ModelClass):
        superclasses = ModelClass.__mro__
        return SubjectiveModel in superclasses and PairedCompSubjectiveModel not in superclasses

    def is_pc_subj_model_class(ModelClass):
        return PairedCompSubjectiveModel in ModelClass.__mro__

    # ModelClass should be either SubjectiveModel or PairedCompSubjectiveModel
    is_all_subjective_model = all([is_subj_model_class(ModelClass) for ModelClass in ModelClasses])
    is_all_pc_subjective_model = all([is_pc_subj_model_class(ModelClass) for ModelClass in ModelClasses])
    assert (is_all_subjective_model and not is_all_pc_subjective_model) or \
           (not is_all_subjective_model and is_all_pc_subjective_model), \
        f'is_all_subjective_model: {is_all_subjective_model}, ' \
        f'is_all_pc_subjective_model: {is_all_pc_subjective_model}'
    if is_all_subjective_model:
        DatasetReaderClass = RawDatasetReader
    else:
        DatasetReaderClass = PairedCompDatasetReader

    do_plot = []
    if plot_raw_data:
        do_plot.append('raw_scores')
        if is_all_subjective_model:
            do_plot.append('raw_counts')
            do_plot.append('raw_counts_per_subject')
    if plot_dis_videos:
        do_plot.append('quality_scores')
    if plot_observers:
        do_plot.append('subject_scores')

    dataset, subjective_models, results = run_subjective_models(
        dataset_filepath=dataset,
        subjective_model_classes=ModelClasses,
        do_plot=do_plot,
        dataset_reader_class=DatasetReaderClass,
    )

    output = format_output_of_run_subjective_models(
        dataset, subjective_models, results)

    if output_dir is None:
        json_output = json.dumps(output, indent=4)
        print(json_output)
        DisplayConfig.show()
    else:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'output.json'), 'w') as fp:
            json.dump(output, fp, indent=4)
        DisplayConfig.show(write_to_dir=output_dir)
        print(f'output is written to directory {output_dir}.')

    return 0


if __name__ == '__main__':
    ret = main()
    exit(ret)
