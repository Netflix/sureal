#!/usr/bin/env python

import os
import sys

from sureal.subjective_model import SubjectiveModel
from sureal.routine import run_subjective_models
from sureal.tools.misc import get_file_name_with_extension, get_cmd_option
from sureal.config import DisplayConfig

__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"

SUBJECTIVE_MODELS = ['MOS', 'MLE', 'MLE_CO', 'DMOS', 'DMOS_MLE', 'DMOS_MLE_CO', 'SR_MOS', 'ZS_SR_MOS', 'SR_DMOS', 'ZS_SR_DMOS']

def print_usage():
    print "usage: " + os.path.basename(sys.argv[0]) + " subjective_model dataset_filepath [--output-dir output_dir]\n"
    print "subjective_model:\n\t" + "\n\t".join(SUBJECTIVE_MODELS) + "\n"

def main():
    if len(sys.argv) < 3:
        print_usage()
        return 2

    try:
        subjective_model = sys.argv[1]
        dataset_filepath = sys.argv[2]
    except ValueError:
        print_usage()
        return 2

    output_dir = get_cmd_option(sys.argv, 3, len(sys.argv), '--output-dir')

    do_plot = ['raw_scores', 'quality_scores']
    if subjective_model in ['MLE', 'MLE_CO', 'DMOS_MLE', 'DMOS_MLE_CO']:
        do_plot.append('subject_scores')
    if subjective_model in ['MLE', 'DMOS_MLE']:
        do_plot.append('content_scores')

    try:
        subjective_model_class = SubjectiveModel.find_subclass(subjective_model)
    except Exception as e:
        print "Error: " + str(e)
        return 1

    print "Run model {} on dataset {}".format(
        subjective_model_class.__name__, get_file_name_with_extension(dataset_filepath)
    )

    run_subjective_models(
        dataset_filepath=dataset_filepath,
        subjective_model_classes = [subjective_model_class,],
        normalize_final=False, # True or False
        do_plot=do_plot,
        plot_type='errorbar',
        gradient_method='simplified',
    )

    if output_dir is None:
        DisplayConfig.show()
    else:
        print("Output wrote to {}.".format(output_dir))
        DisplayConfig.show(write_to_dir=output_dir)

    return 0

if __name__ == '__main__':
    ret = main()
    exit(ret)
