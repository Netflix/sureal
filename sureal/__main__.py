import argparse
import json
import os

from sureal.routine import run_subjective_models, \
    format_output_of_run_subjective_models
from sureal.subjective_model import SubjectMLEModelProjectionSolver2, \
    SubjrejMosModel, BiasremvSubjrejMosModel, SubjectiveModel


class BT500Model(SubjrejMosModel):
    TYPE = 'BT500'


class P913124Model(BiasremvSubjrejMosModel):
    TYPE = 'P913'


class P910AnnexEModel(SubjectMLEModelProjectionSolver2):
    TYPE = 'P910'


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
    parser
    args = parser.parse_args()
    dataset = args.dataset[0]
    models = args.models
    output_dir = args.output_dir[0] if args.output_dir else None

    ModelClasses = list()
    for model in models:
        ModelClass = SubjectiveModel.find_subclass(model)
        ModelClasses.append(ModelClass)

    dataset, subjective_models, results = run_subjective_models(
        dataset_filepath=dataset,
        subjective_model_classes=ModelClasses,
    )

    output = format_output_of_run_subjective_models(
        dataset, subjective_models, results)

    if output_dir is None:
        json_output = json.dumps(output, indent=4)
        print(json_output)
    else:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'output.json'), 'w') as fp:
            json.dump(output, fp, indent=4)

    return 0


if __name__ == '__main__':
    ret = main()
    exit(ret)
