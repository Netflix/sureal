import numpy as np

from sureal.dataset_reader import SelectSubjectRawDatasetReader
from sureal.subjective_model import MaximumLikelihoodEstimationModel


class MaximumLikelihoodEstimationModelWithBootstrapping(MaximumLikelihoodEstimationModel):

    TYPE = 'MLE_BSTP'
    VERSION = MaximumLikelihoodEstimationModel.VERSION + "_0.1"

    DEFAULT_N_BOOTSTRAP = 10

    @classmethod
    def _run_modeling(cls, dataset_reader, **kwargs):
        result = super(MaximumLikelihoodEstimationModelWithBootstrapping, cls).\
            _run_modeling(dataset_reader, **kwargs)
        dataset = dataset_reader.to_dataset()
        n_subj = dataset_reader.num_observers

        n_bootstrap = kwargs['n_bootstrap'] if 'n_bootstrap' in kwargs \
                                               and kwargs['n_bootstrap'] is not None else cls.DEFAULT_N_BOOTSTRAP
        assert isinstance(n_bootstrap, int) and n_bootstrap > 0

        bootstrap_results = []
        for ibootstrap in range(n_bootstrap):
            np.random.seed(ibootstrap)
            selected_subjects = np.random.choice(range(n_subj), size=n_subj, replace=True)

            select_subj_reader = SelectSubjectRawDatasetReader(
                dataset, input_dict={'selected_subjects': selected_subjects})

            bootstrap_result = super(MaximumLikelihoodEstimationModelWithBootstrapping, cls).\
                _run_modeling(select_subj_reader, **kwargs)
            bootstrap_results.append(bootstrap_result)

        bootstrap_quality_scoress = [r['quality_scores'] for r in bootstrap_results]
        bootstrap_quality_scoress = np.array(bootstrap_quality_scoress)
        result['quality_scores_ci95'] = [
            np.array(result['quality_scores']) - np.percentile(bootstrap_quality_scoress, 2.5, axis=0),
            np.percentile(bootstrap_quality_scoress, 97.5, axis=0) - np.array(result['quality_scores'])
        ]

        return result


class MaximumLikelihoodEstimationModelContentObliviousWithBootstrapping(MaximumLikelihoodEstimationModelWithBootstrapping):
    TYPE = 'MLE_CO_BSTP' # maximum likelihood estimation (no content modeling) with bootstrapping
    VERSION = MaximumLikelihoodEstimationModelWithBootstrapping.VERSION + "_0.1"
    mode = 'CONTENT_OBLIVIOUS'