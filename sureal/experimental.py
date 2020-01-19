import copy

import numpy as np

from sureal.dataset_reader import SelectSubjectRawDatasetReader, SelectDisVideoRawDatasetReader
from sureal.subjective_model import MaximumLikelihoodEstimationModel


class MaximumLikelihoodEstimationModelWithBootstrapping(MaximumLikelihoodEstimationModel):

    TYPE = 'MLE_BSTP'
    VERSION = MaximumLikelihoodEstimationModel.VERSION + "_0.1"

    DEFAULT_N_BOOTSTRAP = 30
    DEFAULT_BOOTSTRAP_SUBJECTS = True
    DEFAULT_BOOSTRAP_DIS_VIDEOS = True

    @classmethod
    def _run_modeling(cls, dataset_reader, **kwargs):

        force_subjbias_zeromean = kwargs['force_subjbias_zeromean'] \
            if 'force_subjbias_zeromean' in kwargs \
               and kwargs['force_subjbias_zeromean'] is not None else MaximumLikelihoodEstimationModel.DEFAULT_FORCE_SUBJBIAS_ZEROMEAN
        assert isinstance(force_subjbias_zeromean, bool)
        new_kwargs = copy.deepcopy(kwargs)
        new_kwargs['force_subjbias_zeromean'] = False

        result = super(MaximumLikelihoodEstimationModelWithBootstrapping, cls).\
            _run_modeling(dataset_reader, **new_kwargs)

        dataset = dataset_reader.to_dataset()
        n_subj = dataset_reader.num_observers
        n_disvideo = dataset_reader.num_dis_videos

        n_bootstrap = kwargs['n_bootstrap'] if 'n_bootstrap' in kwargs \
                                               and kwargs['n_bootstrap'] is not None else cls.DEFAULT_N_BOOTSTRAP
        assert isinstance(n_bootstrap, int) and n_bootstrap > 0

        bootstrap_subjects = kwargs['bootstrap_subjects'] \
            if 'bootstrap_subjects' in kwargs \
               and kwargs['bootstrap_subjects'] is not None else cls.DEFAULT_BOOTSTRAP_SUBJECTS
        assert isinstance(bootstrap_subjects, bool)

        boostrap_dis_videos = kwargs['boostrap_dis_videos'] \
            if 'boostrap_dis_videos' in kwargs \
               and kwargs['boostrap_dis_videos'] is not None else cls.DEFAULT_BOOSTRAP_DIS_VIDEOS
        assert isinstance(boostrap_dis_videos, bool)

        if bootstrap_subjects:
            quality_scores_ci95 = \
                cls._bootstrap_subjects(dataset, result, n_subj, n_bootstrap, new_kwargs)
            result['quality_scores_ci95'] = quality_scores_ci95
        else:
            del result['quality_scores_ci95']

        if boostrap_dis_videos:
            observer_bias_ci95, observer_inconsistency_ci95 = \
                cls._boostrap_dis_videos(dataset, result, n_disvideo, n_bootstrap, new_kwargs)
            result['observer_bias_ci95'] = observer_bias_ci95
            result['observer_inconsistency_ci95'] = observer_inconsistency_ci95
        else:
            del result['observer_bias_ci95']
            del result['observer_inconsistency_ci95']

        if force_subjbias_zeromean is True:
            assert 'quality_scores' in result
            assert 'observer_bias' in result
            mean_b_s = np.mean(result['observer_bias'])
            result['observer_bias'] = list(np.array(result['observer_bias']) - mean_b_s)
            result['quality_scores'] = list(np.array(result['quality_scores']) + mean_b_s)

        return result

    @classmethod
    def _bootstrap_subjects(cls, dataset, result, n_subj, n_bootstrap, kwargs):
        bootstrap_results = []
        for ibootstrap in range(n_bootstrap):
            print(f"Bootstrap with seed {ibootstrap}")

            np.random.seed(ibootstrap)
            selected_subjects = np.random.choice(range(n_subj), size=n_subj, replace=True)

            select_subj_reader = SelectSubjectRawDatasetReader(
                dataset, input_dict={'selected_subjects': selected_subjects})

            bootstrap_result = super(MaximumLikelihoodEstimationModelWithBootstrapping, cls). \
                _run_modeling(select_subj_reader, **kwargs)

            bootstrap_observer_bias_offset = np.mean(
                np.array(bootstrap_result['observer_bias']) -
                np.array(result['observer_bias'])[selected_subjects]
            )

            bootstrap_result['observer_bias'] = list(np.array(bootstrap_result['observer_bias']) -
                                                     bootstrap_observer_bias_offset)
            bootstrap_result['quality_scores'] = list(np.array(bootstrap_result['quality_scores']) +
                                                      bootstrap_observer_bias_offset)

            bootstrap_results.append(bootstrap_result)
        bootstrap_quality_scoress = np.array([r['quality_scores'] for r in bootstrap_results])
        quality_scores_ci95 = [
            np.array(result['quality_scores']) - np.percentile(bootstrap_quality_scoress, 2.5, axis=0),
            np.percentile(bootstrap_quality_scoress, 97.5, axis=0) - np.array(result['quality_scores'])
        ]
        return quality_scores_ci95

    @classmethod
    def _boostrap_dis_videos(cls, dataset, result, n_dis_videos, n_bootstrap, kwargs):
        bootstrap_results = []
        for ibootstrap in range(n_bootstrap):
            print(f"Bootstrap with seed {ibootstrap}")

            np.random.seed(ibootstrap)
            selected_dis_videos = np.random.choice(range(n_dis_videos), size=n_dis_videos, replace=True)

            select_dis_video_reader = SelectDisVideoRawDatasetReader(
                dataset, input_dict={'selected_dis_videos': selected_dis_videos})

            bootstrap_result = super(MaximumLikelihoodEstimationModelWithBootstrapping, cls). \
                _run_modeling(select_dis_video_reader, **kwargs)

            bootstrap_quality_scores_offset = np.mean(
                np.array(bootstrap_result['quality_scores']) -
                np.array(result['quality_scores'])[selected_dis_videos]
            )

            bootstrap_result['quality_scores'] = list(np.array(bootstrap_result['quality_scores']) -
                                                      bootstrap_quality_scores_offset)
            bootstrap_result['observer_bias'] = list(np.array(bootstrap_result['observer_bias']) +
                                                     bootstrap_quality_scores_offset)

            bootstrap_results.append(bootstrap_result)
        bootstrap_observer_biass = np.array([r['observer_bias'] for r in bootstrap_results])
        observer_bias_ci95 = [
            np.array(result['observer_bias']) - np.percentile(bootstrap_observer_biass, 2.5, axis=0),
            np.percentile(bootstrap_observer_biass, 97.5, axis=0) - np.array(result['observer_bias'])
        ]
        bootstrap_observer_inconsistencys = np.array([r['observer_inconsistency'] for r in bootstrap_results])
        observer_inconsistency_ci95 = [
            np.array(result['observer_inconsistency']) - np.percentile(bootstrap_observer_inconsistencys, 2.5, axis=0),
            np.percentile(bootstrap_observer_inconsistencys, 97.5, axis=0) - np.array(result['observer_inconsistency'])
        ]
        return observer_bias_ci95, observer_inconsistency_ci95


class MaximumLikelihoodEstimationModelContentObliviousWithBootstrapping(MaximumLikelihoodEstimationModelWithBootstrapping):
    TYPE = 'MLE_CO_BSTP' # maximum likelihood estimation (no content modeling) with bootstrapping
    VERSION = MaximumLikelihoodEstimationModelWithBootstrapping.VERSION + "_0.1"
    mode = 'CONTENT_OBLIVIOUS'