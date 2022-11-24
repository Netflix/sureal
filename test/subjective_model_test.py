import os
import unittest
import warnings

import numpy as np
from sureal.config import SurealConfig
from sureal.dataset_reader import RawDatasetReader, MissingDataRawDatasetReader, \
    SyntheticRawDatasetReader, CorruptSubjectRawDatasetReader
from sureal.subjective_model import MosModel, DmosModel, \
    LegacyMaximumLikelihoodEstimationModel, MaximumLikelihoodEstimationModel, \
    LiveDmosModel, MaximumLikelihoodEstimationDmosModel, LeastSquaresModel, \
    SubjrejMosModel, ZscoringSubjrejMosModel, SubjrejDmosModel, \
    ZscoringSubjrejDmosModel, PerSubjectModel, \
    MaximumLikelihoodEstimationModelContentOblivious, \
    MaximumLikelihoodEstimationModelSubjectOblivious, ZscoringMosModel, BiasremvMosModel, BiasremvSubjrejMosModel, \
    SubjrejMosModelPearson, SubjrejMosModelSpearman, BiasremvSubjrejMosModelPearson, BiasremvSubjrejMosModelSpearman, \
    SubjectMLEModelProjectionSolver, SubjectMLEModelProjectionSolver2
from sureal.tools.misc import import_python_file, MyTestCase

__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"


class SubjectiveModelTest(MyTestCase):

    def setUp(self):
        super().setUp()
        self.dataset_filepath = SurealConfig.test_resource_path('NFLX_dataset_public_raw.py')
        self.output_dataset_filepath = SurealConfig.workdir_path('NFLX_dataset_public_test.py')
        self.output_dataset_pyc_filepath = SurealConfig.workdir_path('NFLX_dataset_public_test.pyc')

    def tearDown(self):
        if os.path.exists(self.output_dataset_filepath):
            os.remove(self.output_dataset_filepath)
        if os.path.exists(self.output_dataset_pyc_filepath):
            os.remove(self.output_dataset_pyc_filepath)
        super().tearDown()

    def test_mos_subjective_model(self):
        dataset = import_python_file(self.dataset_filepath)
        dataset_reader = RawDatasetReader(dataset)
        subjective_model = MosModel(dataset_reader)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling()
        self.assertTrue('observers' not in result)
        scores = result['quality_scores']
        self.assertAlmostEqual(scores[0], 4.884615384615385, places=4)
        self.assertAlmostEqual(scores[10], 2.0769230769230771, places=4)
        self.assertAlmostEqual(float(np.mean(scores)), 3.544790652385589, places=4)
        scores_std = result['quality_scores_std']
        self.assertAlmostEqual(float(np.mean(scores_std)), 0.12986637295658307, places=4)
        quality_ambiguity = result['quality_ambiguity']
        self.assertAlmostEqual(float(np.mean(quality_ambiguity)), 0.6621911698651353, places=4)

        self.assertAlmostEqual(result['dof'], 0.07692307692307693, places=6)
        self.assertAlmostEqual(result['loglikelihood'], -0.9384709649191117, places=6)
        self.assertAlmostEqual(float(np.std(result['raw_scores'])), 1.3654128030298962, places=6)
        self.assertAlmostEqual(float(np.std(result['reconstructions'])), 1.183732241710059, places=6)
        self.assertAlmostEqual(result['aic'], 2.0307880836843775, places=6)
        self.assertAlmostEqual(result['bic'], 2.4636761137219545, places=6)

    def test_mos_subjective_model_output(self):
        dataset = import_python_file(self.dataset_filepath)
        dataset_reader = RawDatasetReader(dataset)
        subjective_model = MosModel(dataset_reader)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            subjective_model.run_modeling()
        subjective_model.to_aggregated_dataset_file(self.output_dataset_filepath)
        self.assertTrue(os.path.exists(self.output_dataset_filepath))
        dataset2 = import_python_file(self.output_dataset_filepath)
        dis_video = dataset2.dis_videos[0]
        self.assertTrue('groundtruth' in dis_video)
        self.assertTrue('groundtruth_std' in dis_video)
        self.assertTrue('os' not in dis_video)
        self.assertAlmostEqual(dis_video['groundtruth'], 4.884615384615385, places=4)
        self.assertAlmostEqual(dis_video['groundtruth_std'], 0.08461538461538462, places=4)

    def test_mos_subjective_model_output_aggregate_content_ids(self):
        dataset = import_python_file(self.dataset_filepath)
        dataset_reader = RawDatasetReader(dataset)
        subjective_model = MosModel(dataset_reader)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            subjective_model.run_modeling()
        subjective_model.to_aggregated_dataset_file(self.output_dataset_filepath, aggregate_content_ids=[0, 2])
        self.assertTrue(os.path.exists(self.output_dataset_filepath))
        dataset2 = import_python_file(self.output_dataset_filepath)
        dis_video = dataset2.dis_videos[0]
        self.assertTrue('groundtruth' in dis_video)
        self.assertTrue('groundtruth_std' in dis_video)
        self.assertTrue('os' not in dis_video)
        self.assertAlmostEqual(dis_video['groundtruth'], 4.884615384615385, places=4)
        self.assertAlmostEqual(dis_video['groundtruth_std'], 0.08461538461538462, places=4)

    def test_mos_subjective_model_output_aggregate_asset_ids(self):
        dataset = import_python_file(self.dataset_filepath)
        dataset_reader = RawDatasetReader(dataset)
        subjective_model = MosModel(dataset_reader)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            subjective_model.run_modeling()
        subjective_model.to_aggregated_dataset_file(self.output_dataset_filepath, aggregate_asset_ids=[0, 2])
        self.assertTrue(os.path.exists(self.output_dataset_filepath))
        dataset2 = import_python_file(self.output_dataset_filepath)
        dis_video = dataset2.dis_videos[0]
        self.assertTrue('groundtruth' in dis_video)
        self.assertTrue('groundtruth_std' in dis_video)
        self.assertTrue('os' not in dis_video)
        self.assertAlmostEqual(dis_video['groundtruth'], 4.884615384615385, places=4)
        self.assertAlmostEqual(dis_video['groundtruth_std'], 0.08461538461538462, places=4)

    def test_mos_subjective_model_output_os_is_dict_style(self):
        dataset = import_python_file(SurealConfig.test_resource_path('test_dataset_os_as_dict.py'))
        dataset_reader = RawDatasetReader(dataset)
        subjective_model = MosModel(dataset_reader)
        result = subjective_model.run_modeling()
        self.assertTrue('observers' in result)
        subjective_model.to_aggregated_dataset_file(self.output_dataset_filepath)
        self.assertTrue(os.path.exists(self.output_dataset_filepath))
        dataset2 = import_python_file(self.output_dataset_filepath)
        dis_video = dataset2.dis_videos[0]
        print(dataset2.dis_videos)
        self.assertTrue('groundtruth' in dis_video)
        self.assertTrue('groundtruth_std' in dis_video)
        self.assertTrue('os' not in dis_video)
        self.assertAlmostEqual(dis_video['groundtruth'], 2.6666666666666665, places=4)
        self.assertAlmostEqual(dis_video['groundtruth_std'], 0.881917103688197, places=4)

        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 7.333333333333332, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 0.09876543209876538, places=4)

    def test_mos_subjective_model_output_custom_resampling(self):
        dataset = import_python_file(self.dataset_filepath)
        dataset_reader = RawDatasetReader(dataset)
        subjective_model = MosModel(dataset_reader)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            subjective_model.run_modeling()
        subjective_model.to_aggregated_dataset_file(self.output_dataset_filepath, resampling_type='lanczos')
        self.assertTrue(os.path.exists(self.output_dataset_filepath))
        dataset2 = import_python_file(self.output_dataset_filepath)
        self.assertFalse(hasattr(dataset2, 'quality_height'))
        self.assertFalse(hasattr(dataset2, 'quality_width'))
        self.assertEqual(dataset2.resampling_type, 'lanczos')
        dis_video = dataset2.dis_videos[0]
        self.assertTrue('groundtruth' in dis_video)
        self.assertTrue('groundtruth_std' in dis_video)
        self.assertTrue('os' not in dis_video)
        self.assertAlmostEqual(dis_video['groundtruth'], 4.884615384615385, places=4)
        self.assertAlmostEqual(dis_video['groundtruth_std'], 0.08461538461538462, places=4)

    def test_mos_subjective_model_output2(self):
        dataset = import_python_file(self.dataset_filepath)
        dataset_reader = RawDatasetReader(dataset)
        subjective_model = MosModel(dataset_reader)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            subjective_model.run_modeling()
        dataset2 = subjective_model.to_aggregated_dataset()
        dis_video = dataset2.dis_videos[0]
        self.assertTrue('groundtruth' in dis_video)
        self.assertTrue('groundtruth_std' in dis_video)
        self.assertTrue('os' not in dis_video)
        self.assertAlmostEqual(dis_video['groundtruth'], 4.884615384615385, places=4)
        self.assertAlmostEqual(dis_video['groundtruth_std'], 0.08461538461538462, places=4)

    def test_mos_subjective_model_normalize_final(self):
        dataset = import_python_file(self.dataset_filepath)
        dataset_reader = RawDatasetReader(dataset)
        subjective_model = MosModel(dataset_reader)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling(normalize_final=True)
        scores = result['quality_scores']
        self.assertAlmostEqual(scores[0], 1.1318646945818083, places=4)
        self.assertAlmostEqual(scores[10], -1.2400334499143002, places=4)
        self.assertAlmostEqual(float(np.mean(scores)), 0.0, places=4)

    def test_mos_subjective_model_transform_final(self):
        dataset = import_python_file(self.dataset_filepath)
        dataset_reader = RawDatasetReader(dataset)
        subjective_model = MosModel(dataset_reader)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling(transform_final={'p1': 10, 'p0': 1})
        scores = result['quality_scores']
        self.assertAlmostEqual(scores[0], 49.84615384615385, places=4)
        self.assertAlmostEqual(scores[10], 21.769230769230771, places=4)
        self.assertAlmostEqual(float(np.mean(scores)), 36.44790652385589, places=4)

    def test_from_dataset_file(self):
        subjective_model = MosModel.from_dataset_file(self.dataset_filepath)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling()
        scores = result['quality_scores']
        self.assertAlmostEqual(scores[0], 4.884615384615385, places=4)
        self.assertAlmostEqual(scores[10], 2.0769230769230771, places=4)
        self.assertAlmostEqual(float(np.mean(scores)), 3.544790652385589, places=4)

    def test_dmos_subjective_model(self):
        subjective_model = DmosModel.from_dataset_file(self.dataset_filepath)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling()
        self.assertTrue('observers' not in result)
        scores = result['quality_scores']
        self.assertAlmostEqual(scores[0], 5.0, places=4)
        self.assertAlmostEqual(scores[10], 2.1923076923076921, places=4)
        self.assertAlmostEqual(float(np.mean(scores)), 3.7731256085686473, places=4)
        scores_std = result['quality_scores_std']
        self.assertAlmostEqual(float(np.mean(scores_std)), 0.12986637295658307, places=4)

    def test_dmos_subjective_model_normalize_final(self):
        subjective_model = DmosModel.from_dataset_file(self.dataset_filepath)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling(normalize_final=True)
        scores = result['quality_scores']
        self.assertAlmostEqual(scores[0], 1.0440613892053001, places=4)
        self.assertAlmostEqual(scores[10], -1.3452648137895296, places=4)
        self.assertAlmostEqual(float(np.mean(scores)), 0.0, places=4)

    def test_dmos_subjective_model_dscore_mode_same(self):
        subjective_model = DmosModel.from_dataset_file(self.dataset_filepath)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling(normalize_final=True)
        scores = result['quality_scores']
        self.assertAlmostEqual(scores[0], 1.0440613892053001, places=4)
        self.assertAlmostEqual(scores[10], -1.3452648137895296, places=4)
        self.assertAlmostEqual(float(np.mean(scores)), 0.0, places=4)

    def test_observer_aware_subjective_model_with_dscoring(self):
        subjective_model = LegacyMaximumLikelihoodEstimationModel.from_dataset_file(
            self.dataset_filepath)
        result = subjective_model.run_modeling(dscore_mode=True, force_subjbias_zeromean=False)
        self.assertTrue('observers' not in result)

        self.assertAlmostEqual(float(np.sum(result['observer_bias'])), -0.090840910829083799, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_bias'])), 0.089032585621095089, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency'])), 15.681766163430936, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_inconsistency'])), 0.012565584832977776, places=4)

        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 298.35293969059796, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 1.4163670233392607, places=4)

    def test_observer_aware_subjective_model_with_zscoring(self):
        subjective_model = LegacyMaximumLikelihoodEstimationModel.from_dataset_file(
            self.dataset_filepath)
        result = subjective_model.run_modeling(zscore_mode=True, force_subjbias_zeromean=False)

        self.assertAlmostEqual(float(np.sum(result['observer_bias'])), 0.0, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_bias'])), 0.0, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency'])), 11.568205661696393, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_inconsistency'])), 0.0079989301785523791, places=4)

        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 0.0, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 0.80942484781493518, places=4)

    def test_observer_aware_subjective_model_with_dscoring_and_zscoring(self):
        subjective_model = LegacyMaximumLikelihoodEstimationModel.from_dataset_file(
            self.dataset_filepath)
        result = subjective_model.run_modeling(dscore_mode=True, zscore_mode=True, force_subjbias_zeromean=False)

        self.assertAlmostEqual(float(np.sum(result['observer_bias'])), 0.0, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_bias'])), 0.0, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency'])), 11.628499078069273, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_inconsistency'])), 0.0082089371266301642, places=4)

        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 0.0, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 0.80806512456121071, places=4)

    def test_observer_aware_subjective_model_use_log(self):
        subjective_model = LegacyMaximumLikelihoodEstimationModel.from_dataset_file(
            self.dataset_filepath)
        result = subjective_model.run_modeling(use_log=True, force_subjbias_zeromean=False)

        self.assertAlmostEqual(float(np.sum(result['observer_bias'])), -0.082429594509296211, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_bias'])), 0.089032585621095089, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency'])), 15.681766163430936, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_inconsistency'])), 0.012565584832977776, places=4)

        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 280.2889206910113, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 1.4355485462027884, places=4)

    def test_observer_content_aware_subjective_model(self):
        subjective_model = MaximumLikelihoodEstimationModel.from_dataset_file(
            self.dataset_filepath)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling(force_subjbias_zeromean=False)
        self.assertTrue('observers' not in result)

        self.assertAlmostEqual(float(np.sum(result['content_ambiguity'])), 3.8972884776604402, places=4)
        self.assertAlmostEqual(float(np.var(result['content_ambiguity'])), 0.0041122094732031289, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_bias'])), -0.055712761348815837, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_bias'])), 0.085842891905121704, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency'])), 10.164665557559516, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_inconsistency'])), 0.028749990587721687, places=4)

        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 280.20774261173619, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 1.4351342153719635, places=4)

        self.assertAlmostEqual(float(np.sum(result['content_ambiguity_std'])), 0.30465244947706538, places=4)
        self.assertAlmostEqual(float(np.sum(result['observer_bias_std'])), 1.7392847550878989, places=4)
        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency_std'])), 22.108576292956428, places=4)
        self.assertAlmostEqual(float(np.sum(result['quality_scores_std'])), 8.8863877635750423, places=4)

        self.assertAlmostEqual(result['dof'], 0.06815968841285297, places=6)
        self.assertAlmostEqual(result['loglikelihood'], -0.8897673811562866, places=6)
        self.assertAlmostEqual(float(np.std(result['raw_scores'])), 1.3654128030298962, places=6)
        self.assertAlmostEqual(float(np.std(result['reconstructions'])), 1.2332790063154353, places=6)
        self.assertAlmostEqual(result['aic'], 1.915854139138279, places=6)
        self.assertAlmostEqual(result['bic'], 2.299425811323474, places=6)

    def test_observer_content_aware_subjective_model_subjbias_zeromean(self):
        subjective_model = MaximumLikelihoodEstimationModel.from_dataset_file(
            self.dataset_filepath)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling()

        self.assertAlmostEqual(float(np.sum(result['content_ambiguity'])), 3.8972884776604402, places=4)
        self.assertAlmostEqual(float(np.var(result['content_ambiguity'])), 0.0041122094732031289, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_bias'])), 0.0, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_bias'])), 0.085842891905121704, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency'])), 10.164665557559516, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_inconsistency'])), 0.028749990587721687, places=4)

        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 280.0384615291764, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 1.4351342153719635, places=4)

        self.assertAlmostEqual(float(np.sum(result['content_ambiguity_std'])), 0.30465244947706538, places=4)
        self.assertAlmostEqual(float(np.sum(result['observer_bias_std'])), 1.7392847550878989, places=4)
        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency_std'])), 22.108576292956428, places=4)
        self.assertAlmostEqual(float(np.sum(result['quality_scores_std'])), 8.8863877635750423, places=4)

    def test_observer_content_aware_subjective_model_original(self):
        subjective_model = MaximumLikelihoodEstimationModel.from_dataset_file(
            self.dataset_filepath)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling(gradient_method='original', force_subjbias_zeromean=False)

        self.assertAlmostEqual(float(np.sum(result['content_ambiguity'])), 3.8972884776604402, places=4)
        self.assertAlmostEqual(float(np.var(result['content_ambiguity'])), 0.0041122094732031289, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_bias'])), -0.055712761348815837, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_bias'])), 0.085842891905121704, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency'])), 10.164665557559516, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_inconsistency'])), 0.028749990587721687, places=4)

        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 280.20774261173619, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 1.4351342153719635, places=4)

        self.assertAlmostEqual(float(np.sum(result['content_ambiguity_std'])), 0.30465244947706538, places=4)
        self.assertAlmostEqual(float(np.sum(result['observer_bias_std'])), 1.7392847550878989, places=4)
        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency_std'])), 22.108576292956428, places=4)
        self.assertAlmostEqual(float(np.sum(result['quality_scores_std'])), 8.8863877635750423, places=4)

    def test_observer_content_aware_subjective_model_numerical(self):
        subjective_model = MaximumLikelihoodEstimationModel.from_dataset_file(
            self.dataset_filepath)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling(gradient_method='numerical', force_subjbias_zeromean=False)

        self.assertAlmostEqual(float(np.sum(result['content_ambiguity'])), 3.8972884776604402, places=4)
        self.assertAlmostEqual(float(np.var(result['content_ambiguity'])), 0.0041122094732031289, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_bias'])), -0.055712761348815837, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_bias'])), 0.085842891905121704, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency'])), 10.164665557559516, places=3)
        self.assertAlmostEqual(float(np.var(result['observer_inconsistency'])), 0.028749990587721687, places=4)

        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 280.20774261173619, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 1.4351342153719635, places=4)

        self.assertAlmostEqual(float(np.sum(result['content_ambiguity_std'])), 0.30465244947706538, places=4)
        self.assertAlmostEqual(float(np.sum(result['observer_bias_std'])), 1.7392847550878989, places=4)
        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency_std'])), 12.393285044624955, places=4)
        self.assertAlmostEqual(float(np.sum(result['quality_scores_std'])), 8.8863877635750423, places=4)

    def test_observer_content_aware_subjective_model_missingdata(self):

        dataset = import_python_file(self.dataset_filepath)

        np.random.seed(0)
        info_dict = {
            'missing_probability': 0.1,
        }
        dataset_reader = MissingDataRawDatasetReader(dataset, input_dict=info_dict)

        subjective_model = MaximumLikelihoodEstimationModel(dataset_reader)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling(force_subjbias_zeromean=False)

        self.assertAlmostEqual(float(np.sum(result['content_ambiguity'])), 3.9104244772977128, places=4)
        self.assertAlmostEqual(float(np.var(result['content_ambiguity'])), 0.0037713583509767193, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_bias'])), -0.21903272050455846, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_bias'])), 0.084353684687185043, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency'])), 9.8168943054654481, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_inconsistency'])), 0.028159236075789944, places=4)

        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 280.05548186797336, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 1.4339487982797514, places=4)

        np.random.seed(0)
        info_dict = {
            'missing_probability': 0.5,
        }
        dataset_reader = MissingDataRawDatasetReader(dataset, input_dict=info_dict)

        subjective_model = MaximumLikelihoodEstimationModel(dataset_reader)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling(force_subjbias_zeromean=False)

        self.assertAlmostEqual(float(np.sum(result['content_ambiguity'])), 2.63184284168883, places=4)
        self.assertAlmostEqual(float(np.var(result['content_ambiguity'])), 0.019164097909450246, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_bias'])), 0.2263148440748638, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_bias'])), 0.070613033112114504, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency'])), 12.317917502439435, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_inconsistency'])), 0.029455722248727296, places=4)

        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 280.29962156788139, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 1.4717366222424826, places=4)

    def test_observer_content_aware_subjective_model_nocontent(self):
        subjective_model = MaximumLikelihoodEstimationModelContentOblivious.from_dataset_file(
            self.dataset_filepath)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling(force_subjbias_zeromean=False)

        self.assertAlmostEqual(float(np.sum(result['observer_bias'])), -0.090840910829083799, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_bias'])), 0.089032585621095089, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency'])), 15.681766163430936, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_inconsistency'])), 0.012565584832977776, places=4)

        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 280.31447815213642, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 1.4355485462027884, places=4)

        self.assertAlmostEqual(result['dof'], 0.06377799415774099, places=6)
        self.assertAlmostEqual(result['loglikelihood'], -0.8967394355890235, places=6)
        self.assertAlmostEqual(float(np.std(result['raw_scores'])), 1.3654128030298962, places=6)
        self.assertAlmostEqual(float(np.std(result['reconstructions'])), 1.2347392971084559, places=6)
        self.assertAlmostEqual(result['aic'], 1.921034859493529, places=6)
        self.assertAlmostEqual(result['bic'], 2.2799483527525326, places=6)

        self.assertAlmostEqual(float(np.sum(result['observer_bias_std'])), 1.7643365374531321, places=4)
        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency_std'])), 1.2475743287658851, places=4)
        self.assertAlmostEqual(float(np.sum(result['quality_scores_std'])), 8.907545016644042, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_bias_ci95'])), 6.916058079893282, places=4)
        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency_ci95'])), 5.002792923339208, places=4)
        self.assertAlmostEqual(float(np.sum(result['quality_scores_ci95'])), 34.91686386164329, places=4)

    def test_observer_content_aware_subjective_model_nocontent_subjbias_zeromean(self):
        subjective_model = MaximumLikelihoodEstimationModelContentOblivious.from_dataset_file(
            self.dataset_filepath)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling()

        self.assertAlmostEqual(float(np.sum(result['observer_bias'])), 0.0, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_bias'])), 0.089032585621095089, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency'])), 15.681766163430936, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_inconsistency'])), 0.012565584832977776, places=4)

        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 280.0384615384633, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 1.4355485462027884, places=4)

        self.assertAlmostEqual(result['dof'], 0.06377799415774099, places=6)
        self.assertAlmostEqual(result['loglikelihood'], -0.8967394355890235, places=6)
        self.assertAlmostEqual(float(np.std(result['raw_scores'])), 1.3654128030298962, places=6)
        self.assertAlmostEqual(float(np.std(result['reconstructions'])), 1.2347392971084559, places=6)
        self.assertAlmostEqual(result['aic'], 1.921034859493529, places=6)
        self.assertAlmostEqual(result['bic'], 2.2799483527525326, places=6)

    def test_observer_content_aware_subjective_model_nosubject(self):
        subjective_model = MaximumLikelihoodEstimationModelSubjectOblivious.from_dataset_file(
            self.dataset_filepath)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling(force_subjbias_zeromean=False)

        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 280.0384615384616, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 1.4012220200639218, places=4)

        self.assertAlmostEqual(float(np.sum(result['content_ambiguity'])), 6.06982228334157, places=4)
        self.assertAlmostEqual(float(np.var(result['content_ambiguity'])), 0.0045809756997836721, places=4)

        self.assertAlmostEqual(result['dof'], 0.042843232716650435, places=6)
        self.assertAlmostEqual(result['loglikelihood'], -1.02419628655795, places=6)
        self.assertAlmostEqual(float(np.std(result['raw_scores'])), 1.3654128030298962, places=6)
        self.assertAlmostEqual(float(np.std(result['reconstructions'])), 1.183732241710059, places=6)
        self.assertAlmostEqual(result['aic'], 2.1340790385492006, places=6)
        self.assertAlmostEqual(result['bic'], 2.3751812324941803, places=6)

    def test_observer_aware_subjective_model_synthetic(self):

        np.random.seed(0)

        dataset = import_python_file(self.dataset_filepath)
        info_dict = {
            'quality_scores': np.random.uniform(1, 5, 79),
            'observer_bias': np.random.normal(0, 1, 26),
            'observer_inconsistency': np.abs(np.random.uniform(0.4, 0.6, 26)),
            'content_bias': np.zeros(9),
            'content_ambiguity': np.zeros(9),
        }
        dataset_reader = SyntheticRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = LegacyMaximumLikelihoodEstimationModel(dataset_reader)
        result = subjective_model.run_modeling(force_subjbias_zeromean=False)

        self.assertAlmostEqual(float(np.sum(result['observer_bias'])), -0.90138622499935517, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_bias'])), 0.84819162765420342, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency'])), 12.742288471632817, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_inconsistency'])), 0.0047638169604076975, places=4)

        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 236.78529213581052, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 1.3059726132293354, places=4)

    def test_observer_aware_subjective_model(self):
        subjective_model = LegacyMaximumLikelihoodEstimationModel.from_dataset_file(
            self.dataset_filepath)
        result = subjective_model.run_modeling(force_subjbias_zeromean=False)

        self.assertAlmostEqual(float(np.sum(result['observer_bias'])), -0.090840910829083799, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_bias'])), 0.089032585621095089, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency'])), 15.681766163430936, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_inconsistency'])), 0.012565584832977776, places=4)

        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 280.31447815213642, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 1.4355485462027884, places=4)

    def test_observer_aware_subjective_model_missingdata(self):

        dataset = import_python_file(self.dataset_filepath)

        np.random.seed(0)
        info_dict = {
            'missing_probability': 0.1,
        }
        dataset_reader = MissingDataRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = LegacyMaximumLikelihoodEstimationModel(dataset_reader)
        result = subjective_model.run_modeling(force_subjbias_zeromean=False)

        self.assertAlmostEqual(float(np.sum(result['observer_bias'])), -0.18504017984241944, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_bias'])), 0.087350553292201705, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency'])), 15.520738471447299, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_inconsistency'])), 0.010940587327083341, places=4)

        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 279.94975274863879, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 1.4325574378911554, places=4)

        np.random.seed(0)
        info_dict = {
            'missing_probability': 0.5,
        }
        dataset_reader = MissingDataRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = LegacyMaximumLikelihoodEstimationModel(dataset_reader)
        result = subjective_model.run_modeling(force_subjbias_zeromean=False)

        self.assertAlmostEqual(float(np.sum(result['observer_bias'])), 0.057731868199093525, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_bias'])), 0.081341845650928557, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency'])), 14.996238224489693, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_inconsistency'])), 0.013666025579465165, places=4)

        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 280.67100837103203, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 1.4637917512768972, places=4)

    def test_livedmos_subjective_model(self):
        subjective_model = LiveDmosModel.from_dataset_file(self.dataset_filepath)
        result = subjective_model.run_modeling()
        scores = result['quality_scores']
        self.assertAlmostEqual(scores[0], 65.307711974116913, places=4)
        self.assertAlmostEqual(scores[10], 30.204773267864258, places=4)
        self.assertAlmostEqual(float(np.mean(scores)), 50.0, places=4)

    def test_livedmos_subjective_model_normalize_final(self):
        subjective_model = LiveDmosModel.from_dataset_file(self.dataset_filepath)
        result = subjective_model.run_modeling(normalize_final=True)
        scores = result['quality_scores']
        self.assertAlmostEqual(scores[0], 1.0392964273048528, places=4)
        self.assertAlmostEqual(scores[10], -1.3439701802061783, places=4)
        self.assertAlmostEqual(float(np.mean(scores)), 0.0, places=4)

    def test_livedmos_subjective_model_dscore_mode_bad(self):
        subjective_model = LiveDmosModel.from_dataset_file(self.dataset_filepath)
        with self.assertRaises(AssertionError):
            subjective_model.run_modeling(dscore_mode=True)

    def test_observer_aware_subjective_model_corruptdata(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = LegacyMaximumLikelihoodEstimationModel(dataset_reader)
        result = subjective_model.run_modeling(force_subjbias_zeromean=False)

        self.assertAlmostEqual(float(np.mean(result['quality_scores'])), 3.5573073781669944, places=4) # 3.5482845335713469
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 1.3559834438740614, places=4) # 1.4355485462027884

    def test_mos_subjective_model_corruptdata(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = MosModel(dataset_reader)
        result = subjective_model.run_modeling()
        scores = result['quality_scores']

        self.assertAlmostEqual(float(np.mean(scores)), 3.5447906523855899, places=4)
        self.assertAlmostEqual(float(np.var(scores)), 0.95893305294535369, places=4) # 1.4012220200639218

    def test_mos_subjective_model_corruptdata_subjreject(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = MosModel(dataset_reader)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling(subject_rejection=True)
        scores = result['quality_scores']

        self.assertAlmostEqual(float(np.mean(scores)), 3.5611814345991566, places=4)
        self.assertAlmostEqual(float(np.var(scores)), 1.1049505732699529, places=4) # 1.4012220200639218

    def test_zscore_mos_subjective_model_corruptdata_subjreject(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = MosModel(dataset_reader)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling(zscore_mode=True, subject_rejection=True)
        scores = result['quality_scores']

        self.assertAlmostEqual(float(np.mean(scores)), 0.0, places=4)
        self.assertAlmostEqual(float(np.var(scores)), 0.66670826882879042, places=4)

    def test_observer_aware_subjective_model_subjreject(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = LegacyMaximumLikelihoodEstimationModel(dataset_reader)
        with self.assertRaises(AssertionError):
            subjective_model.run_modeling(subject_rejection=True)

    def test_observer_content_aware_subjective_model_subjreject(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = MaximumLikelihoodEstimationModel(dataset_reader)
        with self.assertRaises(AssertionError):
            subjective_model.run_modeling(subject_rejection=True, force_subjbias_zeromean=False)

    def test_observer_content_aware_subjective_dmos_model(self):
        subjective_model = MaximumLikelihoodEstimationDmosModel.from_dataset_file(
            self.dataset_filepath)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling(force_subjbias_zeromean=False)

        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 288.56842946051466, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 1.4166132275824235, places=4)

        self.assertAlmostEqual(float(np.sum(result['content_ambiguity'])), 3.8972884776604402, places=4)
        self.assertAlmostEqual(float(np.var(result['content_ambiguity'])), 0.0041122094732031289, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_bias'])), 3.1293776428507774, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_bias'])), 0.085842891905121704, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency'])), 10.164665557559516, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_inconsistency'])), 0.028749990587721687, places=4)

    def test_dmos_mle_co_model(self):
        subjective_model = MaximumLikelihoodEstimationModelContentOblivious.from_dataset_file(
            self.dataset_filepath)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling(force_subjbias_zeromean=False)

        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 280.31447815213642, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 1.4355485462027884, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_bias'])), -0.090840910829074084, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_bias'])), 0.089032585621095048, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency'])), 15.681766163430936, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_inconsistency'])), 0.01256558483297778, places=4)

    def test_least_squares_model(self):
        subjective_model = LeastSquaresModel.from_dataset_file(
            self.dataset_filepath)
        result = subjective_model.run_modeling()

        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 280.03846153847428, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 1.4012220200638821, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_bias'])), 0, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_bias'])), 0.089032585621522581, places=4)

    def test_subjrejmos_subjective_model_corruptdata_subjreject(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = SubjrejMosModel(dataset_reader)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling()
        scores = result['quality_scores']

        self.assertAlmostEqual(float(np.mean(scores)), 3.5611814345991566, places=4)
        self.assertAlmostEqual(float(np.var(scores)), 1.1049505732699529, places=4) # 1.4012220200639218

        self.assertAlmostEqual(result['dof'], 0.07692307692307693, places=6)
        self.assertAlmostEqual(result['loglikelihood'], -1.2051956998810835, places=6)
        self.assertAlmostEqual(float(np.std(result['raw_scores'])), 1.3565171169581582, places=6)
        self.assertAlmostEqual(float(np.std(result['reconstructions'])), 1.0511662919205282, places=6)
        self.assertAlmostEqual(result['aic'], 2.564237553608321, places=6)
        self.assertAlmostEqual(result['bic'], 2.9971255836458983, places=6)

    def test_subjrejmos_pearson_subjective_model_corruptdata_subjreject(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = SubjrejMosModelPearson(dataset_reader)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling()
        scores = result['quality_scores']

        self.assertAlmostEqual(float(np.mean(scores)), 3.5322483423749245, places=4)
        self.assertAlmostEqual(float(np.var(scores)), 1.403470793861528, places=4) # 1.4012220200639218

        self.assertAlmostEqual(result['dof'], 0.07692307692307693, places=6)
        self.assertAlmostEqual(result['loglikelihood'], -0.9355458920134561, places=6)
        self.assertAlmostEqual(float(np.std(result['raw_scores'])), 1.3636600914070514, places=6)
        self.assertAlmostEqual(float(np.std(result['reconstructions'])), 1.1846817268201313, places=6)
        self.assertAlmostEqual(result['aic'], 2.02493793787306, places=6)
        self.assertAlmostEqual(result['bic'], 2.457825967910643, places=6)

    def test_subjrejmos_spearman_subjective_model_corruptdata_subjreject(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = SubjrejMosModelSpearman(dataset_reader)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling()
        scores = result['quality_scores']

        self.assertAlmostEqual(float(np.mean(scores)), 3.5322483423749245, places=4)
        self.assertAlmostEqual(float(np.var(scores)), 1.403470793861528, places=4) # 1.4012220200639218

        self.assertAlmostEqual(result['dof'], 0.07692307692307693, places=6)
        self.assertAlmostEqual(result['loglikelihood'], -0.9355458920134561, places=6)
        self.assertAlmostEqual(float(np.std(result['raw_scores'])), 1.3636600914070514, places=6)
        self.assertAlmostEqual(float(np.std(result['reconstructions'])), 1.1846817268201313, places=6)
        self.assertAlmostEqual(result['aic'], 2.02493793787306, places=6)
        self.assertAlmostEqual(result['bic'], 2.457825967910643, places=6)

    def test_zscoremos_subjective_model_corruptdata_subjreject(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = ZscoringMosModel(dataset_reader)
        result = subjective_model.run_modeling()
        scores = result['quality_scores']

        self.assertAlmostEqual(float(np.mean(scores)), 0.0, places=4)
        self.assertAlmostEqual(float(np.var(scores)), 0.5405866214633748, places=4) # 1.4012220200639218

        self.assertAlmostEqual(result['dof'], 0.07692307692307693, places=6)
        self.assertAlmostEqual(result['loglikelihood'], -0.9696021743118809, places=6)
        self.assertAlmostEqual(float(np.std(result['raw_scores'])), 0.99365072945774, places=6)
        self.assertAlmostEqual(float(np.std(result['reconstructions'])), 0.7352459598415858, places=6)
        self.assertAlmostEqual(result['aic'], 2.0930505024699158, places=6)
        self.assertAlmostEqual(result['bic'], 2.525938532507493, places=6)

    def test_biasremv_mos_subjective_model_corruptdata(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = BiasremvMosModel(dataset_reader)
        result = subjective_model.run_modeling()
        scores = result['quality_scores']
        bias = result['observer_bias']
        self.assertTrue('observers' not in result)

        self.assertAlmostEqual(float(np.mean(scores)), 3.5447906523855885, places=8)
        self.assertAlmostEqual(float(np.var(scores)), 0.9589330529453537, places=8)
        self.assertAlmostEqual(float(np.mean(bias)), 0.0, places=8)
        self.assertAlmostEqual(float(np.var(bias)), 0.08903258562151982, places=8)

        self.assertAlmostEqual(result['dof'], 0.08958130477117819, places=6)
        self.assertAlmostEqual(result['loglikelihood'], -1.2761533126002955, places=6)
        self.assertAlmostEqual(float(np.std(result['raw_scores'])), 1.332411174171261, places=6)
        self.assertAlmostEqual(float(np.std(result['reconstructions'])), 0.9792512716077287, places=6)
        self.assertAlmostEqual(result['aic'], 2.7314692347429474, places=6)
        self.assertAlmostEqual(result['bic'], 3.2355920039006323, places=6)

    def test_biasremv_subjrej_mos_subjective_model_corruptdata(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = BiasremvSubjrejMosModel(dataset_reader)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling()
        scores = result['quality_scores']
        bias = result['observer_bias']

        self.assertAlmostEqual(float(np.mean(scores)), 3.5447906523855885, places=8)
        self.assertAlmostEqual(float(np.var(scores)), 1.09500013352561, places=8)
        self.assertAlmostEqual(float(np.mean(bias)), 0.0, places=8)
        self.assertAlmostEqual(float(np.var(bias)), 0.08903258562151982, places=8)

        self.assertAlmostEqual(result['dof'], 0.08958130477117819, places=6)
        self.assertAlmostEqual(result['loglikelihood'], -1.1737836830835549, places=6)
        self.assertAlmostEqual(float(np.std(result['raw_scores'])), 1.3307052960550632, places=6)
        self.assertAlmostEqual(float(np.std(result['reconstructions'])), 1.04642254062382, places=6)
        self.assertAlmostEqual(result['aic'], 2.526729975709466, places=6)
        self.assertAlmostEqual(result['bic'], 3.030852744867151, places=6)

    def test_biasremv_subjrej_pearson_mos_subjective_model_corruptdata(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = BiasremvSubjrejMosModelPearson(dataset_reader)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling()
        scores = result['quality_scores']
        bias = result['observer_bias']

        self.assertAlmostEqual(float(np.mean(scores)), 3.5447906523855885, places=8)
        self.assertAlmostEqual(float(np.var(scores)), 1.403470793861527, places=8)
        self.assertAlmostEqual(float(np.mean(bias)), 0.0, places=8)
        self.assertAlmostEqual(float(np.var(bias)), 0.08903258562151982, places=8)

        self.assertAlmostEqual(result['dof'], 0.08958130477117819, places=6)
        self.assertAlmostEqual(result['loglikelihood'], -0.8483987293047435, places=6)
        self.assertAlmostEqual(float(np.std(result['raw_scores'])), 1.3269967925381838, places=6)
        self.assertAlmostEqual(float(np.std(result['reconstructions'])), 1.18468172682013088, places=6)
        self.assertAlmostEqual(result['aic'], 1.8759600681518434, places=6)
        self.assertAlmostEqual(result['bic'], 2.380082837309528, places=6)

    def test_biasremv_subjrej_spearman_mos_subjective_model_corruptdata(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = BiasremvSubjrejMosModelSpearman(dataset_reader)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling()
        scores = result['quality_scores']
        bias = result['observer_bias']

        self.assertAlmostEqual(float(np.mean(scores)), 3.5447906523855885, places=8)
        self.assertAlmostEqual(float(np.var(scores)), 1.403470793861527, places=8)
        self.assertAlmostEqual(float(np.mean(bias)), 0.0, places=8)
        self.assertAlmostEqual(float(np.var(bias)), 0.08903258562151982, places=8)

        self.assertAlmostEqual(result['dof'], 0.08958130477117819, places=6)
        self.assertAlmostEqual(result['loglikelihood'], -0.8483987293047435, places=6)
        self.assertAlmostEqual(float(np.std(result['raw_scores'])), 1.3269967925381838, places=6)
        self.assertAlmostEqual(float(np.std(result['reconstructions'])), 1.18468172682013088, places=6)
        self.assertAlmostEqual(result['aic'], 1.8759600681518434, places=6)
        self.assertAlmostEqual(result['bic'], 2.380082837309528, places=6)

    def test_zscoresubjrejmos_subjective_model_corruptdata_subjreject(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = ZscoringSubjrejMosModel(dataset_reader)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling()
        scores = result['quality_scores']

        self.assertAlmostEqual(float(np.mean(scores)), 0, places=4)
        self.assertAlmostEqual(float(np.var(scores)), 0.66670826882879042, places=4) # 1.4012220200639218

    def test_subjrejdmos_subjective_model_corruptdata_subjreject(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = SubjrejDmosModel(dataset_reader)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling()
        scores = result['quality_scores']

        self.assertAlmostEqual(float(np.mean(scores)), 4.0246673158065542, places=4)
        self.assertAlmostEqual(float(np.var(scores)), 1.0932580358187849, places=4) # 1.4012220200639218

    def test_zscoresubjrejdmos_subjective_model_corruptdata_subjreject(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = ZscoringSubjrejDmosModel(dataset_reader)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling()
        scores = result['quality_scores']

        self.assertAlmostEqual(float(np.mean(scores)), 0, places=4)
        self.assertAlmostEqual(float(np.var(scores)), 0.66405245792414114, places=4) # 1.4012220200639218

    def test_persubject_subjective_model_output(self):
        dataset = import_python_file(self.dataset_filepath)
        dataset_reader = RawDatasetReader(dataset)
        subjective_model = PerSubjectModel(dataset_reader)
        subjective_model.run_modeling(transform_final={'p1':25, 'p0':-25})
        subjective_model.to_aggregated_dataset_file(self.output_dataset_filepath)
        self.assertTrue(os.path.exists(self.output_dataset_filepath))
        dataset2 = import_python_file(self.output_dataset_filepath)
        dis_video = dataset2.dis_videos[0]
        self.assertTrue('groundtruth' in dis_video)
        self.assertTrue('os' not in dis_video)
        self.assertAlmostEqual(dis_video['groundtruth'], 100.0, places=4)

    def test_proj_mle_subjective_model_corruptdata(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = SubjectMLEModelProjectionSolver(dataset_reader)
        result = subjective_model.run_modeling()
        scores = result['quality_scores']
        bias = result['observer_bias']
        inconsistency = result['observer_inconsistency']
        self.assertTrue('observers' not in result)

        self.assertAlmostEqual(float(np.mean(scores)), 3.5447906523855877, places=8)
        self.assertAlmostEqual(float(np.var(scores)), 1.3559834679453553, places=8)
        self.assertAlmostEqual(float(np.mean(bias)), 0.0, places=8)
        self.assertAlmostEqual(float(np.var(bias)), 0.08903258562151985, places=8)
        self.assertAlmostEqual(float(np.mean(inconsistency)), 0.8091663380211014, places=8)
        self.assertAlmostEqual(float(np.var(inconsistency)), 0.21269010120806528, places=8)

        self.assertAlmostEqual(float(np.sum(result['observer_bias_std'])), 2.3669964674034123, places=4)
        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency_std'])), 1.6737192530463552, places=4)
        self.assertAlmostEqual(float(np.sum(result['quality_scores_std'])), 9.592833401286343, places=4)

    def test_proj_mle_subjective_model_corruptdata_nonzero_bias(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = SubjectMLEModelProjectionSolver(dataset_reader)
        result = subjective_model.run_modeling(force_subjbias_zeromean=False)
        scores = result['quality_scores']
        bias = result['observer_bias']
        inconsistency = result['observer_inconsistency']

        self.assertAlmostEqual(float(np.mean(scores)), 3.5447906523855877, places=8)
        self.assertAlmostEqual(float(np.var(scores)), 1.3559834679453553, places=8)
        self.assertAlmostEqual(float(np.mean(bias)), 0.0, places=8)
        self.assertAlmostEqual(float(np.var(bias)), 0.08903258562151985, places=8)
        self.assertAlmostEqual(float(np.mean(inconsistency)), 0.8091663380211014, places=8)
        self.assertAlmostEqual(float(np.var(inconsistency)), 0.21269010120806528, places=8)

        self.assertAlmostEqual(result['dof'], 0.06377799415774099, places=6)
        self.assertAlmostEqual(result['loglikelihood'], -1.084797535188502, places=6)
        self.assertAlmostEqual(float(np.std(result['raw_scores'])), 1.3654128030298962, places=6)
        self.assertAlmostEqual(float(np.std(result['reconstructions'])), 1.2020882040879586, places=6)
        self.assertAlmostEqual(result['aic'], 2.297151058692486, places=6)
        self.assertAlmostEqual(result['bic'], 2.6560645519514896, places=6)

        self.assertAlmostEqual(float(np.sum(result['observer_bias_std'])), 2.3669964674034123, places=4)
        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency_std'])), 1.6737192530463552, places=4)
        self.assertAlmostEqual(float(np.sum(result['quality_scores_std'])), 9.592833401286343, places=4)

    def test_mleco_subjective_model_corruptdata_nonzero_bias(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = MaximumLikelihoodEstimationModelContentOblivious(dataset_reader)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling(force_subjbias_zeromean=False)
        scores = result['quality_scores']
        bias = result['observer_bias']
        inconsistency = result['observer_inconsistency']

        self.assertAlmostEqual(float(np.mean(scores)), 3.5580494278512447, places=8)
        self.assertAlmostEqual(float(np.var(scores)), 1.3559834445021643, places=8)
        self.assertAlmostEqual(float(np.mean(bias)), -0.013258775465654477, places=8)
        self.assertAlmostEqual(float(np.var(bias)), 0.08903258562151789, places=8)
        self.assertAlmostEqual(float(np.mean(inconsistency)), 0.8091663380211014, places=8)
        self.assertAlmostEqual(float(np.var(inconsistency)), 0.2126900961328451, places=8)

        self.assertAlmostEqual(result['dof'], 0.06377799415774099, places=6)
        self.assertAlmostEqual(result['loglikelihood'], -1.0847975351885024, places=6)
        self.assertAlmostEqual(float(np.std(result['raw_scores'])), 1.3654128030298962, places=6)
        self.assertAlmostEqual(float(np.std(result['reconstructions'])), 1.2020881956510854, places=6)
        self.assertAlmostEqual(result['aic'], 2.297151058692487, places=6)
        self.assertAlmostEqual(result['bic'], 2.6560645519514905, places=6)

    def test_proj_mle_subjective_model_corruptdata_missingdata(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        dataset_reader1 = CorruptSubjectRawDatasetReader(dataset, input_dict={'selected_subjects': range(5)})
        dataset1 = dataset_reader1.to_dataset()
        dataset_reader2 = MissingDataRawDatasetReader(dataset1, input_dict={'missing_probability': 0.0001})
        subjective_model = SubjectMLEModelProjectionSolver(dataset_reader2)
        result = subjective_model.run_modeling(force_subjbias_zeromean=False)
        scores = result['quality_scores']
        bias = result['observer_bias']
        inconsistency = result['observer_inconsistency']

        self.assertAlmostEqual(float(np.mean(scores)), 3.5441674307897983, places=8)
        self.assertAlmostEqual(float(np.var(scores)), 1.3557530628643795, places=8)
        self.assertAlmostEqual(float(np.mean(bias)), 0.00011539474984923769, places=8)
        self.assertAlmostEqual(float(np.var(bias)), 0.08879525615906458, places=8)
        self.assertAlmostEqual(float(np.mean(inconsistency)), 0.8088220663739162, places=8)
        self.assertAlmostEqual(float(np.var(inconsistency)), 0.21296014750848657, places=8)

        self.assertAlmostEqual(float(np.sum(result['observer_bias_std'])), 2.3663572924335963, places=4)
        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency_std'])), 1.673273950838568, places=4)
        self.assertAlmostEqual(float(np.sum(result['quality_scores_std'])), 9.589031768667335, places=4)

    def test_proj_mle_subjective_model2_corruptdata_nonzero_bias(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }
        dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)
        subjective_model = SubjectMLEModelProjectionSolver2(dataset_reader)
        result = subjective_model.run_modeling(force_subjbias_zeromean=False)
        scores = result['quality_scores']
        bias = result['observer_bias']
        inconsistency = result['observer_inconsistency']

        self.assertAlmostEqual(float(np.mean(scores)), 3.5447906523855877, places=8)
        self.assertAlmostEqual(float(np.var(scores)), 1.3559834679453553, places=8)
        self.assertAlmostEqual(float(np.mean(bias)), 0.0, places=8)
        self.assertAlmostEqual(float(np.var(bias)), 0.08903258562151985, places=8)
        self.assertAlmostEqual(float(np.mean(inconsistency)), 0.8091663380211014, places=8)
        self.assertAlmostEqual(float(np.var(inconsistency)), 0.21269010120806528, places=8)

        self.assertAlmostEqual(result['dof'], 0.06377799415774099, places=6)
        self.assertAlmostEqual(result['loglikelihood'], -1.084797535188502, places=6)
        self.assertAlmostEqual(float(np.std(result['raw_scores'])), 1.3654128030298962, places=6)
        self.assertAlmostEqual(float(np.std(result['reconstructions'])), 1.2020882040879586, places=6)
        self.assertAlmostEqual(result['aic'], 2.297151058692486, places=6)
        self.assertAlmostEqual(result['bic'], 2.6560645519514896, places=6)

        self.assertAlmostEqual(float(np.sum(result['observer_bias_std'])), 2.3669964674034123, places=4)
        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency_std'])), 1.6737192530463552, places=4)
        self.assertAlmostEqual(float(np.sum(result['quality_scores_std'])), 13.711486766043402, places=4)

    def test_proj_mle_subjective_model2_corruptdata_missingdata(self):
        dataset = import_python_file(self.dataset_filepath)
        np.random.seed(0)
        dataset_reader1 = CorruptSubjectRawDatasetReader(dataset, input_dict={'selected_subjects': range(5)})
        dataset1 = dataset_reader1.to_dataset()
        dataset_reader2 = MissingDataRawDatasetReader(dataset1, input_dict={'missing_probability': 0.0001})
        subjective_model = SubjectMLEModelProjectionSolver2(dataset_reader2)
        result = subjective_model.run_modeling(force_subjbias_zeromean=False)
        scores = result['quality_scores']
        bias = result['observer_bias']
        inconsistency = result['observer_inconsistency']

        self.assertAlmostEqual(float(np.mean(scores)), 3.5441674307897983, places=8)
        self.assertAlmostEqual(float(np.var(scores)), 1.3557530628643795, places=8)
        self.assertAlmostEqual(float(np.mean(bias)), 0.00011539474984923769, places=8)
        self.assertAlmostEqual(float(np.var(bias)), 0.08879525615906458, places=8)
        self.assertAlmostEqual(float(np.mean(inconsistency)), 0.8088220663739162, places=8)
        self.assertAlmostEqual(float(np.var(inconsistency)), 0.21296014750848657, places=8)

        self.assertAlmostEqual(float(np.sum(result['observer_bias_std'])), 2.3663572924335963, places=4)
        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency_std'])), 1.673273950838568, places=4)
        self.assertAlmostEqual(float(np.sum(result['quality_scores_std'])), 13.712083371807026, places=4)


class SubjectiveModelPartialTest(MyTestCase):

    def setUp(self):
        super().setUp()
        self.dataset_filepath = SurealConfig.test_resource_path('NFLX_dataset_public_raw_PARTIAL.py')
        self.output_dataset_filepath = SurealConfig.workdir_path('NFLX_dataset_public_test_PARTIAL.py')
        self.output_dataset_pyc_filepath = SurealConfig.workdir_path('NFLX_dataset_public_test_PARTIAL.pyc')

    def tearDown(self):
        if os.path.exists(self.output_dataset_filepath):
            os.remove(self.output_dataset_filepath)
        if os.path.exists(self.output_dataset_pyc_filepath):
            os.remove(self.output_dataset_pyc_filepath)
        super().tearDown()

    def test_mos_subjective_model(self):
        dataset = import_python_file(self.dataset_filepath)
        dataset_reader = RawDatasetReader(dataset)
        subjective_model = MosModel(dataset_reader)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling()
        scores = result['quality_scores']
        self.assertAlmostEqual(scores[0], 4.884615384615385, places=4)
        self.assertAlmostEqual(scores[10], 2.8076923076923075, places=4)
        self.assertAlmostEqual(float(np.mean(scores)), 3.4871794871794877, places=4)
        scores_std = result['quality_scores_std']
        self.assertAlmostEqual(float(np.mean(scores_std)), 0.13125250408357622, places=4)

    def test_mos_subjective_model_output(self):
        dataset = import_python_file(self.dataset_filepath)
        dataset_reader = RawDatasetReader(dataset)
        subjective_model = MosModel(dataset_reader)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            subjective_model.run_modeling()
        subjective_model.to_aggregated_dataset_file(self.output_dataset_filepath)
        self.assertTrue(os.path.exists(self.output_dataset_filepath))
        dataset2 = import_python_file(self.output_dataset_filepath)
        dis_video = dataset2.dis_videos[0]
        self.assertTrue('groundtruth' in dis_video)
        self.assertTrue('groundtruth_std' in dis_video)
        self.assertTrue('os' not in dis_video)
        self.assertAlmostEqual(dis_video['groundtruth'], 4.884615384615385, places=4)
        self.assertAlmostEqual(dis_video['groundtruth_std'], 0.06389710663783135, places=4)

    def test_mos_subjective_model_normalize_final(self):
        dataset = import_python_file(self.dataset_filepath)
        dataset_reader = RawDatasetReader(dataset)
        subjective_model = MosModel(dataset_reader)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling(normalize_final=True)
        scores = result['quality_scores']
        self.assertAlmostEqual(scores[0], 1.1666952279897338, places=4)
        self.assertAlmostEqual(scores[10], -0.56729217507757768, places=4)
        self.assertAlmostEqual(float(np.mean(scores)), 0.0, places=4)

    def test_mos_subjective_model_transform_final(self):
        dataset = import_python_file(self.dataset_filepath)
        dataset_reader = RawDatasetReader(dataset)
        subjective_model = MosModel(dataset_reader)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling(transform_final={'p1': 10, 'p0': 1})
        scores = result['quality_scores']
        self.assertAlmostEqual(scores[0], 49.84615384615385, places=4)
        self.assertAlmostEqual(scores[10], 29.076923076923073, places=4)
        self.assertAlmostEqual(float(np.mean(scores)), 35.871794871794876, places=4)

    def test_from_dataset_file(self):
        subjective_model = MosModel.from_dataset_file(self.dataset_filepath)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling()
        scores = result['quality_scores']
        self.assertAlmostEqual(scores[0], 4.884615384615385, places=4)
        self.assertAlmostEqual(scores[10], 2.8076923076923075, places=4)
        self.assertAlmostEqual(float(np.mean(scores)), 3.4871794871794877, places=4)

    def test_dmos_subjective_model(self):
        subjective_model = DmosModel.from_dataset_file(self.dataset_filepath)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling()
        scores = result['quality_scores']
        self.assertAlmostEqual(scores[0], 5.0, places=4)
        self.assertAlmostEqual(scores[10], 2.9230769230769225, places=4)
        self.assertAlmostEqual(float(np.mean(scores)), 3.7473604826546003, places=4)
        scores_std = result['quality_scores_std']
        self.assertAlmostEqual(float(np.mean(scores_std)), 0.13125250408357622, places=4)

    def test_observer_aware_subjective_model_with_dscoring(self):
        subjective_model = LegacyMaximumLikelihoodEstimationModel.from_dataset_file(
            self.dataset_filepath)
        result = subjective_model.run_modeling(dscore_mode=True, force_subjbias_zeromean=False)

        self.assertAlmostEqual(float(np.sum(result['observer_bias'])), -0.038360699965619777, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_bias'])), 0.095605013092265739, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency'])), 15.81030572681315, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_inconsistency'])), 0.014607671806207905, places=4)

        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 191.1906306037788, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 1.4711930351190119, places=4)

    def test_observer_aware_subjective_model_use_log(self):
        subjective_model = LegacyMaximumLikelihoodEstimationModel.from_dataset_file(
            self.dataset_filepath)
        result = subjective_model.run_modeling(use_log=True, force_subjbias_zeromean=False)

        self.assertAlmostEqual(float(np.sum(result['observer_bias'])), -0.02907696993595069, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_bias'])), 0.095605013092265725, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency'])), 15.810305727732661, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_inconsistency'])), 0.014607671851733216, places=4)

        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 177.90318944102833, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 1.4830610455789057, places=4)

    def test_observer_aware_subjective_model_output_os_is_dict_style(self):
        dataset = import_python_file(SurealConfig.test_resource_path('test_dataset_os_as_dict.py'))
        dataset_reader = RawDatasetReader(dataset)
        subjective_model = LegacyMaximumLikelihoodEstimationModel(dataset_reader)
        result = subjective_model.run_modeling()
        self.assertTrue('observers' in result)
        subjective_model.to_aggregated_dataset_file(self.output_dataset_filepath)
        self.assertTrue(os.path.exists(self.output_dataset_filepath))
        dataset2 = import_python_file(self.output_dataset_filepath)
        dis_video = dataset2.dis_videos[0]
        print(dataset2.dis_videos)
        self.assertTrue('groundtruth' in dis_video)
        self.assertTrue('os' not in dis_video)
        self.assertAlmostEqual(dis_video['groundtruth'], 2.444444460976023, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_bias'])), 0.0, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_bias'])), 0.17283950617283952, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency'])), 2.8802122896041467, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_inconsistency'])), 0.4856715281793693, places=4)

        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 7.333333333333334, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 0.6666666530952383, places=4)

    def test_observer_content_aware_subjective_model(self):
        subjective_model = MaximumLikelihoodEstimationModel.from_dataset_file(
            self.dataset_filepath)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling(force_subjbias_zeromean=False)

        self.assertAlmostEqual(float(np.nansum(result['content_ambiguity'])), 2.653508643860357, places=4)
        self.assertAlmostEqual(float(np.nanvar(result['content_ambiguity'])), 0.0092892978862108271, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_bias'])), -0.020313188445860726, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_bias'])), 0.091830942654165318, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency'])), 11.232923468639161, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_inconsistency'])), 0.027721095664357907, places=4)

        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 177.88599894484821, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 1.4896077857605587, places=4)

        # self.assertAlmostEqual(np.nansum(result['content_ambiguity_std']), 0.30465244947706538, places=4)
        self.assertAlmostEqual(float(np.sum(result['observer_bias_std'])), 2.165903882505483, places=4)
        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency_std'])), 27.520643824238352, places=4)
        self.assertAlmostEqual(float(np.sum(result['quality_scores_std'])), 5.7355563435912256, places=4)

    def test_observer_content_aware_subjective_model_nocontent(self):
        subjective_model = MaximumLikelihoodEstimationModelContentOblivious.from_dataset_file(
            self.dataset_filepath)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling(force_subjbias_zeromean=False)

        self.assertAlmostEqual(float(np.sum(result['observer_bias'])), -0.038360699965624648, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_bias'])), 0.095605013092265753, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency'])), 15.81030572681315, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_inconsistency'])), 0.014607671806207895, places=4)

        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 177.92139983454805, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 1.4830610442685492, places=4)

    def test_ap_subjective_model_output_os_is_dict_style(self):
        dataset = import_python_file(SurealConfig.test_resource_path('test_dataset_os_as_dict.py'))
        dataset_reader = RawDatasetReader(dataset)
        subjective_model = SubjectMLEModelProjectionSolver(dataset_reader)
        result = subjective_model.run_modeling()
        self.assertTrue('observers' in result)
        subjective_model.to_aggregated_dataset_file(self.output_dataset_filepath)
        self.assertTrue(os.path.exists(self.output_dataset_filepath))
        dataset2 = import_python_file(self.output_dataset_filepath)
        dis_video = dataset2.dis_videos[0]
        print(dataset2.dis_videos)
        self.assertTrue('groundtruth' in dis_video)
        self.assertTrue('os' not in dis_video)
        self.assertAlmostEqual(dis_video['groundtruth'], 2.444444460976023, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_bias'])), 0.0, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_bias'])), 0.17283950617283952, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency'])), 2.8802122896041467, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_inconsistency'])), 0.4856715281793693, places=4)

        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 7.333333333333334, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 0.6666666530952383, places=4)


if __name__ == '__main__':
    unittest.main()
