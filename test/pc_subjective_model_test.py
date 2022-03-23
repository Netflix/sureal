import unittest

import numpy as np
import scipy.stats as st

from sureal.config import SurealConfig
from sureal.dataset_reader import RawDatasetReader, PairedCompDatasetReader
from sureal.pc_subjective_model import BradleyTerryNewtonRaphsonPairedCompSubjectiveModel, \
    BradleyTerryMlePairedCompSubjectiveModel, ThurstoneMlePairedCompSubjectiveModel
from sureal.tools.misc import import_python_file

__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"


class PcSubjectiveModelTest(unittest.TestCase):

    def setUp(self):
        pc_dataset = import_python_file(SurealConfig.test_resource_path('lukas_pc_dataset.py'))
        self.pc_dataset_reader = PairedCompDatasetReader(pc_dataset)

    def test_btnr_subjective_model(self):
        subjective_model = BradleyTerryNewtonRaphsonPairedCompSubjectiveModel(self.pc_dataset_reader)
        result = subjective_model.run_modeling(zscore_output=True)
        self.assertTrue('observers' in result)
        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 0, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 1, places=4)
        self.assertAlmostEqual(st.kurtosis(result['quality_scores']), -0.05721221160408296, places=4)
        self.assertTrue(result['quality_scores_std'] is None)

    def test_btmle_subjective_model(self):
        subjective_model = BradleyTerryMlePairedCompSubjectiveModel(self.pc_dataset_reader)
        result = subjective_model.run_modeling()
        self.assertTrue('observers' in result)
        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), -187.18634399309573, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 3.1442888768417054, places=4)
        self.assertAlmostEqual(st.kurtosis(result['quality_scores']), 0.5649254682803901, places=4)
        self.assertAlmostEqual(float(np.sum(result['quality_scores_std'])), 11.136592174843651, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores_std'])), 0.003890667402965306, places=8)
        self.assertAlmostEqual(st.kurtosis(result['quality_scores_std']), 1.960577186185537, places=4)
        self.assertAlmostEqual(st.kurtosis(result['quality_scores_p']), 9.249782166616258, places=4)
        self.assertAlmostEqual(float(np.sum(result['quality_scores_p_std'])), 0.25667807232011897, places=4)
        self.assertAlmostEqual(float(np.sum(result['quality_scores_p_cov'])), 6.488285445421619e-16, places=4)
        self.assertAlmostEqual(float(np.sum(result['quality_scores_v_cov'])), 2.4459463440137217, places=4)
        self.assertAlmostEqual(float(np.sum(np.sqrt(np.diag(result['quality_scores_v_cov'])))), float(np.sum(result['quality_scores_std'])), places=4)
        self.assertAlmostEqual(float(np.sum(result['quality_scores_v_cov'])), float(np.sum(result['quality_scores_p_cov'] / (np.expand_dims(result['quality_scores_p'], axis=1) * (np.expand_dims(result['quality_scores_p'], axis=1).T)))), places=4)

    def test_thrustone_mle_subjective_model(self):
        subjective_model = ThurstoneMlePairedCompSubjectiveModel(self.pc_dataset_reader)
        result = subjective_model.run_modeling(zscore_output=True)
        self.assertTrue('observers' in result)
        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 0.0, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 1.0, places=4)
        self.assertAlmostEqual(st.kurtosis(result['quality_scores']), -0.3411839039618667, places=4)
        self.assertAlmostEqual(result['quality_scores'][0], 0.3791409047569019, places=4)
        self.assertAlmostEqual(result['quality_scores'][-1], -0.41006265745757303, places=4)
        self.assertAlmostEqual(float(np.sum(result['quality_scores_std'])), 4.893685013444614, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores_std'])), 0.0001601422716472649, places=4)
        self.assertAlmostEqual(st.kurtosis(result['quality_scores_std']), -0.6892814746512612, places=4)
        self.assertAlmostEqual(result['quality_scores_std'][0], 0.12571225814158712, places=4)
        self.assertAlmostEqual(result['quality_scores_std'][-1], 0.11003033360622685, places=4)

    def test_thrustone_mle_subjective_model_unsimplified_lbda(self):
        subjective_model = ThurstoneMlePairedCompSubjectiveModel(self.pc_dataset_reader)
        result = subjective_model.run_modeling(zscore_output=True, use_simplified_lbda=False)
        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 0.0, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 1.0, places=4)
        self.assertAlmostEqual(st.kurtosis(result['quality_scores']), -0.3411839039618667, places=4)
        self.assertAlmostEqual(result['quality_scores'][0], 0.3791409047569019, places=4)
        self.assertAlmostEqual(result['quality_scores'][-1], -0.41006265745757303, places=4)
        self.assertAlmostEqual(float(np.sum(result['quality_scores_std'])), 4.893685013444614, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores_std'])), 0.0001601422716472649, places=4)
        self.assertAlmostEqual(st.kurtosis(result['quality_scores_std']), -0.6892814746512612, places=4)
        self.assertAlmostEqual(result['quality_scores_std'][0], 0.12571225814158712, places=4)
        self.assertAlmostEqual(result['quality_scores_std'][-1], 0.11003033360622685, places=4)


class PcSubjectiveModelTestSynthetic(unittest.TestCase):

    def setUp(self):
        dataset_filepath = SurealConfig.test_resource_path('NFLX_dataset_public_raw.py')
        dataset = import_python_file(dataset_filepath)
        dataset_reader = RawDatasetReader(dataset)
        pc_dataset = dataset_reader.to_pc_dataset(pc_type='within_subject')
        self.pc_dataset_reader = PairedCompDatasetReader(pc_dataset)

    def test_btnr_subjective_model(self):
        subjective_model = BradleyTerryNewtonRaphsonPairedCompSubjectiveModel(self.pc_dataset_reader)
        result = subjective_model.run_modeling(zscore_output=True)
        self.assertTrue('observers' in result)
        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 0, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 1, places=4)
        self.assertAlmostEqual(st.kurtosis(result['quality_scores']), -0.6783168176396557, places=4)
        self.assertTrue(result['quality_scores_std'] is None)

    def test_btmle_subjective_model(self):
        subjective_model = BradleyTerryMlePairedCompSubjectiveModel(self.pc_dataset_reader)
        result = subjective_model.run_modeling()
        self.assertTrue('observers' in result)
        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), -441.51458317430405, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 4.286932098917939, places=4)
        self.assertAlmostEqual(st.kurtosis(result['quality_scores']), -0.6783168176396557, places=4)
        self.assertAlmostEqual(float(np.sum(result['quality_scores_std'])), 5.278359062068779, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores_std'])), 0.00019082518290164445, places=8)
        self.assertAlmostEqual(st.kurtosis(result['quality_scores_std']), 3.579717582250833, places=4)
        self.assertAlmostEqual(st.kurtosis(result['quality_scores_p']), -0.8426035121511268, places=4)
