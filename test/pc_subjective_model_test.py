import unittest

import numpy as np
import scipy.stats as st

from sureal.config import SurealConfig
from sureal.dataset_reader import RawDatasetReader, PairedCompDatasetReader
from sureal.pc_subjective_model import BradleyTerryNewtonRaphsonPairedCompSubjectiveModel, \
    BradleyTerryMlePairedCompSubjectiveModel
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
        self.assertAlmostEquals(float(np.sum(result['quality_scores'])), 0, places=4)
        self.assertAlmostEquals(float(np.var(result['quality_scores'])), 1, places=4)
        self.assertAlmostEqual(st.kurtosis(result['quality_scores']), -0.05721221160408296, places=4)
        self.assertTrue(result['quality_scores_std'] is None)

    def test_btmle_subjective_model(self):
        subjective_model = BradleyTerryMlePairedCompSubjectiveModel(self.pc_dataset_reader)
        result = subjective_model.run_modeling(zscore_output=True)
        self.assertAlmostEquals(float(np.sum(result['quality_scores'])), -200.18931468202894, places=4)
        self.assertAlmostEquals(float(np.var(result['quality_scores'])), 3.143945927451612, places=4)
        self.assertAlmostEqual(st.kurtosis(result['quality_scores']), -0.07475628102734166, places=4)
        self.assertAlmostEquals(float(np.sum(result['quality_scores_std'])), 11.027811626712305, places=4)
        self.assertAlmostEquals(float(np.var(result['quality_scores_std'])), 0.002834787482229151, places=8)
        self.assertAlmostEqual(st.kurtosis(result['quality_scores_std']), -0.5347528088201456, places=4)


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
        self.assertAlmostEquals(float(np.sum(result['quality_scores'])), 0, places=4)
        self.assertAlmostEquals(float(np.var(result['quality_scores'])), 1, places=4)
        self.assertAlmostEqual(st.kurtosis(result['quality_scores']), -0.6783168176396557, places=4)
        self.assertTrue(result['quality_scores_std'] is None)

    def test_btmle_subjective_model(self):
        subjective_model = BradleyTerryMlePairedCompSubjectiveModel(self.pc_dataset_reader)
        result = subjective_model.run_modeling(zscore_output=True)
        self.assertAlmostEquals(float(np.sum(result['quality_scores'])), -518.137899354101, places=4)
        self.assertAlmostEquals(float(np.var(result['quality_scores'])), 4.286932098917939, places=4)
        self.assertAlmostEqual(st.kurtosis(result['quality_scores']), -0.6783168176396557, places=4)
        self.assertAlmostEquals(float(np.sum(result['quality_scores_std'])), 5.2435100711014675, places=4)
        self.assertAlmostEquals(float(np.var(result['quality_scores_std'])), 0.00012353987792172564, places=8)
        self.assertAlmostEqual(st.kurtosis(result['quality_scores_std']), 1.5798669747439735, places=4)
