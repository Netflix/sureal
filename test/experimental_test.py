import os
import unittest
import warnings

import numpy as np

from sureal.config import SurealConfig
from sureal.experimental import MaximumLikelihoodEstimationModelContentObliviousWithBootstrapping


class SubjectiveModelTest(unittest.TestCase):

    def setUp(self):
        self.dataset_filepath = SurealConfig.test_resource_path('NFLX_dataset_public_raw.py')
        self.output_dataset_filepath = SurealConfig.workdir_path('NFLX_dataset_public_test.py')
        self.output_dataset_pyc_filepath = SurealConfig.workdir_path('NFLX_dataset_public_test.pyc')

    def tearDown(self):
        if os.path.exists(self.output_dataset_filepath):
            os.remove(self.output_dataset_filepath)
        if os.path.exists(self.output_dataset_pyc_filepath):
            os.remove(self.output_dataset_pyc_filepath)

    def test_observer_content_aware_subjective_model_bootstrapping_nocontent(self):
        subjective_model = MaximumLikelihoodEstimationModelContentObliviousWithBootstrapping.from_dataset_file(
            self.dataset_filepath)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling(force_subjbias_zeromean=False, n_bootstrap=3)

        self.assertAlmostEqual(float(np.sum(result['observer_bias'])), -0.090840910829083799, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_bias'])), 0.089032585621095089, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency'])), 15.681766163430936, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_inconsistency'])), 0.012565584832977776, places=4)

        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 280.31447815213642, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 1.4355485462027884, places=4)

        self.assertAlmostEqual(float(np.sum(result['quality_scores_ci95'][0])), 16.40231025161599, places=6)
        self.assertAlmostEqual(float(np.sum(result['quality_scores_ci95'][1])), 16.03875029097417, places=6)

        self.assertAlmostEqual(float(np.sum(result['observer_bias_ci95'][0])), 1.5802456315996944, places=6)
        self.assertAlmostEqual(float(np.sum(result['observer_bias_ci95'][1])), 1.7042998573080086, places=6)

        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency_ci95'][0])), 1.4129652283023595, places=6)
        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency_ci95'][1])), 0.9315979842046503, places=6)

    def test_observer_content_aware_subjective_model_bootstrapping_nocontent_subjbias_zeromean(self):
        subjective_model = MaximumLikelihoodEstimationModelContentObliviousWithBootstrapping.from_dataset_file(
            self.dataset_filepath)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = subjective_model.run_modeling(n_bootstrap=3)

        self.assertAlmostEqual(float(np.sum(result['observer_bias'])), 0.0, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_bias'])), 0.089032585621095089, places=4)

        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency'])), 15.681766163430936, places=4)
        self.assertAlmostEqual(float(np.var(result['observer_inconsistency'])), 0.012565584832977776, places=4)

        self.assertAlmostEqual(float(np.sum(result['quality_scores'])), 280.0384615384633, places=4)
        self.assertAlmostEqual(float(np.var(result['quality_scores'])), 1.4355485462027884, places=4)

        self.assertAlmostEqual(float(np.sum(result['quality_scores_ci95'][0])), 16.40231025161599, places=6)
        self.assertAlmostEqual(float(np.sum(result['quality_scores_ci95'][1])), 16.03875029097417, places=6)

        self.assertAlmostEqual(float(np.sum(result['observer_bias_ci95'][0])), 1.5802456315996944, places=6)
        self.assertAlmostEqual(float(np.sum(result['observer_bias_ci95'][1])), 1.7042998573080086, places=6)

        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency_ci95'][0])), 1.4129652283023595, places=6)
        self.assertAlmostEqual(float(np.sum(result['observer_inconsistency_ci95'][1])), 0.9315979842046503, places=6)

