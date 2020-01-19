import os
import unittest
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
        result = subjective_model.run_modeling(force_subjbias_zeromean=False, n_bootstrap=3)

        self.assertAlmostEqual(np.float(np.sum(result['observer_bias'])), -0.090840910829083799, places=4)
        self.assertAlmostEqual(np.float(np.var(result['observer_bias'])), 0.089032585621095089, places=4)

        self.assertAlmostEqual(np.float(np.sum(result['observer_inconsistency'])), 15.681766163430936, places=4)
        self.assertAlmostEqual(np.float(np.var(result['observer_inconsistency'])), 0.012565584832977776, places=4)

        self.assertAlmostEqual(np.float(np.sum(result['quality_scores'])), 280.31447815213642, places=4)
        self.assertAlmostEqual(np.float(np.var(result['quality_scores'])), 1.4355485462027884, places=4)

        self.assertAlmostEqual(np.float(np.sum(result['quality_scores_ci95'][0])), 13.826041058489736, places=6)
        self.assertAlmostEqual(np.float(np.sum(result['quality_scores_ci95'][1])), 18.765216680244393, places=6)

    def test_observer_content_aware_subjective_model_bootstrapping_nocontent_subjbias_zeromean(self):
        subjective_model = MaximumLikelihoodEstimationModelContentObliviousWithBootstrapping.from_dataset_file(
            self.dataset_filepath)
        result = subjective_model.run_modeling(n_bootstrap=3)

        self.assertAlmostEqual(np.float(np.sum(result['observer_bias'])), 0.0, places=4)
        self.assertAlmostEqual(np.float(np.var(result['observer_bias'])), 0.089032585621095089, places=4)

        self.assertAlmostEqual(np.float(np.sum(result['observer_inconsistency'])), 15.681766163430936, places=4)
        self.assertAlmostEqual(np.float(np.var(result['observer_inconsistency'])), 0.012565584832977776, places=4)

        self.assertAlmostEqual(np.float(np.sum(result['quality_scores'])), 280.0384615384633, places=4)
        self.assertAlmostEqual(np.float(np.var(result['quality_scores'])), 1.4355485462027884, places=4)

        self.assertAlmostEqual(np.float(np.sum(result['quality_scores_ci95'][0])), 15.282049602317477, places=6)
        self.assertAlmostEqual(np.float(np.sum(result['quality_scores_ci95'][1])), 17.487919756077957, places=6)

