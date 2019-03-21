import unittest

import numpy as np
import scipy.stats as st

from sureal.config import SurealConfig
from sureal.dataset_reader import RawDatasetReader, PairedCompDatasetReader
from sureal.pc_subjective_model import BradleyTerryNewtonRaphsonPairedCompSubjectiveModel
from sureal.tools.misc import import_python_file

__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"


class PcSubjectiveModelTest(unittest.TestCase):

    def setUp(self):
        dataset_filepath = SurealConfig.test_resource_path('NFLX_dataset_public_raw.py')
        dataset = import_python_file(dataset_filepath)
        dataset_reader = RawDatasetReader(dataset)
        pc_dataset = dataset_reader.to_pc_dataset()
        self.pc_dataset_reader = PairedCompDatasetReader(pc_dataset)

    def test_btnr_subjective_model(self):
        subjective_model = BradleyTerryNewtonRaphsonPairedCompSubjectiveModel(self.pc_dataset_reader)
        result = subjective_model.run_modeling(zscore_output=True)
        self.assertAlmostEquals(np.sum(result['quality_scores']), 0, places=4)
        self.assertAlmostEquals(np.var(result['quality_scores']), 1, places=4)
        self.assertAlmostEqual(st.kurtosis(result['quality_scores']), -0.8411279645566632, places=4)
        self.assertAlmostEquals(np.sum(result['quality_scores_std']), 2.3474757263050097, places=4)
        self.assertAlmostEquals(np.var(result['quality_scores_std']), 0.0006165796552712534, places=8)
        self.assertAlmostEqual(st.kurtosis(result['quality_scores_std']), 4.8718876179528365, places=4)
