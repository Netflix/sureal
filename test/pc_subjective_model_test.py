import unittest

import numpy as np

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
        result = subjective_model.run_modeling()
        self.assertAlmostEquals(np.sum(result['quality_scores']), -6.943838331662571, places=4)
        self.assertAlmostEquals(np.var(result['quality_scores']), 5.2872992787167785, places=4)
        self.assertAlmostEquals(np.sum(result['quality_scores_std']), 5.397815760274957, places=4)
        self.assertAlmostEquals(np.var(result['quality_scores_std']), 0.003260041166587138, places=4)
