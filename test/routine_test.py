__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import unittest

import numpy as np

from sureal.config import SurealConfig
from sureal.subjective_model import MosModel
from sureal.tools.misc import import_python_file
from sureal.dataset_reader import RawDatasetReader, PairedCompDatasetReader
from sureal.routine import remove_observers


class SubjectRemovalTest(unittest.TestCase):

    def setUp(self):
        dataset1_filepath = SurealConfig.test_resource_path('NFLX_dataset_public_raw.py')
        dataset1 = import_python_file(dataset1_filepath)
        self.dataset1_reader = RawDatasetReader(dataset1)
        dataset2_filepath = SurealConfig.test_resource_path('lukas_pc_dataset.py')
        dataset2 = import_python_file(dataset2_filepath)
        self.dataset2_reader = PairedCompDatasetReader(dataset2)
        dataset3_filepath = SurealConfig.test_resource_path('test_dataset_os_as_dict.py')
        dataset3 = import_python_file(dataset3_filepath)
        self.dataset3_reader = RawDatasetReader(dataset3)
        self.subjective_model_classes = [MosModel]

    def test_not_list(self):
        with self.assertRaises(AssertionError):
            remove_observers(self.dataset2_reader, {'observer': 'vote'})

    def test_subject_removal_on_pc_dataset(self):
        with self.assertRaises(AssertionError):
            remove_observers(self.dataset2_reader, [0, 1])

    def test_remove_str_os_as_dict(self):
        dataset = remove_observers(self.dataset3_reader, ['Pinokio', 'Tom'])
        self.assertEqual(dataset.opinion_score_2darray[0], 4.0)
        self.assertEqual(dataset.opinion_score_2darray[1], 1.0)
        self.assertEqual(dataset.opinion_score_2darray[2], 1.0)

    def test_remove_int_os_as_dict(self):
        dataset = remove_observers(self.dataset3_reader, [0, 2])
        self.assertEqual(dataset.opinion_score_2darray[0], 4.0)
        self.assertEqual(dataset.opinion_score_2darray[1], 1.0)
        self.assertEqual(dataset.opinion_score_2darray[2], 1.0)

    def test_remove_int_os_as_list(self):
        dataset = remove_observers(self.dataset1_reader, [1, 6])
        self.assertEqual(np.mean(dataset.opinion_score_2darray[0, :]), 5.0)
        self.assertEqual(len(dataset.opinion_score_2darray[0, :]), 24)


if __name__ == '__main__':
    unittest.main()
