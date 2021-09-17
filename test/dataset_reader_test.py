__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import unittest
import six

import numpy as np

from sureal.config import SurealConfig
from sureal.tools.misc import import_python_file, indices
from sureal.dataset_reader import RawDatasetReader, SyntheticRawDatasetReader, \
    MissingDataRawDatasetReader, SelectSubjectRawDatasetReader, \
    CorruptSubjectRawDatasetReader, CorruptDataRawDatasetReader, PairedCompDatasetReader, SelectDisVideoRawDatasetReader


class RawDatasetReaderTest(unittest.TestCase):

    def setUp(self):
        dataset_filepath = SurealConfig.test_resource_path('NFLX_dataset_public_raw.py')
        self.dataset = import_python_file(dataset_filepath)
        self.dataset_reader = RawDatasetReader(self.dataset)

    def test_read_dataset_stats(self):
        self.assertEqual(self.dataset_reader.num_ref_videos, 9)
        self.assertEqual(self.dataset_reader.max_content_id_of_ref_videos, 8)
        self.assertEqual(self.dataset_reader.num_dis_videos, 79)
        self.assertEqual(self.dataset_reader.num_observers, 26)
        self.assertEqual(self.dataset_reader.max_repetitions, 1)

    def test_opinion_score_3darray(self):
        os_3darray = self.dataset_reader.opinion_score_3darray
        self.assertAlmostEqual(float(np.mean(os_3darray)), 3.544790652385589, places=4)
        self.assertAlmostEqual(float(np.mean(np.std(os_3darray, axis=1))), 0.64933186478291516, places=4)

    def test_dis_videos_content_ids(self):
        content_ids = self.dataset_reader.content_id_of_dis_videos
        self.assertAlmostEqual(float(np.mean(content_ids)), 3.8607594936708862, places=4)

    def test_disvideo_is_refvideo(self):
        l = self.dataset_reader.disvideo_is_refvideo
        self.assertTrue(all(l[0:9]))

    def test_ref_score(self):
        self.assertEqual(self.dataset_reader.ref_score, 5.0)

    def test_to_persubject_dataset_wrong_dim(self):
        with self.assertRaises(AssertionError):
            dataset = self.dataset_reader.to_persubject_dataset(np.zeros(3000))
            self.assertEqual(len(dataset.dis_videos), 2054)

    def test_to_persubject_dataset(self):
        dataset = self.dataset_reader.to_persubject_dataset(np.zeros([79, 26]))
        self.assertEqual(len(dataset.dis_videos), 2054)


class RawDatasetReaderTest2(unittest.TestCase):

    def setUp(self):
        dataset_filepath1 = SurealConfig.test_resource_path('test_dataset_os_as_dict.py')
        dataset_filepath2 = SurealConfig.test_resource_path('test_dataset_os_as_dict_with_repetitions.py')
        dataset_filepath3 = SurealConfig.test_resource_path('test_dataset_os_as_list_with_repetitions.py')
        self.dataset1 = import_python_file(dataset_filepath1)
        self.dataset2 = import_python_file(dataset_filepath2)
        self.dataset3 = import_python_file(dataset_filepath3)
        self.dataset_reader1 = RawDatasetReader(self.dataset1)
        self.dataset_reader2 = RawDatasetReader(self.dataset2)
        self.dataset_reader3 = RawDatasetReader(self.dataset3)

    def test_read_dataset_stats(self):
        self.assertEqual(self.dataset_reader1.num_ref_videos, 2)
        self.assertEqual(self.dataset_reader2.num_ref_videos, 2)
        self.assertEqual(self.dataset_reader3.num_ref_videos, 2)
        self.assertEqual(self.dataset_reader1.max_content_id_of_ref_videos, 1)
        self.assertEqual(self.dataset_reader2.max_content_id_of_ref_videos, 1)
        self.assertEqual(self.dataset_reader3.max_content_id_of_ref_videos, 1)
        self.assertEqual(self.dataset_reader1.num_dis_videos, 3)
        self.assertEqual(self.dataset_reader2.num_dis_videos, 3)
        self.assertEqual(self.dataset_reader3.num_dis_videos, 3)
        self.assertEqual(self.dataset_reader1.num_observers, 3)
        self.assertEqual(self.dataset_reader2.num_observers, 3)
        self.assertEqual(self.dataset_reader3.num_observers, 3)
        self.assertEqual(self.dataset_reader1.max_repetitions, 1)
        self.assertEqual(self.dataset_reader2.max_repetitions, 3)
        self.assertEqual(self.dataset_reader3.max_repetitions, 3)

    def test_opinion_score_3darray(self):
        os_3darray1 = self.dataset_reader1.opinion_score_3darray
        os_3darray2 = self.dataset_reader2.opinion_score_3darray
        os_3darray3 = self.dataset_reader3.opinion_score_3darray
        self.assertAlmostEqual(float(np.mean(os_3darray1)), 2.4444444444444446, places=4)
        self.assertAlmostEqual(float(np.mean(np.std(os_3darray1, axis=1))), 1.1036449462590066, places=4)
        self.assertAlmostEqual(float(np.nanmean(os_3darray2)), 2.3076923076923075, places=4)
        self.assertAlmostEqual(float(np.nanmean(np.nanstd(os_3darray2, axis=1))), 0.6351558064628366, places=4)
        self.assertAlmostEqual(float(np.nanmean(os_3darray3)), 2.3076923076923075, places=4)
        self.assertAlmostEqual(float(np.nanmean(np.nanstd(os_3darray3, axis=1))), 0.6351558064628366, places=4)


class RawDatasetReaderPartialTest(unittest.TestCase):

    def setUp(self):
        dataset_filepath = SurealConfig.test_resource_path('NFLX_dataset_public_raw_PARTIAL.py')
        self.dataset = import_python_file(dataset_filepath)
        self.dataset_reader = RawDatasetReader(self.dataset)

    def test_read_dataset_stats(self):
        self.assertEqual(self.dataset_reader.num_ref_videos, 7)
        self.assertEqual(self.dataset_reader.max_content_id_of_ref_videos, 8)
        self.assertEqual(self.dataset_reader.num_dis_videos, 51)
        self.assertEqual(self.dataset_reader.num_observers, 26)
        self.assertEqual(self.dataset_reader.max_repetitions, 1)

    def test_opinion_score_3darray(self):
        os_3darray = self.dataset_reader.opinion_score_3darray
        self.assertAlmostEqual(float(np.mean(os_3darray)), 3.4871794871794872, places=4)
        self.assertAlmostEqual(float(np.mean(np.std(os_3darray, axis=1))), 0.65626252041788125, places=4)

    def test_dis_videos_content_ids(self):
        content_ids = self.dataset_reader.content_id_of_dis_videos
        self.assertAlmostEqual(float(np.mean(content_ids)), 3.9215686274509802, places=4)

    def test_disvideo_is_refvideo(self):
        l = self.dataset_reader.disvideo_is_refvideo
        self.assertTrue(all(l[0:7]))

    def test_ref_score(self):
        self.assertEqual(self.dataset_reader.ref_score, 5.0)

    def test_to_persubject_dataset_wrong_dim(self):
        with self.assertRaises(AssertionError):
            dataset = self.dataset_reader.to_persubject_dataset(np.zeros(3000))
            self.assertEqual(len(dataset.dis_videos), 2054)

    def test_to_persubject_dataset(self):
        dataset = self.dataset_reader.to_persubject_dataset(np.zeros([79, 26]))
        self.assertEqual(len(dataset.dis_videos), 1326)


class SyntheticDatasetReaderTest(unittest.TestCase):

    def setUp(self):
        dataset_filepath = SurealConfig.test_resource_path('NFLX_dataset_public_raw.py')
        dataset = import_python_file(dataset_filepath)

        np.random.seed(0)
        info_dict = {
            'quality_scores': np.random.randint(1, 6, 79),
            'observer_bias': np.random.normal(0, 1, 26),
            'observer_inconsistency': np.abs(np.random.normal(0, 0.1, 26)),
            'content_bias': np.zeros(9),
            'content_ambiguity': np.zeros(9),
        }

        self.dataset_reader = SyntheticRawDatasetReader(dataset, input_dict=info_dict)

    def test_read_dataset_stats(self):
        self.assertEqual(self.dataset_reader.num_ref_videos, 9)
        self.assertEqual(self.dataset_reader.num_dis_videos, 79)
        self.assertEqual(self.dataset_reader.num_observers, 26)
        self.assertEqual(self.dataset_reader.max_repetitions, 1)

    def test_opinion_score_3darray(self):
        os_3darray = self.dataset_reader.opinion_score_3darray
        self.assertAlmostEqual(float(np.mean(os_3darray)), 3.1912209428772669, places=4)

    def test_dis_videos_content_ids(self):
        content_ids = self.dataset_reader.content_id_of_dis_videos
        self.assertAlmostEqual(float(np.mean(content_ids)), 3.8607594936708862, places=4)

    def test_disvideo_is_refvideo(self):
        l = self.dataset_reader.disvideo_is_refvideo
        self.assertTrue(all(l[0:9]))

    def test_ref_score(self):
        self.assertEqual(self.dataset_reader.ref_score, 5.0)

    def test_to_dataset(self):
        dataset = self.dataset_reader.to_dataset()

        old_scores = [dis_video['os'] for dis_video in self.dataset_reader.dataset.dis_videos]
        new_scores = [dis_video['os'] for dis_video in dataset.dis_videos]

        self.assertNotEqual(old_scores, new_scores)


class MissingDatasetReaderTest(unittest.TestCase):

    def setUp(self):
        dataset_filepath = SurealConfig.test_resource_path('NFLX_dataset_public_raw.py')
        dataset = import_python_file(dataset_filepath)

        np.random.seed(0)
        info_dict = {
            'missing_probability': 0.1,
        }

        self.dataset_reader = MissingDataRawDatasetReader(dataset, input_dict=info_dict)

    def test_opinion_score_3darray(self):
        os_3darray = self.dataset_reader.opinion_score_3darray
        self.assertTrue(np.isnan(np.mean(os_3darray)))
        self.assertEqual(np.isnan(os_3darray).sum(), 201)

    def test_to_dataset(self):
        dataset = self.dataset_reader.to_dataset()

        old_scores = [dis_video['os'] for dis_video in self.dataset_reader.dataset.dis_videos]
        new_scores = [dis_video['os'] for dis_video in dataset.dis_videos]

        self.assertNotEqual(old_scores, new_scores)


class SelectedSubjectDatasetReaderTest(unittest.TestCase):

    def setUp(self):
        dataset_filepath = SurealConfig.test_resource_path('NFLX_dataset_public_raw.py')
        dataset = import_python_file(dataset_filepath)

        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }

        self.dataset_reader = SelectSubjectRawDatasetReader(dataset, input_dict=info_dict)

    def test_read_dataset_stats(self):
        self.assertEqual(self.dataset_reader.num_ref_videos, 9)
        self.assertEqual(self.dataset_reader.num_dis_videos, 79)
        self.assertEqual(self.dataset_reader.num_observers, 5)
        self.assertEqual(self.dataset_reader.max_repetitions, 1)

    def test_opinion_score_3darray(self):
        os_3darray = self.dataset_reader.opinion_score_3darray
        self.assertEqual(os_3darray.shape, (79, 5, 1))
        self.assertEqual(os_3darray[0, 0], 5.0)
        self.assertEqual(os_3darray[1, 0], 4.0)
        self.assertEqual(os_3darray[2, 0], 5.0)
        self.assertEqual(os_3darray[0, 1], 4.0)
        self.assertEqual(os_3darray[1, 1], 5.0)
        self.assertEqual(os_3darray[2, 1], 5.0)

    def test_to_dataset(self):
        dataset = self.dataset_reader.to_dataset()

        old_scores = [dis_video['os'] for dis_video in self.dataset_reader.dataset.dis_videos]
        new_scores = [dis_video['os'] for dis_video in dataset.dis_videos]

        self.assertNotEqual(old_scores, new_scores)


class SelectedSubjectDatasetReaderTest2(unittest.TestCase):

    def setUp(self):
        dataset2_filepath = SurealConfig.test_resource_path('test_dataset_os_as_dict.py')
        dataset2 = import_python_file(dataset2_filepath)

        np.random.seed(0)
        info_dict2 = {
            'selected_subjects': np.array([1, 2]),
        }

        self.dataset2_reader = SelectSubjectRawDatasetReader(dataset2, input_dict=info_dict2)

    def test_read_dataset_stats_os_as_dict(self):
        self.assertEqual(self.dataset2_reader.num_ref_videos, 2)
        self.assertEqual(self.dataset2_reader.num_dis_videos, 3)
        self.assertEqual(self.dataset2_reader.num_observers, 2)
        self.assertEqual(self.dataset2_reader.max_repetitions, 1)

    def test_opinion_score_3darray_os_as_dict(self):
        opinion_score_3darray = self.dataset2_reader.opinion_score_3darray
        self.assertEqual(opinion_score_3darray[0, 0], 1.0)
        self.assertEqual(opinion_score_3darray[1, 0], 3.0)
        self.assertEqual(opinion_score_3darray[2, 0], 3.0)
        self.assertEqual(opinion_score_3darray[0, 1], 3.0)
        self.assertEqual(opinion_score_3darray[1, 1], 2.0)
        self.assertEqual(opinion_score_3darray[2, 1], 4.0)

    def test_to_dataset_os_as_dict(self):
        dataset2 = self.dataset2_reader.to_dataset()

        old_scores = [dis_video['os'] for dis_video in self.dataset2_reader.dataset.dis_videos]
        new_scores = [dis_video['os'] for dis_video in dataset2.dis_videos]

        self.assertNotEqual(old_scores, new_scores)


class SelectedSubjectDatasetReaderTest3(unittest.TestCase):

    def setUp(self):
        dataset2_filepath = SurealConfig.test_resource_path('quality_variation_2017_agh_tv_dataset.py')
        dataset2 = import_python_file(dataset2_filepath)

        np.random.seed(0)
        info_dict2 = {
            'selected_subjects': list(range(13)),
        }

        self.dataset2_reader = SelectSubjectRawDatasetReader(dataset2, input_dict=info_dict2)

    def test_read_dataset_stats_os_as_dict(self):
        self.assertEqual(self.dataset2_reader.num_observers, 13)
        self.assertEqual(self.dataset2_reader.num_ref_videos, 20)
        self.assertEqual(self.dataset2_reader.num_dis_videos, 320)
        self.assertEqual(self.dataset2_reader.max_repetitions, 1)

    def test_opinion_score_3darray_os_as_dict(self):
        opinion_score_3darray = self.dataset2_reader.opinion_score_3darray
        self.assertEqual(opinion_score_3darray[0, 0], 2.0)
        self.assertEqual(opinion_score_3darray[1, 0], 1.0)
        self.assertTrue(np.isnan(opinion_score_3darray[2, 0]))
        self.assertTrue(np.isnan(opinion_score_3darray[0, 1]))
        self.assertTrue(np.isnan(opinion_score_3darray[1, 1]))
        self.assertEqual(opinion_score_3darray[2, 1], 1.0)

    def test_to_dataset_os_as_dict(self):
        dataset2 = self.dataset2_reader.to_dataset()

        old_scores = [dis_video['os'] for dis_video in self.dataset2_reader.dataset.dis_videos]
        new_scores = [dis_video['os'] for dis_video in dataset2.dis_videos]

        self.assertNotEqual(old_scores, new_scores)


class SelectDisVideoDatasetReaderTest(unittest.TestCase):

    def setUp(self):
        dataset_filepath = SurealConfig.test_resource_path('NFLX_dataset_public_raw.py')
        dataset = import_python_file(dataset_filepath)

        np.random.seed(0)
        info_dict = {
            'selected_dis_videos': range(15),
        }

        self.dataset_reader = SelectDisVideoRawDatasetReader(dataset, input_dict=info_dict)

    def test_opinion_score_3darray(self):
        os_3darray = self.dataset_reader.opinion_score_3darray
        self.assertEqual(os_3darray.shape, (15, 26, 1))

    def test_read_dataset_stats(self):
        self.assertEqual(self.dataset_reader.num_ref_videos, 9)
        self.assertEqual(self.dataset_reader.num_dis_videos, 15)
        self.assertEqual(self.dataset_reader.num_observers, 26)
        self.assertEqual(self.dataset_reader.max_repetitions, 1)

    def test_to_dataset(self):
        with self.assertRaises(NotImplementedError):
            self.dataset_reader.to_dataset()


class CorruptSubjectDatasetReaderTestWithCorruptionProb(unittest.TestCase):

    def setUp(self):
        dataset_filepath = SurealConfig.test_resource_path('NFLX_dataset_public_raw.py')
        self.dataset = import_python_file(dataset_filepath)

        np.random.seed(0)

    def test_opinion_score_3darray_with_corruption_prob(self):
        info_dict = {
            'selected_subjects': range(5),
            'corrupt_probability': 0.0,
        }
        self.dataset_reader = CorruptSubjectRawDatasetReader(self.dataset, input_dict=info_dict)
        os_3darray = self.dataset_reader.opinion_score_3darray
        self.assertEqual(os_3darray.shape, (79, 26, 1))
        self.assertAlmostEqual(float(np.mean(np.std(os_3darray, axis=1))), 0.64933186478291516, places=4)

    def test_opinion_score_3darray_with_corruption_prob2(self):
        info_dict = {
            'selected_subjects': range(5),
            'corrupt_probability': 0.2,
        }
        self.dataset_reader = CorruptSubjectRawDatasetReader(self.dataset, input_dict=info_dict)
        os_3darray = self.dataset_reader.opinion_score_3darray
        self.assertEqual(os_3darray.shape, (79, 26, 1))
        self.assertAlmostEqual(float(np.mean(np.std(os_3darray, axis=1))), 0.73123067709849221, places=4)

    def test_opinion_score_3darray_with_corruption_prob3(self):
        info_dict = {
            'selected_subjects': range(5),
            'corrupt_probability': 0.7,
        }
        self.dataset_reader = CorruptSubjectRawDatasetReader(self.dataset, input_dict=info_dict)
        os_3darray = self.dataset_reader.opinion_score_3darray
        self.assertEqual(os_3darray.shape, (79, 26, 1))
        self.assertAlmostEqual(float(np.mean(np.std(os_3darray, axis=1))), 0.85118397722242856, places=4)

    def test_opinion_score_3darray_with_corruption_prob4(self):
        info_dict = {
            'selected_subjects': range(5),
            'corrupt_probability': 1.0,
        }
        self.dataset_reader = CorruptSubjectRawDatasetReader(self.dataset, input_dict=info_dict)
        os_3darray = self.dataset_reader.opinion_score_3darray
        self.assertEqual(os_3darray.shape, (79, 26, 1))
        self.assertAlmostEqual(float(np.mean(np.std(os_3darray, axis=1))), 0.96532565883975119, places=4)

    def test_opinion_score_3darray_with_corruption_prob_flip(self):
        info_dict = {
            'selected_subjects': range(5),
            'corrupt_probability': 0.0,
            'corrupt_behavior': 'flip',
        }
        self.dataset_reader = CorruptSubjectRawDatasetReader(self.dataset, input_dict=info_dict)
        os_3darray = self.dataset_reader.opinion_score_3darray
        self.assertEqual(os_3darray.shape, (79, 26, 1))
        self.assertAlmostEqual(float(np.mean(np.std(os_3darray, axis=1))), 0.64933186478291516, places=4)

    def test_opinion_score_3darray_with_corruption_prob2_flip(self):
        info_dict = {
            'selected_subjects': range(5),
            'corrupt_probability': 0.2,
            'corrupt_behavior': 'flip',
        }
        self.dataset_reader = CorruptSubjectRawDatasetReader(self.dataset, input_dict=info_dict)
        os_3darray = self.dataset_reader.opinion_score_3darray
        self.assertEqual(os_3darray.shape, (79, 26, 1))
        self.assertAlmostEqual(float(np.mean(np.std(os_3darray, axis=1))), 0.8313104324219012, places=4)

    def test_opinion_score_3darray_with_corruption_prob3_flip(self):
        info_dict = {
            'selected_subjects': range(5),
            'corrupt_probability': 0.7,
            'corrupt_behavior': 'flip',
        }
        self.dataset_reader = CorruptSubjectRawDatasetReader(self.dataset, input_dict=info_dict)
        os_3darray = self.dataset_reader.opinion_score_3darray
        self.assertEqual(os_3darray.shape, (79, 26, 1))
        self.assertAlmostEqual(float(np.mean(np.std(os_3darray, axis=1))), 1.105462191789546, places=4)

    def test_opinion_score_3darray_with_corruption_prob4_flip(self):
        info_dict = {
            'selected_subjects': range(5),
            'corrupt_probability': 1.0,
            'corrupt_behavior': 'flip',
        }
        self.dataset_reader = CorruptSubjectRawDatasetReader(self.dataset, input_dict=info_dict)
        os_3darray = self.dataset_reader.opinion_score_3darray
        self.assertEqual(os_3darray.shape, (79, 26, 1))
        self.assertAlmostEqual(float(np.mean(np.std(os_3darray, axis=1))), 1.2037017964018906, places=4)


    def test_opinion_score_3darray_with_corruption_prob_min(self):
        info_dict = {
            'selected_subjects': range(5),
            'corrupt_probability': 0.0,
            'corrupt_behavior': 'min',
        }
        self.dataset_reader = CorruptSubjectRawDatasetReader(self.dataset, input_dict=info_dict)
        os_3darray = self.dataset_reader.opinion_score_3darray
        self.assertEqual(os_3darray.shape, (79, 26, 1))
        self.assertAlmostEqual(float(np.mean(np.std(os_3darray, axis=1))), 0.64933186478291516, places=4)

    def test_opinion_score_3darray_with_corruption_prob2_min(self):
        info_dict = {
            'selected_subjects': range(5),
            'corrupt_probability': 0.2,
            'corrupt_behavior': 'min',
        }
        self.dataset_reader = CorruptSubjectRawDatasetReader(self.dataset, input_dict=info_dict)
        os_3darray = self.dataset_reader.opinion_score_3darray
        self.assertEqual(os_3darray.shape, (79, 26, 1))
        self.assertAlmostEqual(float(np.mean(np.std(os_3darray, axis=1))), 0.8313708086943223, places=4)

    def test_opinion_score_3darray_with_corruption_prob3_min(self):
        info_dict = {
            'selected_subjects': range(5),
            'corrupt_probability': 0.7,
            'corrupt_behavior': 'min',
        }
        self.dataset_reader = CorruptSubjectRawDatasetReader(self.dataset, input_dict=info_dict)
        os_3darray = self.dataset_reader.opinion_score_3darray
        self.assertEqual(os_3darray.shape, (79, 26, 1))
        self.assertAlmostEqual(float(np.mean(np.std(os_3darray, axis=1))), 1.1056046722467132, places=4)

    def test_opinion_score_3darray_with_corruption_prob4_min(self):
        info_dict = {
            'selected_subjects': range(5),
            'corrupt_probability': 1.0,
            'corrupt_behavior': 'min',
        }
        self.dataset_reader = CorruptSubjectRawDatasetReader(self.dataset, input_dict=info_dict)
        os_3darray = self.dataset_reader.opinion_score_3darray
        self.assertEqual(os_3darray.shape, (79, 26, 1))
        self.assertAlmostEqual(float(np.mean(np.std(os_3darray, axis=1))), 1.2039488364011253, places=4)

    def test_opinion_score_3darray_with_corruption_prob3_mid(self):
        info_dict = {
            'selected_subjects': range(5),
            'corrupt_probability': 0.7,
            'corrupt_behavior': 'mid',
        }
        self.dataset_reader = CorruptSubjectRawDatasetReader(self.dataset, input_dict=info_dict)
        os_3darray = self.dataset_reader.opinion_score_3darray
        self.assertEqual(os_3darray.shape, (79, 26, 1))
        self.assertAlmostEqual(float(np.mean(np.std(os_3darray, axis=1))), 0.7698970309018135, places=4)

    def test_opinion_score_3darray_with_corruption_prob3_max(self):
        info_dict = {
            'selected_subjects': range(5),
            'corrupt_probability': 0.7,
            'corrupt_behavior': 'max',
        }
        self.dataset_reader = CorruptSubjectRawDatasetReader(self.dataset, input_dict=info_dict)
        os_3darray = self.dataset_reader.opinion_score_3darray
        self.assertEqual(os_3darray.shape, (79, 26, 1))
        self.assertAlmostEqual(float(np.mean(np.std(os_3darray, axis=1))), 0.8401951693458558, places=4)

    def test_opinion_score_3darray_with_corruption_prob3_constant(self):
        info_dict = {
            'selected_subjects': range(5),
            'corrupt_probability': 0.7,
            'corrupt_behavior': 'constant',
        }
        self.dataset_reader = CorruptSubjectRawDatasetReader(self.dataset, input_dict=info_dict)
        os_3darray = self.dataset_reader.opinion_score_3darray
        self.assertEqual(os_3darray.shape, (79, 26,1))
        self.assertAlmostEqual(float(np.mean(np.std(os_3darray, axis=1))), 0.8358848954630226, places=4)


class CorruptSubjectDatasetReaderTest(unittest.TestCase):

    def setUp(self):
        dataset_filepath = SurealConfig.test_resource_path('NFLX_dataset_public_raw.py')
        dataset = import_python_file(dataset_filepath)

        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
        }

        self.dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)

    def test_read_dataset_stats(self):
        self.assertEqual(self.dataset_reader.num_ref_videos, 9)
        self.assertEqual(self.dataset_reader.num_dis_videos, 79)
        self.assertEqual(self.dataset_reader.num_observers, 26)
        self.assertEqual(self.dataset_reader.max_repetitions, 1)

    def test_opinion_score_3darray(self):
        os_3darray = self.dataset_reader.opinion_score_3darray
        self.assertEqual(os_3darray.shape, (79, 26, 1))
        self.assertAlmostEqual(float(np.mean(np.std(os_3darray, axis=1))), 0.93177573807000225, places=4)

    def test_to_dataset(self):
        dataset = self.dataset_reader.to_dataset()

        old_scores = [dis_video['os'] for dis_video in self.dataset_reader.dataset.dis_videos]
        new_scores = [dis_video['os'] for dis_video in dataset.dis_videos]

        self.assertNotEqual(old_scores, new_scores)


class CorruptSubjectDatasetReaderTestFlip(unittest.TestCase):

    def setUp(self):
        dataset_filepath = SurealConfig.test_resource_path('NFLX_dataset_public_raw.py')
        dataset = import_python_file(dataset_filepath)

        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
            'corrupt_behavior': 'flip',
        }

        self.dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)

    def test_read_dataset_stats(self):
        self.assertEqual(self.dataset_reader.num_ref_videos, 9)
        self.assertEqual(self.dataset_reader.num_dis_videos, 79)
        self.assertEqual(self.dataset_reader.num_observers, 26)
        self.assertEqual(self.dataset_reader.max_repetitions, 1)

    def test_opinion_score_3darray(self):
        os_3darray = self.dataset_reader.opinion_score_3darray
        self.assertEqual(os_3darray.shape, (79, 26, 1))
        self.assertAlmostEqual(float(np.mean(np.std(os_3darray, axis=1))), 1.2037017964018906, places=4)

    def test_to_dataset(self):
        dataset = self.dataset_reader.to_dataset()

        old_scores = [dis_video['os'] for dis_video in self.dataset_reader.dataset.dis_videos]
        new_scores = [dis_video['os'] for dis_video in dataset.dis_videos]

        self.assertNotEqual(old_scores, new_scores)


class CorruptSubjectDatasetReaderTestMin(unittest.TestCase):

    def setUp(self):
        dataset_filepath = SurealConfig.test_resource_path('NFLX_dataset_public_raw.py')
        dataset = import_python_file(dataset_filepath)

        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
            'corrupt_behavior': 'min',
        }

        self.dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)

    def test_read_dataset_stats(self):
        self.assertEqual(self.dataset_reader.num_ref_videos, 9)
        self.assertEqual(self.dataset_reader.num_dis_videos, 79)
        self.assertEqual(self.dataset_reader.num_observers, 26)
        self.assertEqual(self.dataset_reader.max_repetitions, 1)

    def test_opinion_score_3darray(self):
        os_3darray = self.dataset_reader.opinion_score_3darray
        self.assertEqual(os_3darray.shape, (79, 26, 1))
        self.assertAlmostEqual(float(np.mean(np.std(os_3darray, axis=1))), 1.2039488364011253, places=4)

    def test_to_dataset(self):
        dataset = self.dataset_reader.to_dataset()

        old_scores = [dis_video['os'] for dis_video in self.dataset_reader.dataset.dis_videos]
        new_scores = [dis_video['os'] for dis_video in dataset.dis_videos]

        self.assertNotEqual(old_scores, new_scores)


class CorruptSubjectDatasetReaderTestMid(unittest.TestCase):

    def setUp(self):
        dataset_filepath = SurealConfig.test_resource_path('NFLX_dataset_public_raw.py')
        dataset = import_python_file(dataset_filepath)

        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
            'corrupt_behavior': 'mid',
        }

        self.dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)

    def test_read_dataset_stats(self):
        self.assertEqual(self.dataset_reader.num_ref_videos, 9)
        self.assertEqual(self.dataset_reader.num_dis_videos, 79)
        self.assertEqual(self.dataset_reader.num_observers, 26)
        self.assertEqual(self.dataset_reader.max_repetitions, 1)

    def test_opinion_score_3darray(self):
        os_3darray = self.dataset_reader.opinion_score_3darray
        self.assertEqual(os_3darray.shape, (79, 26, 1))
        self.assertAlmostEqual(float(np.mean(np.std(os_3darray, axis=1))), 0.7894505437942192, places=4)

    def test_to_dataset(self):
        dataset = self.dataset_reader.to_dataset()

        old_scores = [dis_video['os'] for dis_video in self.dataset_reader.dataset.dis_videos]
        new_scores = [dis_video['os'] for dis_video in dataset.dis_videos]

        self.assertNotEqual(old_scores, new_scores)


class CorruptSubjectDatasetReaderTestMax(unittest.TestCase):

    def setUp(self):
        dataset_filepath = SurealConfig.test_resource_path('NFLX_dataset_public_raw.py')
        dataset = import_python_file(dataset_filepath)

        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
            'corrupt_behavior': 'max',
        }

        self.dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)

    def test_read_dataset_stats(self):
        self.assertEqual(self.dataset_reader.num_ref_videos, 9)
        self.assertEqual(self.dataset_reader.num_dis_videos, 79)
        self.assertEqual(self.dataset_reader.num_observers, 26)
        self.assertEqual(self.dataset_reader.max_repetitions, 1)

    def test_opinion_score_3darray(self):
        os_3darray = self.dataset_reader.opinion_score_3darray
        self.assertEqual(os_3darray.shape, (79, 26, 1))
        self.assertAlmostEqual(float(np.mean(np.std(os_3darray, axis=1))), 0.8754822173027904, places=4)

    def test_to_dataset(self):
        dataset = self.dataset_reader.to_dataset()

        old_scores = [dis_video['os'] for dis_video in self.dataset_reader.dataset.dis_videos]
        new_scores = [dis_video['os'] for dis_video in dataset.dis_videos]

        self.assertNotEqual(old_scores, new_scores)


class CorruptSubjectDatasetReaderTestConstant(unittest.TestCase):

    def setUp(self):
        dataset_filepath = SurealConfig.test_resource_path('NFLX_dataset_public_raw.py')
        dataset = import_python_file(dataset_filepath)

        np.random.seed(0)
        info_dict = {
            'selected_subjects': range(5),
            'corrupt_behavior': 'constant',
        }

        self.dataset_reader = CorruptSubjectRawDatasetReader(dataset, input_dict=info_dict)

    def test_read_dataset_stats(self):
        self.assertEqual(self.dataset_reader.num_ref_videos, 9)
        self.assertEqual(self.dataset_reader.num_dis_videos, 79)
        self.assertEqual(self.dataset_reader.num_observers, 26)
        self.assertEqual(self.dataset_reader.max_repetitions, 1)

    def test_opinion_score_3darray(self):
        os_3darray = self.dataset_reader.opinion_score_3darray
        self.assertEqual(os_3darray.shape, (79, 26, 1))
        self.assertAlmostEqual(float(np.mean(np.std(os_3darray, axis=1))), 0.7838541915456235, places=4)

    def test_to_dataset(self):
        dataset = self.dataset_reader.to_dataset()

        old_scores = [dis_video['os'] for dis_video in self.dataset_reader.dataset.dis_videos]
        new_scores = [dis_video['os'] for dis_video in dataset.dis_videos]

        self.assertNotEqual(old_scores, new_scores)


class CorruptDataDatasetReaderTest(unittest.TestCase):

    def setUp(self):
        dataset_filepath = SurealConfig.test_resource_path('NFLX_dataset_public_raw.py')
        dataset = import_python_file(dataset_filepath)

        np.random.seed(0)
        info_dict = {
            'corrupt_probability': 0.1,
        }

        self.dataset_reader = CorruptDataRawDatasetReader(dataset, input_dict=info_dict)

    def test_opinion_score_3darray(self):
        os_3darray = self.dataset_reader.opinion_score_3darray
        self.assertAlmostEqual(float(np.mean(np.std(os_3darray, axis=1))), 0.79796204942957094, places=4)

    def test_to_dataset(self):
        dataset = self.dataset_reader.to_dataset()

        old_scores = [dis_video['os'] for dis_video in self.dataset_reader.dataset.dis_videos]
        new_scores = [dis_video['os'] for dis_video in dataset.dis_videos]

        self.assertNotEqual(old_scores, new_scores)


class RawDatasetReaderPCTest(unittest.TestCase):

    def setUp(self):
        dataset_filepath = SurealConfig.test_resource_path('NFLX_dataset_public_raw.py')
        dataset = import_python_file(dataset_filepath)
        self.dataset_reader = RawDatasetReader(dataset)

    def test_dataset_to_pc_dataset(self):
        pc_dataset = self.dataset_reader.to_pc_dataset()
        pc_dataset_reader = PairedCompDatasetReader(pc_dataset)
        opinion_score_3darray = pc_dataset_reader.opinion_score_3darray
        self.assertEqual(np.nansum(opinion_score_3darray), 8242)
        self.assertEqual(np.nanmean(opinion_score_3darray), 0.816039603960396)
        self.assertEqual(np.nanmin(opinion_score_3darray), 0.5)
        self.assertEqual(np.nanmax(opinion_score_3darray), 1.0)

    def test_dataset_to_pc_dataset_within_subject(self):
        pc_dataset = self.dataset_reader.to_pc_dataset(pc_type='within_subject')
        pc_dataset_reader = PairedCompDatasetReader(pc_dataset)
        opinion_score_3darray = pc_dataset_reader.opinion_score_3darray
        self.assertEqual(np.nansum(opinion_score_3darray), 80106)
        self.assertEqual(np.nanmean(opinion_score_3darray), 0.8050935185278244)
        self.assertEqual(np.nanmin(opinion_score_3darray), 0.5)
        self.assertEqual(np.nanmax(opinion_score_3darray), 1.0)

    def test_dataset_to_pc_dataset_coin_toss(self):
        pc_dataset = self.dataset_reader.to_pc_dataset(tiebreak_method='coin_toss')
        pc_dataset_reader = PairedCompDatasetReader(pc_dataset)
        opinion_score_3darray = pc_dataset_reader.opinion_score_3darray
        self.assertEqual(np.nansum(opinion_score_3darray), 8242)
        self.assertEqual(np.nanmean(opinion_score_3darray), 1.0)
        self.assertEqual(np.nanmin(opinion_score_3darray), 1.0)
        self.assertEqual(np.nanmax(opinion_score_3darray), 1.0)

    def test_dataset_to_pc_dataset_random(self):
        import random
        random.seed(0)
        pc_dataset = self.dataset_reader.to_pc_dataset(cointoss_rate=0.5)
        pc_dataset_reader = PairedCompDatasetReader(pc_dataset)
        opinion_score_3darray = pc_dataset_reader.opinion_score_3darray
        self.assertEqual(np.nansum(opinion_score_3darray), 8242)
        # check: python2 values seem to fluctuate quite a bit
        self.assertAlmostEqual(float(np.nanmean(opinion_score_3darray)), 0.8966492602262838, delta=0.01)
        self.assertEqual(np.nanmin(opinion_score_3darray), 0.5)
        self.assertEqual(np.nanmax(opinion_score_3darray), 1.0)

    def test_dataset_to_pc_dataset_sampling_rate(self):
        import random
        random.seed(0)
        pc_dataset = self.dataset_reader.to_pc_dataset(sampling_rate=0.1)
        pc_dataset_reader = PairedCompDatasetReader(pc_dataset)
        opinion_score_3darray = pc_dataset_reader.opinion_score_3darray
        self.assertEqual(np.nansum(opinion_score_3darray), 844)
        self.assertAlmostEqual(float(np.nanmean(opinion_score_3darray)), 0.8107588856868396, delta=0.0001)
        self.assertEqual(np.nanmin(opinion_score_3darray), 0.5)
        self.assertEqual(np.nanmax(opinion_score_3darray), 1.0)

    def test_dataset_to_pc_dataset_sampling_rate_greater_than_1(self):
        import random
        random.seed(0)
        pc_dataset = self.dataset_reader.to_pc_dataset(sampling_rate=2.1)
        pc_dataset_reader = PairedCompDatasetReader(pc_dataset)
        opinion_score_3darray = pc_dataset_reader.opinion_score_3darray
        self.assertEqual(np.nansum(opinion_score_3darray), 8242)
        self.assertAlmostEqual(float(np.nanmean(opinion_score_3darray)), 0.816039603960396, delta=0.0001)
        self.assertEqual(np.nanmin(opinion_score_3darray), 0.5)
        self.assertEqual(np.nanmax(opinion_score_3darray), 1.0)

    def test_dataset_to_pc_dataset_per_asset_sampling_rates(self):
        import random
        random.seed(0)
        pc_dataset = self.dataset_reader.to_pc_dataset(per_asset_sampling_rates=np.hstack([np.ones(39), np.ones(40) * 0.1]))
        pc_dataset_reader = PairedCompDatasetReader(pc_dataset)
        opinion_score_3darray = pc_dataset_reader.opinion_score_3darray
        self.assertEqual(np.nansum(opinion_score_3darray), 4546)
        self.assertAlmostEqual(float(np.nanmean(opinion_score_3darray)), 0.8116303960042811, delta=0.01)
        self.assertEqual(np.nanmin(opinion_score_3darray), 0.5)
        self.assertEqual(np.nanmax(opinion_score_3darray), 1.0)

    def test_dataset_to_pc_dataset_per_asset_cointoss_rates(self):
        import random
        random.seed(0)
        pc_dataset = self.dataset_reader.to_pc_dataset(per_asset_cointoss_rates=np.hstack([np.ones(39), np.ones(40) * 0.1]))
        pc_dataset_reader = PairedCompDatasetReader(pc_dataset)
        opinion_score_3darray = pc_dataset_reader.opinion_score_3darray
        self.assertEqual(np.nansum(opinion_score_3darray), 8242)
        self.assertAlmostEqual(float(np.nanmean(opinion_score_3darray)), 0.91102022769979, delta=0.01)
        self.assertEqual(np.nanmin(opinion_score_3darray), 0.5)
        self.assertEqual(np.nanmax(opinion_score_3darray), 1.0)

    def test_dataset_to_pc_dataset_per_asset_noise_levels(self):
        import random
        random.seed(0)
        pc_dataset = self.dataset_reader.to_pc_dataset(per_asset_noise_levels=np.hstack([np.ones(39), np.ones(40) * 0.1]))
        pc_dataset_reader = PairedCompDatasetReader(pc_dataset)
        opinion_score_3darray = pc_dataset_reader.opinion_score_3darray
        self.assertEqual(np.nansum(opinion_score_3darray), 8242)
        self.assertAlmostEqual(float(np.nanmean(opinion_score_3darray)), 1.0, delta=0.0001)
        self.assertEqual(np.nanmin(opinion_score_3darray), 1.0)
        self.assertEqual(np.nanmax(opinion_score_3darray), 1.0)

    def test_dataset_to_pc_dataset_within_subject_per_asset_mean_scores(self):
        pc_dataset = self.dataset_reader.to_pc_dataset(pc_type='within_subject', per_asset_mean_scores=np.ones(79))
        pc_dataset_reader = PairedCompDatasetReader(pc_dataset)
        opinion_score_3darray = pc_dataset_reader.opinion_score_3darray
        self.assertEqual(np.nansum(opinion_score_3darray), 80106)
        self.assertEqual(np.nanmean(opinion_score_3darray), 0.5)
        self.assertEqual(np.nanmin(opinion_score_3darray), 0.5)
        self.assertEqual(np.nanmax(opinion_score_3darray), 0.5)


if __name__ == '__main__':
    unittest.main()
