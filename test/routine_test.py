import glob
import os
import shutil
import unittest
import warnings
from sys import platform

import matplotlib.pyplot as plt
from sureal.config import SurealConfig, DisplayConfig
from sureal.dataset_reader import PairedCompDatasetReader, \
    SelectDisVideoRawDatasetReader
from sureal.pc_subjective_model import ThurstoneMlePairedCompSubjectiveModel, \
    BradleyTerryMlePairedCompSubjectiveModel
from sureal.routine import run_subjective_models, \
    format_output_of_run_subjective_models
from sureal.subjective_model import MosModel, SubjectMLEModelProjectionSolver, \
    MaximumLikelihoodEstimationModel, SubjrejMosModel, BiasremvSubjrejMosModel, \
    LegacyMaximumLikelihoodEstimationModel
from sureal.tools.misc import MyTestCase


class RunSubjectiveModelsTest(MyTestCase):

    def setUp(self):
        super().setUp()
        self.dataset_filepath = SurealConfig.test_resource_path('NFLX_dataset_public_raw.py')
        self.output_dir = SurealConfig.workdir_path('routine_test')

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        super().tearDown()

    def test_run_subjective_models(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dataset, subjective_models, results = run_subjective_models(
                dataset_filepath=self.dataset_filepath,
                subjective_model_classes=[MosModel, SubjectMLEModelProjectionSolver],
            )
        self.assertEqual(len(dataset.dis_videos), 79)
        self.assertEqual(len(dataset.ref_videos), 9)
        self.assertEqual(len(subjective_models), 2)
        self.assertTrue(len(results) == 2)
        self.assertAlmostEqual(results[0]['quality_scores'][0], 4.884615384615385, places=4)
        self.assertAlmostEqual(results[1]['quality_scores'][0], 4.918072566018977, places=4)
        self.assertAlmostEqual(results[0]['quality_scores'][-2], 4.230769230769231, places=4)
        self.assertAlmostEqual(results[1]['quality_scores'][-2], 4.258802804871559, places=4)
        self.assertAlmostEqual(results[0]['reconstruction_stats']['cc'], 0.8669409273761887, places=4)
        self.assertAlmostEqual(results[0]['reconstruction_stats']['srocc'], 0.8398755052867866, places=4)
        self.assertAlmostEqual(results[0]['reconstruction_stats']['rmse'], 0.6805366284146919, places=4)
        self.assertAlmostEqual(results[1]['reconstruction_stats']['cc'], 0.8934493186953418, places=4)
        self.assertAlmostEqual(results[1]['reconstruction_stats']['srocc'], 0.8698396941667169, places=4)
        self.assertAlmostEqual(results[1]['reconstruction_stats']['rmse'], 0.6134731448525833, places=4)

    def test_run_subjective_models_selected_dis_videos(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dataset, subjective_models, results = run_subjective_models(
                dataset_filepath=self.dataset_filepath,
                subjective_model_classes=[MosModel, SubjectMLEModelProjectionSolver],
                dataset_reader_class=SelectDisVideoRawDatasetReader,
                dataset_reader_info_dict={'selected_dis_videos': range(78)},
            )
        self.assertEqual(len(dataset.dis_videos), 79)  # returned dis_videos not filtered
        self.assertEqual(len(dataset.ref_videos), 9)
        self.assertEqual(len(subjective_models), 2)
        self.assertTrue(len(results) == 2)
        self.assertEqual(len(results[0]['quality_scores']), 78)
        self.assertEqual(len(results[1]['quality_scores']), 78)
        self.assertAlmostEqual(results[0]['quality_scores'][0], 4.884615384615385, places=4)
        self.assertAlmostEqual(results[1]['quality_scores'][0], 4.919299145524252, places=4)
        self.assertAlmostEqual(results[0]['quality_scores'][-1], 4.230769230769231, places=4)
        self.assertAlmostEqual(results[1]['quality_scores'][-1], 4.253960932963961, places=4)
        self.assertTrue('dis_video_names' in results[0])
        self.assertTrue('dis_video_names' in results[1])
        self.assertEqual(len(results[0]['dis_video_names']), 78)
        self.assertEqual(len(results[1]['dis_video_names']), 78)
        self.assertTrue('observers' not in results[0])
        self.assertTrue('observers' not in results[1])

    def test_run_subjective_models_with_plots(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            run_subjective_models(
                dataset_filepath=self.dataset_filepath,
                subjective_model_classes=[MosModel, SubjectMLEModelProjectionSolver],
                do_plot=['raw_scores', 'quality_scores', 'subject_scores']
            )
        DisplayConfig.show(write_to_dir=self.output_dir)
        self.assertEqual(len(glob.glob(os.path.join(self.output_dir, '*.png'))), 3)

    def test_run_subjective_models_with_processed_output(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dataset, subjective_models, results = run_subjective_models(
                dataset_filepath=self.dataset_filepath,
                subjective_model_classes=[
                    MosModel, SubjrejMosModel, BiasremvSubjrejMosModel,
                    SubjectMLEModelProjectionSolver,
                    LegacyMaximumLikelihoodEstimationModel,
                    MaximumLikelihoodEstimationModel],
            )
        output = format_output_of_run_subjective_models(dataset, subjective_models, results)
        self.assertAlmostEqual(output['stats']['models']['BR_SR_MOS']['aic'], 1.863316331686836, places=4)


class RunSubjectiveModelsTestDictStyle(MyTestCase):

    def setUp(self):
        super().setUp()
        self.dataset_filepath = SurealConfig.test_resource_path('test_dataset_os_as_dict.py')

    def test_run_subjective_models(self):
        dataset, subjective_models, results = run_subjective_models(
            dataset_filepath=self.dataset_filepath,
            subjective_model_classes=[MosModel, SubjectMLEModelProjectionSolver],
        )
        self.assertEqual(len(dataset.dis_videos), 3)
        self.assertEqual(len(dataset.ref_videos), 2)
        self.assertEqual(len(subjective_models), 2)
        self.assertTrue(len(results) == 2)
        self.assertAlmostEqual(results[0]['quality_scores'][0], 2.6666666666666665, places=4)
        self.assertAlmostEqual(results[1]['quality_scores'][0], 2.444444443373016, places=4)
        self.assertAlmostEqual(results[0]['quality_scores'][-2], 2.0, places=4)
        self.assertAlmostEqual(results[1]['quality_scores'][-2], 1.44444445515873, places=4)
        self.assertTrue('dis_video_names' in results[0])
        self.assertTrue('dis_video_names' in results[1])
        self.assertEqual(len(results[0]['dis_video_names']), 3)
        self.assertEqual(len(results[1]['dis_video_names']), 3)
        self.assertTrue('observers' in results[0])
        self.assertTrue('observers' in results[1])

    def test_run_subjective_models_with_processed_output(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dataset, subjective_models, results = run_subjective_models(
                dataset_filepath=self.dataset_filepath,
                subjective_model_classes=[
                    MosModel, SubjrejMosModel, BiasremvSubjrejMosModel,
                    SubjectMLEModelProjectionSolver,
                    LegacyMaximumLikelihoodEstimationModel,
                    MaximumLikelihoodEstimationModel],
            )
        format_output_of_run_subjective_models(dataset, subjective_models, results)



class RunSubjectiveModelsTestWithSubjReject(MyTestCase):

    def setUp(self):
        super().setUp()
        self.dataset_filepath = SurealConfig.test_resource_path('NFLX_dataset_public_raw_last4outliers.py')
        self.output_dir = SurealConfig.workdir_path('routine_test')

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        super().tearDown()

    def test_run_subjective_models(self):
        fig0, [ax0, ax1] = plt.subplots(nrows=2, ncols=1)
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        fig4, ax4 = plt.subplots()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dataset, subjective_models, results = run_subjective_models(
                dataset_filepath=self.dataset_filepath,
                subjective_model_classes=[SubjrejMosModel, BiasremvSubjrejMosModel],
                do_plot=[
                    'subject_scores',
                    'observer_rejected',
                    'observer_rejected_1st_stats',
                    'observer_rejected_2nd_stats',
                ],
                ax_dict={
                    'ax_observer_bias': ax0,
                    'ax_observer_inconsistency': ax1,
                    'ax_rejected': ax2,
                    'ax_rejected_1st_stats': ax3,
                    'ax_rejected_2nd_stats': ax4,
                },
            )
        self.assertEqual(len(dataset.dis_videos), 79)
        self.assertEqual(len(dataset.ref_videos), 9)
        self.assertEqual(len(subjective_models), 2)
        self.assertTrue(len(results) == 2)
        self.assertAlmostEqual(results[0]['quality_scores'][0], 1.3333333333333333, places=4)
        self.assertAlmostEqual(results[1]['quality_scores'][0], 1.3430848570089073, places=4)
        self.assertAlmostEqual(results[0]['quality_scores'][-2], 4.555555555555555, places=4)
        self.assertAlmostEqual(results[1]['quality_scores'][-2], 4.454195968120018, places=4)
        self.assertAlmostEqual(results[0]['reconstruction_stats']['cc'], 0.8464373811744285, places=4)
        self.assertAlmostEqual(results[0]['reconstruction_stats']['srocc'], 0.8177055264518673, places=4)
        self.assertAlmostEqual(results[0]['reconstruction_stats']['rmse'], 0.7245056285684048, places=4)
        self.assertAlmostEqual(results[1]['reconstruction_stats']['cc'], 0.8623377727911882, places=4)
        self.assertAlmostEqual(results[1]['reconstruction_stats']['srocc'], 0.8315791903547592, places=4)
        self.assertAlmostEqual(results[1]['reconstruction_stats']['rmse'], 0.6740625245804693, places=4)
        DisplayConfig.show(write_to_dir=self.output_dir)
        self.assertEqual(len(glob.glob(os.path.join(self.output_dir, '*.png'))), 4)


class RunPCSubjectiveModelsTest(MyTestCase):

    def setUp(self):
        super().setUp()
        self.dataset_filepath = SurealConfig.resource_path('dataset', 'lukas_pc_dataset.py')

    def test_run_pc_subjective_models(self):
        dataset, subjective_models, results = run_subjective_models(
            dataset_filepath=self.dataset_filepath,
            subjective_model_classes=[
                ThurstoneMlePairedCompSubjectiveModel,
                BradleyTerryMlePairedCompSubjectiveModel],
            dataset_reader_class=PairedCompDatasetReader,
        )
        self.assertEqual(len(dataset.dis_videos), 40)
        self.assertEqual(len(dataset.ref_videos), 5)
        self.assertEqual(len(subjective_models), 2)
        self.assertTrue(len(results) == 2)
        self.assertAlmostEqual(results[0]['quality_scores'][0], 0.3511818058086143, places=4)
        self.assertAlmostEqual(results[1]['quality_scores'][0], -3.8611464747725552, places=4)
        self.assertAlmostEqual(results[0]['quality_scores'][-2], -0.05230527302857299, places=4)
        self.assertAlmostEqual(results[1]['quality_scores'][-2], -4.3964501689472275, places=4)
        self.assertTrue('observers' in results[0])
        self.assertTrue('observers' in results[1])

    def test_run_pc_subjective_models_with_processed_output(self):
        dataset, subjective_models, results = run_subjective_models(
            dataset_filepath=self.dataset_filepath,
            subjective_model_classes=[
                ThurstoneMlePairedCompSubjectiveModel,
                BradleyTerryMlePairedCompSubjectiveModel],
            dataset_reader_class=PairedCompDatasetReader,
        )
        format_output_of_run_subjective_models(dataset, subjective_models, results)
