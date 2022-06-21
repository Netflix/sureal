import glob
import os
import shutil
import warnings

import numpy as np
import matplotlib.pyplot as plt
from sureal.config import SurealConfig, DisplayConfig
from sureal.dataset_reader import PairedCompDatasetReader, \
    SelectDisVideoRawDatasetReader, SyntheticRawDatasetReader
from sureal.pc_subjective_model import ThurstoneMlePairedCompSubjectiveModel, \
    BradleyTerryMlePairedCompSubjectiveModel
from sureal.routine import run_subjective_models, \
    format_output_of_run_subjective_models, validate_with_synthetic_dataset, \
    plot_scatter_target_vs_compared_models
from sureal.subjective_model import MosModel, SubjectMLEModelProjectionSolver, \
    MaximumLikelihoodEstimationModel, SubjrejMosModel, BiasremvSubjrejMosModel, \
    LegacyMaximumLikelihoodEstimationModel
from sureal.tools.misc import MyTestCase


class RunSubjectiveModelsTest(MyTestCase):

    def setUp(self):
        super().setUp()
        plt.close('all')
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
                do_plot=[
                    'raw_scores',
                    'raw_counts',
                    'raw_counts_per_subject',
                    'raw_scores_minus_quality_scores',  # two plots
                    'raw_scores_minus_quality_scores_and_observer_bias',  # one plot
                    'quality_scores_vs_raw_scores',  # two plots
                    'quality_scores',
                    'subject_scores',
                ]
            )
        DisplayConfig.show(write_to_dir=self.output_dir)
        self.assertEqual(len(glob.glob(os.path.join(self.output_dir, '*.png'))), 10)

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
        self.assertAlmostEqual(output['observers'][0]['models']['Subject_MLE_Projection']['observer_inconsistency'], 0.5823933134761798, places=4)
        self.assertAlmostEqual(output['observers'][0]['models']['Subject_MLE_Projection']['observer_inconsistency_std'], 0.046332724296959504, places=4)
        self.assertAlmostEqual(output['observers'][0]['models']['Subject_MLE_Projection']['observer_inconsistency_ci95'][0], 0.07835936236184773, places=4)
        self.assertAlmostEqual(output['observers'][0]['models']['Subject_MLE_Projection']['observer_scores_mean'], 3.3544303797468356, places=4)
        self.assertAlmostEqual(output['observers'][0]['models']['Subject_MLE_Projection']['observer_scores_std'], 1.4323387503610767, places=4)


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
        plt.close('all')
        self.dataset_filepath = SurealConfig.test_resource_path('NFLX_dataset_public_raw_last4outliers.py')
        self.output_dir = SurealConfig.workdir_path('routine_test')

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        super().tearDown()

    def test_run_subjective_models(self):
        fig, [ax0, ax1, ax2, ax3, ax4] = plt.subplots(nrows=5, ncols=1, figsize=[5, 10])
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
        self.assertEqual(len(glob.glob(os.path.join(self.output_dir, '*.png'))), 1)


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


class ValidateWithSyntheticDatasetTest(MyTestCase):

    def setUp(self):
        super().setUp()
        plt.close('all')
        self.output_dir = SurealConfig.workdir_path('routine_test')
        # data obtained from solving the same dataset using said model
        self.synthetic_result_dict = \
            {'quality_scores': np.array(
                [4.91807256, 4.8920448, 4.723263, 4.72239629, 4.85880777,
                 4.91991739, 4.48600731, 4.73463565, 4.76586898, 1.32907989,
                 2.05897095, 2.42123622, 3.12391317, 3.7776905, 4.59205377,
                 4.517852, 4.67660557, 4.88460785, 4.84800457, 2.02080413,
                 1.92374898, 2.63130756, 2.78195285, 3.54615936, 4.40611339,
                 4.8759889, 4.84952473, 0.99047487, 1.99756922, 2.80529793,
                 3.31447984, 3.91435099, 4.32257571, 4.56064178, 1.15358266,
                 1.55354385, 3.07481672, 3.45605265, 4.1495346, 4.565472,
                 4.70605224, 1.25749464, 2.60327048, 2.91576873, 3.31493981,
                 3.76910693, 4.00281009, 4.41694336, 4.71019276, 4.81670822,
                 1.90927438, 3.09328801, 3.2244551, 4.01977345, 4.59802113,
                 4.93619012, 1.11376932, 1.83316167, 2.43419486, 3.01232203,
                 4.26316394, 4.35329757, 4.83565264, 1.02302023, 2.06464513,
                 2.70032144, 3.1824336, 3.75217955, 3.80573258, 4.23571479,
                 4.52578219, 4.40208189, 4.62308751, 1.60361375, 2.48592136,
                 3.20913978, 3.27959085, 4.2588028, 4.60152165]),
                'observer_bias': np.array(
                    [-0.19036027, -0.2030185, 0.24001947, 0.1134372, 0.30331061,
                     -0.07643622, -0.19036027, 0.24001947, -0.31694255, 0.80963973,
                     -0.03846154, 0.32862707, 0.46786758, -0.05111977, -0.03846154,
                     -0.03846154, 0.03748783, -0.34225901, -0.41820837,
                     -0.10175268,
                     -0.01314508, -0.25365141, -0.30428432, -0.48149951,
                     0.42989289,
                     0.08812074]), 'observer_inconsistency': np.array(
                [0.58239332, 0.56856927, 0.76717883, 0.73096034, 0.60861674,
                 0.79119627, 0.87679172, 0.52398134, 0.70589185, 0.6250092,
                 0.59899925, 0.45263148, 0.65065156, 0.77421657, 0.56747137,
                 0.59314773, 0.4464344, 0.59223842, 0.59935489, 0.55678389,
                 0.51506388, 0.48867207, 0.46020279, 0.64011264, 0.47466541,
                 0.49053095]),
                'content_bias': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
                'content_ambiguity': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
                'seed': 5}

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        super().tearDown()

    def test_validate_with_synthetic_dataset(self):

        fig, [ax0, ax1, ax2] = plt.subplots(nrows=1, ncols=3, figsize=[21, 7])
        ax_dict = {'quality_scores': ax0, 'observer_bias': ax1, 'observer_inconsistency': ax2}

        ret = validate_with_synthetic_dataset(
            synthetic_dataset_reader_class=SyntheticRawDatasetReader,
            subjective_model_classes=[SubjectMLEModelProjectionSolver],
            dataset_filepath=SurealConfig.test_resource_path('NFLX_dataset_public_raw.py'),
            synthetic_result=self.synthetic_result_dict,
            ax_dict=ax_dict,
            delta_thr=4e-3,
            color_dict={},
            marker_dict={},
            do_errorbar=True,
            n_bootstrap=None,
            bootstrap_subjects=None,
            boostrap_dis_videos=None,
            force_subjbias_zeromean=True,
            missing_probability=None,
            measure_runtime=True,
        )
        DisplayConfig.show(write_to_dir=self.output_dir)
        self.assertEqual(len(glob.glob(os.path.join(self.output_dir, '*.png'))), 1)
        self.assertAlmostEqual(np.mean(ret['results']['Subject_MLE_Projection']['quality_scores']), 3.5495806672757197, places=4)
        self.assertAlmostEqual(np.mean(np.mean(self.synthetic_result_dict['quality_scores'])), 3.5447906524050636, places=4)
        self.assertAlmostEqual(ret['results']['Subject_MLE_Projection']['quality_scores_ci_perc'], 94.9367088607595, places=4)
        self.assertAlmostEqual(ret['results']['Subject_MLE_Projection']['observer_bias_ci_perc'], 100.0, places=4)
        self.assertAlmostEqual(ret['results']['Subject_MLE_Projection']['observer_inconsistency_ci_perc'], 92.3076923076923, places=4)
        self.assertTrue('runtime' in ret['results']['Subject_MLE_Projection'])

    def test_validate_with_synthetic_dataset_no_errorbar(self):

        fig, [ax0, ax1, ax2] = plt.subplots(nrows=1, ncols=3, figsize=[21, 7])
        ax_dict = {'quality_scores': ax0, 'observer_bias': ax1, 'observer_inconsistency': ax2}

        validate_with_synthetic_dataset(
            synthetic_dataset_reader_class=SyntheticRawDatasetReader,
            subjective_model_classes=[SubjectMLEModelProjectionSolver],
            dataset_filepath=SurealConfig.test_resource_path('NFLX_dataset_public_raw.py'),
            synthetic_result=self.synthetic_result_dict,
            ax_dict=ax_dict,
            delta_thr=4e-3,
            color_dict={},
            marker_dict={},
            do_errorbar=False,  # <==== test this
            n_bootstrap=None,
            bootstrap_subjects=None,
            boostrap_dis_videos=None,
            force_subjbias_zeromean=True,
            missing_probability=None,
            measure_runtime=True,
        )
        DisplayConfig.show(write_to_dir=self.output_dir)
        self.assertEqual(len(glob.glob(os.path.join(self.output_dir, '*.png'))), 1)


class PlotScatterTargetVsComparedModelsTest(MyTestCase):

    def setUp(self):
        super().setUp()
        plt.close('all')
        self.dataset_path = SurealConfig.test_resource_path('NFLX_dataset_public_raw.py')
        self.output_dir = SurealConfig.workdir_path('routine_test')

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        super().tearDown()

    def test_plot_scatter_target_vs_compared_models(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plot_scatter_target_vs_compared_models(
                ['Subject_MLE_Projection', 'Subject_MLE_Projection'],
                ['SR_MOS', 'BR_SR_MOS'],
                [{'path': self.dataset_path}],
                random_seed=1,
            )
        DisplayConfig.show(write_to_dir=self.output_dir)
        self.assertEqual(len(glob.glob(os.path.join(self.output_dir, '*.png'))), 1)

    def test_plot_scatter_target_vs_compared_models_fractional(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plot_scatter_target_vs_compared_models(
                ['Subject_MLE_Projection', 'Subject_MLE_Projection'],
                ['SR_MOS', 'BR_SR_MOS'],
                [{'path': self.dataset_path}],
                random_seed=1,
                target_subj_fraction=None,
                compared_subj_fraction=0.5,
            )
        DisplayConfig.show(write_to_dir=self.output_dir)
        self.assertEqual(len(glob.glob(os.path.join(self.output_dir, '*.png'))), 1)


class PlotSortedQualityScoresTest(MyTestCase):

    def setUp(self):
        super().setUp()
        plt.close('all')
        self.dataset_path = SurealConfig.test_resource_path('NFLX_dataset_public_raw.py')
        self.output_dir = SurealConfig.workdir_path('routine_test')

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        super().tearDown()

    def test_run_subjective_models_and_plot_ordered_quality_scores(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            run_subjective_models(
                dataset_filepath=self.dataset_path,
                subjective_model_classes=[MosModel, SubjectMLEModelProjectionSolver],
                do_plot=[
                    'quality_scores'
                ],
                sort_quality_scores_in_plot=True
            )
        DisplayConfig.show(write_to_dir=self.output_dir)
        self.assertEqual(len(glob.glob(os.path.join(self.output_dir, '*.png'))), 1)
