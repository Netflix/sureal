import warnings

from sureal.config import SurealConfig
from sureal.dataset_reader import PairedCompDatasetReader, \
    SelectDisVideoRawDatasetReader
from sureal.pc_subjective_model import ThurstoneMlePairedCompSubjectiveModel, \
    BradleyTerryMlePairedCompSubjectiveModel
from sureal.routine import run_subjective_models
from sureal.subjective_model import MosModel, SubjectMLEModelProjectionSolver
from sureal.tools.misc import MyTestCase


class RunSubjectiveModelsTest(MyTestCase):

    def setUp(self):
        super().setUp()
        self.dataset_filepath = SurealConfig.test_resource_path('NFLX_dataset_public_raw.py')

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

    def test_run_subjective_models_selected_dis_videos(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dataset, subjective_models, results = run_subjective_models(
                dataset_filepath=self.dataset_filepath,
                subjective_model_classes=[MosModel, SubjectMLEModelProjectionSolver],
                dataset_reader_class=SelectDisVideoRawDatasetReader,
                dataset_reader_info_dict={'selected_dis_videos': range(78)},
                show_dis_video_names=True,
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
