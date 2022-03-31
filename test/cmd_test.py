import os
import shutil

from sureal.config import SurealConfig
from sureal.tools.misc import MyTestCase, run_process


class CommandLineTest(MyTestCase):

    def setUp(self):
        super().setUp()
        self.output_dir = SurealConfig.workdir_path('cmd_test')

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        super().tearDown()

    def test_repetition_with_list(self):
        exe = SurealConfig.root_path('sureal', '__main__.py')
        cmd = "{exe} --dataset {ds} --models MOS P910 --plot-dis-videos --output-dir {od}".format(
            exe=exe, ds=SurealConfig.test_resource_path('test_dataset_os_as_list_with_repetitions.py'),
            od=self.output_dir)
        ret = run_process(cmd, shell=True)
        self.assertEqual(ret, 0)

    def test_repetition_with_dict(self):
        exe = SurealConfig.root_path('sureal', '__main__.py')
        cmd = "{exe} --dataset {ds} --models MOS P910 --plot-dis-videos --output-dir {od}".format(
            exe=exe, ds=SurealConfig.test_resource_path('test_dataset_os_as_dict_with_repetitions.py'),
            od=self.output_dir)
        ret = run_process(cmd, shell=True)
        self.assertEqual(ret, 0)

    def test_repetition_with_pc(self):
        exe = SurealConfig.root_path('sureal', '__main__.py')
        cmd = "{exe} --dataset {ds} --models MOS P910 --plot-dis-videos --output-dir {od}".format(
            exe=exe, ds=SurealConfig.test_resource_path('lukas_pc_dataset.py'),
            od=self.output_dir)
        ret = run_process(cmd, shell=True)
        self.assertEqual(ret, 0)

    def test_deprecated_cmd(self):
        exe = SurealConfig.root_path('sureal', 'cmd_deprecated.py')
        cmd = "{exe} MLE_CO_AP2 {ds} --output-dir {od}".format(
            exe=exe, ds=SurealConfig.test_resource_path('test_dataset_os_as_dict_with_repetitions.py'),
            od=self.output_dir)
        ret = run_process(cmd, shell=True)
        self.assertEqual(ret, 0)
