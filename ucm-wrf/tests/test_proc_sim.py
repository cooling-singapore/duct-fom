import os
import sys

from tests.conftest import copy_files

sys.path.append(os.path.join(os.getcwd(), '..', 'proc_sim'))

import tempfile

from proc_sim.processor import postprocess, ProgressListener, Severity

INPUT_PATH = os.path.join(os.environ['TEST_DATA_PATH'], 'ucmwrf_sim', 'input')
OUTPUT_PATH = os.path.join(os.environ['TEST_DATA_PATH'], 'ucmwrf_sim', 'output')


def test_postprocess():
    with tempfile.TemporaryDirectory() as temp_dir:
        mappings = [{
            'from': os.path.join(OUTPUT_PATH, 'vv-package'), 'to': os.path.join(temp_dir, 'wrf-prep-vv-package')
        }, {
            'from': os.path.join(OUTPUT_PATH, 'run.a1ec2fca', 'information.json'), 'to': os.path.join(temp_dir, 'information')
        }]
        copy_files(mappings)

        run_path = os.path.join(OUTPUT_PATH, 'run.a1ec2fca')

        class LegacyListener(ProgressListener):
            def on_progress_update(self, progress: float) -> None:
                print(f"trigger:progress:{int(progress)}")

            def on_output_available(self, output_name: str) -> None:
                print(f"trigger:output:{output_name}")

            def on_message(self, severity: Severity, message: str) -> None:
                print(f"trigger:message:{severity.value}:{message}")

        # create the processor object and run the
        callback = LegacyListener()

        # perform post-processing
        postprocess(run_path, temp_dir, callback)
        assert os.path.isfile(os.path.join(temp_dir, 'd04-near-surface-climate'))
