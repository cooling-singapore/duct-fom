import os
import tempfile

import h5py

from proc_sim.processor import get_simulation_progress, Parameters, postprocess, convert_to_abs_timestamps, \
    check_palmrun_logs, ProcessorRuntimeError, extract_bounding_box, BoundingBox

test_data_path = os.environ['TEST_DATA_PATH']


def test_get_simulation_progress():
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as f:
        f.write('CBD_TEST_DUCT\n')
        f.write(' 0.30  0.30\n')
        f.flush()

        progress = get_simulation_progress(f.name)
        print(progress)
        assert (progress == 30)


def test_postprocess():
    # read parameters
    parameters_path = os.path.join(test_data_path, 'ucmpalm_sim', 'input', 'parameters')
    parameters = Parameters.parse_file(parameters_path)

    bbox = BoundingBox(
        north=1.2882939577102661,
        east=103.85808563232422,
        south=1.2709641456604004,
        west=103.84086608886719
    )

    # do postprocessing
    raw_output_folder = os.path.join(test_data_path, 'ucmpalm_sim', 'intermediate')
    vv_output_folder = os.path.join(test_data_path, 'ucmpalm_sim', 'output')
    postprocess(raw_output_folder, parameters, bbox, vv_output_folder)


def test_abs_timestamps():
    relative = [3601, 7200, 10800, 14400, 18000, 21600, 25200, 28800, 32400, 36000, 39600, 43200, 46800, 50400,
                54001, 57600, 61201, 64800, 68401, 72000, 75600, 79200, 82800, 86400]

    absolute = convert_to_abs_timestamps(relative, '2020-07-01 06:00:00')
    print(absolute)
    assert (absolute == [20200701070000, 20200701080000, 20200701090000, 20200701100000, 20200701110000,
                         20200701120000, 20200701130000, 20200701140000, 20200701150000, 20200701160000,
                         20200701170000, 20200701180000, 20200701190000, 20200701200000, 20200701210000,
                         20200701220000, 20200701230000, 20200702000000, 20200702010000, 20200702020000,
                         20200702030000, 20200702040000, 20200702050000, 20200702060000])


def test_palmsim_logs():
    log_path = os.path.join(test_data_path, 'ucmpalm_sim', 'output', 'default_d6bbe5cc.21171')
    try:
        check_palmrun_logs(log_path)
        assert False

    except ProcessorRuntimeError as e:
        print(e)
        assert 'Error during simulation run' in e.reason and 'ID: PA0496' in e.details['message']


def test_extract_bounding_box_from_static_driver():
    static_driver_path = os.path.join(test_data_path, 'ucmpalm_prep', 'output', 'static_driver.nc_root')
    bbox = extract_bounding_box(static_driver_path)
    assert (bbox is not None)
    assert bbox == {
        'north': 1.2851252555847168,
        'east': 103.85179138183594,
        'south': 1.267795443534851,
        'west': 103.83457946777344
    }
