import os
import tempfile

from proc_prep.processor import Parameters, generate_eta_levels, create_file_using_template, load_parameters, \
    run_geogrid, inject_geodata, default_p_top_requested, default_z_levels, default_height_limit, rm_mk_dir
from tests.conftest import copy_files

INPUT_PATH = os.path.join(os.environ['TEST_DATA_PATH'], 'ucmwrf_prep', 'input')
OUTPUT_PATH = os.path.join(os.environ['TEST_DATA_PATH'], 'ucmwrf_prep', 'output')


def test_load_parameters():
    try:
        parameters_path = os.path.join(os.environ['TEST_DATA_PATH'], 'ucmwrf_prep', 'input', 'parameters')
        load_parameters(parameters_path)
        assert True
    except Exception:
        assert False

def test_create_file_using_template():
    with tempfile.TemporaryDirectory() as temp_dir:
        template_path = os.path.join(os.getcwd(), '..', 'model', 'index.lulc.template')
        destination_path = os.path.join(temp_dir, 'index.lulc')
        mapping = {
            '###DESCRIPTION###': 'description123'
        }

        create_file_using_template(template_path, destination_path, mapping)
        assert os.path.isfile(destination_path)

def test_generate_eta_levels():
    with tempfile.TemporaryDirectory() as temp_dir:
        height_limit = 300
        p_top_requested = 5000

        # generate ETA levels and determine nz and height conversion
        eta, nz, height_levels = generate_eta_levels(temp_dir, p_top_requested, [
            *list(range(0, 140, 20)),
            *list(range(140, 200, 30)),
            *list(range(200, 400, 100)),
            *list(range(400, 1000, 200)),
            *list(range(1000, 6000, 500)),
            *list(range(6000, 19000, 1000))
        ], height_limit)

        # TODO: need to verify the correct values and use for assertions here
        print(eta)
        print(nz)
        print(height_levels)

        assert os.path.isfile(os.path.join(temp_dir, 'height_conversion.txt'))

def test_run_geogrid():
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ['DATA_PATH'] = temp_dir

        mappings = [{
            'from': os.path.join(os.environ['TEST_DATA_PATH'], 'ucmwrf_prep', 'output', f"geo_em.d0{d+1}.nc"),
            'to': os.path.join(temp_dir, f"geo_em.d0{d+1}.nc")
        } for d in range(4)]
        copy_files(mappings)
        os.makedirs(os.path.join(temp_dir, 'static'), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, 'vv_injection_path'), exist_ok=True)

        parameters_path = os.path.join(os.environ['TEST_DATA_PATH'], 'ucmwrf_prep', 'input', "parameters")
        parameters = Parameters.parse_file(parameters_path)

        n_domains, nx, ny, cell_area = run_geogrid(temp_dir, parameters, simulate_only=True)
        assert n_domains == 4
        assert nx == 210
        assert ny == 129
        assert cell_area == 90000

def test_inject_geodata():
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ['DATA_PATH'] = temp_dir

        mappings = [{
            'from': os.path.join(OUTPUT_PATH, f"geo_em.d0{d+1}.nc"),
            'to': os.path.join(temp_dir, f"geo_em.d0{d+1}.nc")
        } for d in range(4)]

        for fname in ['lcz-map', 'vegfra-map', 'lh-profile', 'sh-profile']:
            mappings.append({
                'from': os.path.join(INPUT_PATH, fname), 'to': os.path.join(temp_dir, fname)
            })

        copy_files(mappings)
        os.makedirs(os.path.join(temp_dir, 'static'), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, 'vv_injection_path'), exist_ok=True)

        parameters_path = os.path.join(os.environ['TEST_DATA_PATH'], 'ucmwrf_prep', 'input', "parameters")
        parameters = Parameters.parse_file(parameters_path)


        # determine ETA levels
        eta_levels, nz, height_levels = generate_eta_levels(temp_dir, default_p_top_requested, default_z_levels,
                                                            default_height_limit)

        # perform geogrid stage
        n_domains, nx, ny, cell_area = run_geogrid(temp_dir, parameters, simulate_only=True)

        # inject geodata
        vv_dir_path = os.path.join(temp_dir, 'vv-contents')
        rm_mk_dir(vv_dir_path)
        inject_geodata(temp_dir, vv_dir_path, n_domains, nx, ny, nz, cell_area, height_levels)
