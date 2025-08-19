import os
import shutil
import tempfile

from proc.processor import function, load_buildings, simplify_geometries, convert_polygons_to_stl

TEST_DATA_PATH = os.environ['TEST_DATA_PATH']
CASE_TEMPLATE_PATH = os.path.join(os.environ['TEST_DATA_PATH'], 'ucmscout', 'case_template')


def test_load_buildings():
    xy_buffer = 50.0
    z_buffer = 25.0
    geojson_path = os.path.join(TEST_DATA_PATH, 'ucmscout', 'input', 'building-footprints')
    features, bmd = load_buildings(geojson_path, xy_buffer, z_buffer)

    min_coords = [int(v) for v in bmd.min_coords]
    max_coords = [int(v) for v in bmd.max_coords]
    assert min_coords == [int(xy_buffer), int(xy_buffer), 0]
    assert max_coords == [632, 744, 75]


def test_convert_geojson_to_stl():
    xy_buffer = 50.0
    z_buffer = 25.0
    geojson_path = os.path.join(TEST_DATA_PATH, 'ucmscout', 'input', 'building-footprints')
    features, bmd, to_wgs84 = load_buildings(geojson_path, xy_buffer, z_buffer)

    simplified_geojson_path = os.path.join(TEST_DATA_PATH, 'ucmscout', 'input', 'building-footprints-simplified.geojson')
    polygons = simplify_geometries(features, to_wgs84=to_wgs84, geojson_path=simplified_geojson_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        stl_path = os.path.join(temp_dir, 'building_footprints.stl')
        convert_polygons_to_stl(polygons, stl_path, bmd)


def test_function():
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ['MODEL_TEMPLATE_PATH'] = os.path.join(TEST_DATA_PATH, 'ucmscout', 'model_template')
        os.environ['CASE_TEMPLATE_PATH'] = os.path.join(TEST_DATA_PATH, 'ucmscout', 'case_template')
        for f in ['parameters', 'building-footprints']:
            shutil.copyfile(os.path.join(TEST_DATA_PATH, 'ucmscout', 'input', f), os.path.join(temp_dir, f))

        case_path = os.path.join(temp_dir, 'case')
        os.makedirs(case_path, exist_ok=True)
        shutil.copytree(
            os.path.join(TEST_DATA_PATH, 'ucmscout', 'output', 'postProcessing'),
            os.path.join(case_path, 'postProcessing')
        )

        function(temp_dir)
        assert os.path.isdir(temp_dir)
