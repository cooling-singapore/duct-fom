import json
import math
import os
import shutil
import tempfile

import sys

sys.path.append('../palm_csd')

from proc_prep.processor import load_parameters, coord_32648_to_4326, coord_4326_to_32648, \
    rasterise_buildings, Parameters, create_coordinate_tiffs, create_static_driver, create_vv_package, \
    create_run_package, rasterise_vegetation, rasterise_landcover, generate_ah_file, BuildingMapping

test_data_path = os.environ['TEST_DATA_PATH']


def test_load_parameters():
    try:
        parameters_path = os.path.join(test_data_path, 'ucmpalm_prep', 'input', 'parameters_2x2')
        p: Parameters = load_parameters(parameters_path)
        assert p is not None
    except Exception:
        assert False


def test_coord_conversion():
    p_xy_m0 = (-10013203.11, 4216415.91)
    p_xy_d0 = coord_32648_to_4326(p_xy_m0)
    p_xy_m1 = coord_4326_to_32648(p_xy_d0)
    print(f"{p_xy_m0} -> {p_xy_d0}")
    print(f"{p_xy_d0} -> {p_xy_m1}")

    m0 = f"{p_xy_m0[0]:.2f},{p_xy_m0[1]:.2f}"
    d0 = f"{p_xy_d0[0]:.2f},{p_xy_d0[1]:.2f}"
    m1 = f"{p_xy_m1[0]:.2f},{p_xy_m1[1]:.2f}"
    print(m0)
    print(d0)
    print(m1)

    assert (m0 == m1)


def test_determine_area_of_interest():
    parameters_path = os.path.join(test_data_path, 'ucmpalm_prep', 'input', 'parameters_2x2')
    parameters = load_parameters(parameters_path)
    print(parameters)

    # the area of interest is dimension of grid x resolution (in m).
    dx = parameters.grid_dim[0] * parameters.resolution[0]
    dy = parameters.grid_dim[1] * parameters.resolution[1]
    dh0 = math.sqrt(dx * dx + dy * dy)
    print(f"area of interest: width={dx}m height={dy}m")

    bbox = parameters.bbox
    print(bbox)
    assert bbox == {
        'west': 103.72988018670357,
        'north': 1.3488615392842194,
        'east': 103.74714410739787,
        'south': 1.331503975344926
    }

    # convert dx into meters
    p0 = coord_4326_to_32648((bbox.west, bbox.north))
    p1 = coord_4326_to_32648((bbox.east, bbox.south))
    dx = p1[0] - p0[0]
    dy = p0[1] - p1[1]
    dh1 = math.sqrt(dx * dx + dy * dy)
    print(f"area of interest: width={dx}m height={dy}m")

    assert int(dh0) == int(dh1)


def test_rasterise_landcover():
    parameters_path = os.path.join(test_data_path, 'ucmpalm_prep', 'input', 'parameters_2x2')
    parameters = load_parameters(parameters_path)

    # determine bbox and shape
    shape = (parameters.grid_dim[1], parameters.grid_dim[0])  # (dy, dx)
    bbox = parameters.bbox

    with tempfile.TemporaryDirectory() as tempdir:
        lc_path = os.path.join(test_data_path, 'ucmpalm_prep', 'input', 'landcover_2x2.geojson')
        vv_paths = {
            'vegetation': os.path.join(tempdir, 'lulc_vegetation.tiff'),
            'patch_height': os.path.join(tempdir, 'lulc_patch_height.tiff'),
            'water': os.path.join(tempdir, 'lulc_water.tiff'),
            'pavement': os.path.join(tempdir, 'lulc_pavement.tiff'),
            'empty_int': os.path.join(tempdir, 'lulc_empty_int.tiff'),
            'empty_float': os.path.join(tempdir, 'lulc_empty_float.tiff'),
            'zt': os.path.join(tempdir, 'lulc_zt.tiff')
        }
        rasterise_landcover(lc_path, bbox, shape, vv_paths)
        for path in vv_paths.values():
            assert os.path.isfile(path)


def test_rasterise_buildings():
    parameters_path = os.path.join(test_data_path, 'ucmpalm_prep', 'input', 'parameters_2x2')
    parameters = load_parameters(parameters_path)

    # determine bbox and shape
    shape = (parameters.grid_dim[1], parameters.grid_dim[0])  # (dy, dx)
    bbox = parameters.bbox

    with tempfile.TemporaryDirectory() as tempdir:
        bld_path = os.path.join(test_data_path, 'ucmpalm_prep', 'input', 'buildings_2x2.geojson')
        vv_paths = {
            'bld_heights': os.path.join(tempdir, 'buildings_heights.tiff'),
            'bld_ids': os.path.join(tempdir, 'buildings_ids.tiff'),
            'bld_types': os.path.join(tempdir, 'buildings_types.tiff')
        }
        rasterise_buildings(bld_path, bbox, shape, vv_paths)
        for path in vv_paths.values():
            assert os.path.isfile(path)


def test_rasterise_vegetation():
    parameters_path = os.path.join(test_data_path, 'ucmpalm_prep', 'input', 'parameters_2x2')
    parameters = load_parameters(parameters_path)

    # determine bbox and shape
    shape = (parameters.grid_dim[1], parameters.grid_dim[0])  # (dy, dx)
    bbox = parameters.bbox

    with tempfile.TemporaryDirectory() as tempdir:
        veg_path = os.path.join(test_data_path, 'ucmpalm_prep', 'input', 'vegetation_2x2.geojson')
        vv_paths = {
            'trees_type': os.path.join(tempdir, 'lulc_trees_type.tiff')
        }
        rasterise_vegetation(veg_path, bbox, shape, vv_paths)
        for path in vv_paths.values():
            assert os.path.isfile(path)


def test_generate_ah_netcdf():
    # create building mapping
    buildings_path = os.path.join(test_data_path, 'ucmpalm_prep', 'input', 'buildings_2x2_with_ah.geojson')
    building_mapping = BuildingMapping()
    with open(buildings_path, 'r') as f:
        geojson = json.load(f)
        for feature in geojson['features']:
            building_mapping.add(feature)

    with tempfile.TemporaryDirectory() as tempdir:
        output_path = os.path.join(tempdir, 'ah.nc')
        generate_ah_file(output_path, building_mapping, (0.00, 0.00), '2005-01-01 22:00:00 +08')
        assert os.path.isfile(output_path)


def test_create_palm_coord_files():
    parameters_path = os.path.join(test_data_path, 'ucmpalm_prep', 'input', 'parameters_2x2')
    parameters = load_parameters(parameters_path)

    # determine bbox and shape
    shape = (parameters.grid_dim[1], parameters.grid_dim[0])  # (dy, dx)

    with tempfile.TemporaryDirectory() as tempdir:
        vv_paths_lonlat = {
            'lat_deg': os.path.join(tempdir, 'coords_lat_deg.tiff'),
            'lon_deg': os.path.join(tempdir, 'coords_lon_deg.tiff'),
            'lat_meters': os.path.join(tempdir, 'coords_lat_meters.tiff'),
            'lon_meters': os.path.join(tempdir, 'coords_lon_meters.tiff'),
        }
        create_coordinate_tiffs(vv_paths_lonlat, shape, parameters.bbox)
        for path in vv_paths_lonlat.values():
            assert os.path.isfile(path)


def test_create_run_package():
    input_path = os.path.join(test_data_path, 'ucmpalm_prep', 'input')

    mappings = {
        'parameters': os.path.join(input_path, 'parameters_2x2'),
        'landcover': os.path.join(input_path, 'landcover_2x2.geojson'),
        'buildings': os.path.join(input_path, 'buildings_2x2_with_ah.geojson'),
        'vegetation': os.path.join(input_path, 'vegetation_2x2.geojson')
    }

    # prepare working directory
    wd_path = os.path.join(test_data_path, 'ucmpalm_prep', 'test_wd_path')
    shutil.rmtree(wd_path, ignore_errors=True)
    os.makedirs(wd_path, exist_ok=True)
    for dst, src in mappings.items():
        shutil.copy(src, os.path.join(wd_path, dst))

    origin_date_time = '2020-01-01 22:00:00 +08'
    parameters = load_parameters(os.path.join(wd_path, 'parameters'))
    templates_path = os.path.join('..', 'templates')
    ah_output_path = os.path.join(wd_path, 'ah.nc')
    origin, building_id_mapping = create_static_driver(parameters, wd_path, wd_path, templates_path)
    generate_ah_file(ah_output_path, building_id_mapping, origin, origin_date_time)
    create_run_package(wd_path, templates_path, parameters, origin_date_time)
    create_vv_package(wd_path)
