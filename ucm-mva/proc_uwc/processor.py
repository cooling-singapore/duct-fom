import sys

from saas.core.logging import Logging
from saas.sdk.adapter import print_progress, print_output

sys.path.insert(0, '..')
sys.path.insert(0, '../dependencies/python')

from dependencies.python.ExtractFeatures import ExtractFeatures
from dependencies.python.LCP_analysis import LCP_analysis
from dependencies.python.ReadData import ReadData

import json
import os
import traceback
import numpy as np
import logging

from geopy import distance
from helpers import rasterise_land_mask, rasterise_building_heights, export_as_tiff

from PIL import Image
Image.MAX_IMAGE_PIXELS = None   # disables DecompressionBombWarning

logger = Logging.get('ucm-mva-uwc', level=logging.DEBUG)


def generate_output(working_directory: str, name: str, bounding_box: dict, FAD, Mask, DirectionWind) -> None:
    # generate the output
    output = LCP_analysis(FAD, Mask, DirectionWind)
    output = output.astype(np.float32)

    # normalise it
    max_value = np.max(output)
    output /= max_value

    # write the output to disk
    output_path = os.path.join(working_directory, name)
    export_as_tiff(output, bounding_box, output.shape[0], output.shape[1], output_path, dtype=np.float32)
    print_output(name)


def function(working_directory: str):
    try:
        print_progress(0)

        # read parameters and determine bounding box
        parameters_path = os.path.join(working_directory, "parameters")
        with open(parameters_path, "r") as f:
            parameters = json.load(f)
        bounding_box = parameters['bounding_box']

        # determine domain size in meters
        mid_lat = 0.5*(bounding_box['north'] + bounding_box['south'])
        mid_lon = 0.5*(bounding_box['west'] + bounding_box['east'])
        domain_width_m = distance.distance((mid_lat, bounding_box['west']), (mid_lat, bounding_box['east'])).m
        domain_height_m = distance.distance((bounding_box['north'], mid_lon), (bounding_box['south'], mid_lon)).m
        print_progress(5)

        # determine width and height so that the resolution is 1m
        n_width = int(domain_width_m / 10)
        n_height = int(domain_height_m / 10)
        shape = (n_height, n_width)

        # determine cell size
        cell_width_m = domain_width_m / shape[1]
        cell_height_m = domain_height_m / shape[0]
        print(f"cell size: {cell_width_m} x {cell_height_m} meters")

        # rasterise building footprints
        bf_path0 = os.path.join(working_directory, 'building-footprints')
        bf_path1 = os.path.join(working_directory, 'building-footprints.geojson')
        os.rename(bf_path0, bf_path1)
        rasterise_building_heights(logger, bf_path1, bounding_box, shape, bf_path0)
        print_output('building-footprints')
        print_progress(20)

        # rasterise city admin zones
        lm_path0 = os.path.join(working_directory, 'land-mask')
        lm_path1 = os.path.join(working_directory, 'land-mask.geojson')
        os.rename(lm_path0, lm_path1)
        rasterise_land_mask(logger, lm_path1, bounding_box, shape, lm_path0)
        print_output('land-mask')
        print_progress(40)

        # determine unit size such that the resulting resolution is achieved
        unit = 10  # hardcoded inside the model
        resolution = parameters['resolution']
        UnitSize = int(resolution / unit)
        print(f"using unit={unit} resolution={resolution} -> UnitSize={UnitSize}")

        # UnitSize = parameters['unit_size']
        bh = ReadData(lm_path0, bf_path0)
        FAD, Lambda_p, VABH, Mask, FAD_mean, z_d, z_0 = ExtractFeatures(bh, UnitSize)
        print_progress(60)

        generate_output(working_directory, 'wind-corridors-ns', bounding_box, FAD, Mask, 0)
        print_progress(70)

        generate_output(working_directory, 'wind-corridors-ew', bounding_box, FAD, Mask, 1)
        print_progress(80)

        generate_output(working_directory, 'wind-corridors-nwse', bounding_box, FAD, Mask, 2)
        print_progress(90)

        generate_output(working_directory, 'wind-corridors-nesw', bounding_box, FAD, Mask, 3)
        print_progress(100)

        logger.info(f"done.")
        success = True

    except RuntimeError as e:
        trace = ''.join(traceback.format_exception(None, e, e.__traceback__))
        logger.error(f"runtime exception encountered:\n{trace}")
        success = False

    except Exception as e:
        trace = ''.join(traceback.format_exception(None, e, e.__traceback__))
        logger.error(f"unhandled exception in function:\n{trace}")
        success = False

    return 0 if success else -1


if __name__ == "__main__":
    _working_directory = sys.argv[1]

    # setup the logger
    _logfile_path = os.path.join(_working_directory, 'log')
    logger = Logging.get('ucm-mva-uwc', level=logging.DEBUG, custom_log_path=_logfile_path)

    return_code = function(_working_directory)
    sys.exit(return_code)
