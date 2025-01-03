import sys

from saas.core.logging import Logging
from saas.sdk.adapter import print_progress, print_output

sys.path.insert(0, '..')
sys.path.insert(0, '../dependencies/python')

from dependencies.python.ExtractFeatures import ExtractFeatures
from dependencies.python.ReadData import ReadData
from dependencies.python.SpatialHotSpotAnalysis import SpatialHotSpotAnalysis

import json
import os
import traceback
import numpy as np
import logging

from geopy import distance

from helpers import export_as_tiff, rasterise_land_mask, rasterise_building_heights

from PIL import Image
Image.MAX_IMAGE_PIXELS = None   # disables DecompressionBombWarning

logger = Logging.get('ucm-wrf-uvp', level=logging.DEBUG)


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
        bh_path = os.path.join(working_directory, 'building-footprints')
        bh_vv_path = os.path.join(working_directory, 'building-footprints.tiff')
        rasterise_building_heights(logger, bh_path, bounding_box, shape, bh_vv_path)
        print_progress(20)

        # rasterise land mask
        mask_path = os.path.join(working_directory, 'land-mask')
        mask_vv_path = os.path.join(working_directory, 'land-mask.tiff')
        rasterise_land_mask(logger, mask_path, bounding_box, shape, mask_vv_path)
        print_progress(40)

        # determine unit size such that the resulting resolution is achieved
        unit = 10  # hardcoded inside the model
        resolution = parameters['resolution']
        UnitSize = int(resolution / unit)
        print(f"using unit={unit} resolution={resolution} -> UnitSize={UnitSize}")

        bh = ReadData(mask_vv_path, bh_vv_path)
        FAD, Lambda_p, VABH, Mask, FAD_mean, z_d, z_0 = ExtractFeatures(bh, UnitSize)
        print_progress(60)

        SpatialNeighborhoodSize = 9
        hotspots = SpatialHotSpotAnalysis(FAD_mean, SpatialNeighborhoodSize, Mask)
        print_progress(80)

        # normalise
        max_value = np.max(hotspots)
        hotspots /= max_value

        # store the input/output data
        export_as_tiff(hotspots,
                       bbox=bounding_box,
                       height=hotspots.shape[0],
                       width=hotspots.shape[1],
                       tiff_out_path=os.path.join(working_directory, 'hotspots'),
                       dtype=np.float32)
        print_output('hotspots')
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
    logger = Logging.get('ucm-wrf-uvp', level=logging.DEBUG, custom_log_path=_logfile_path)

    return_code = function(_working_directory)
    sys.exit(return_code)
