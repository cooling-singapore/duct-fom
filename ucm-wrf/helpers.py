import json
import logging
import os
import random
import string
import subprocess

import rasterio
import numpy as np
import scipy.io.netcdf as nc


from jsonschema import validate
from rasterio.crs import CRS

color_ranges = {
    't2': [
        {
            'value_range': [0, 27],
            'red_range': [255, 0],
            'green_range': [255, 0],
            'blue_range': [255, 255]
        },
        {
            'value_range': [20, 27],
            'red_range': [0, 255],
            'green_range': [0, 255],
            'blue_range': [255, 0]
        },
        {
            'value_range': [27, 34],
            'red_range': [255, 255],
            'green_range': [255, 0],
            'blue_range': [0, 0]
        },
        {
            'value_range': [34, 100],
            'red_range': [255, 0],
            'green_range': [0, 0],
            'blue_range': [0, 0]
        }
    ],
    'rh2': [
        {
            'value_range': [0, 100],
            'red_range': [255, 0],
            'green_range': [255, 0],
            'blue_range': [255, 255]
        }
    ],
    'ws10': [
        {
            'value_range': [0, 10],
            'red_range': [255, 0],
            'green_range': [255, 255],
            'blue_range': [255, 255]
        }
    ],
    'wd10': [
        {
            'value_range': [0, 90],
            'red_range': [255, 255],
            'green_range': [0, 255],
            'blue_range': [0, 0]
        },
        {
            'value_range': [90, 180],
            'red_range': [255, 0],
            'green_range': [255, 255],
            'blue_range': [0, 0]
        },
        {
            'value_range': [180, 270],
            'red_range': [0, 0],
            'green_range': [255, 0],
            'blue_range': [0, 255]
        },
        {
            'value_range': [270, 360],
            'red_range': [0, 255],
            'green_range': [0, 0],
            'blue_range': [255, 0]
        }
    ]
}


def generate_random_string(length, characters=string.ascii_letters+string.digits):
    return ''.join(random.choice(characters) for _ in range(length))


def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def check_environment_variables(logger, required):
    # check if the required environment variables are set
    missing = []
    for k in required:
        if k in os.environ:
            value = os.environ[k]
            logger.debug(f"environment: {k}={value}")

        else:
            missing.append(k)

    if len(missing) > 0:
        raise Exception(f"Environment variable(s) missing: {missing}")


def load_json_from_file(path, schema=None):
    with open(path, 'r') as f:
        content = json.load(f)

        # do we have a schema to validate?
        if schema is not None:
            validate(instance=content, schema=schema)

        return content


def write_json_to_file(content, path, schema=None, indent=4, sort_keys=False):
    with open(path, 'w') as f:
        json.dump(content, f, indent=indent, sort_keys=sort_keys)

        # do we have a schema to validate?
        if schema is not None:
            validate(instance=content, schema=schema)

        return content


def extract_bounding_box(input_path: str) -> dict:
    # source: https://www2.mmm.ucar.edu/wrf/users/docs/user_guide_v4/v4.2/users_guide_chap3.html
    # extract the bounding box using 'corner_lats' and 'corner_lons"
    with nc.netcdf_file(input_path, 'r', mmap=False) as nc_file:
        lat = nc_file.__getattribute__('corner_lats')
        lon = nc_file.__getattribute__('corner_lons')
        return {
            'west': float(lon[14-1]),
            'north': float(lat[14-1]),
            'east': float(lon[16-1]),
            'south': float(lat[16-1])
        }


def export_as_tiff(data: np.ndarray, tiff_out_path: str, bbox: dict, flip_vertically: bool, dtype) -> None:
    if flip_vertically:
        data = np.flipud(data)

    with rasterio.open(tiff_out_path, 'w+', driver='GTiff',
                       width=data.shape[1],
                       height=data.shape[0],
                       count=1,
                       dtype=dtype,
                       crs=CRS.from_string("EPSG:4326"),
                       transform=rasterio.transform.from_bounds(bbox['west'], bbox['south'],
                                                                bbox['east'], bbox['north'],
                                                                data.shape[1], data.shape[0])
                       ) as dataset:
        dataset.write(data, 1)


def export_as_png(r_data: np.ndarray, g_data: np.ndarray, b_data: np.ndarray, png_out_path: str) -> None:
    rgb = np.dstack((r_data, g_data, b_data))
    rgb = rgb.transpose((2, 0, 1))
    with rasterio.Env():
        with rasterio.open(png_out_path, 'w+', driver='PNG',
                           width=r_data.shape[1],
                           height=r_data.shape[0],
                           count=3,
                           dtype=rgb.dtype,
                           nodata=0,
                           compress='deflate') as f:
            f.write(rgb)


def indices_and_color_values(values_flattened: np.ndarray, v_range: (int, int),
                             r_range: (int, int), g_range: (int, int), b_range: (int, int)) -> (np.ndarray, np.ndarray,
                                                                                                np.ndarray, np.ndarray):

    # make a working copy and set all values outside of the value range to NaN
    temp: np.ndarray = values_flattened.copy()
    temp[((values_flattened < v_range[0]) | (values_flattened > v_range[1]))] = np.nan

    # determine the indices of all non-NaN values
    indices = np.argwhere(~np.isnan(temp))

    # convert values into red, green, blue component values
    r = (r_range[1]-r_range[0]) * ((temp - v_range[0]) / (v_range[1]-v_range[0])) + r_range[0]
    g = (g_range[1]-g_range[0]) * ((temp - v_range[0]) / (v_range[1]-v_range[0])) + g_range[0]
    b = (b_range[1]-b_range[0]) * ((temp - v_range[0]) / (v_range[1]-v_range[0])) + b_range[0]

    # get the color values only
    r = r[indices]
    g = g[indices]
    b = b[indices]

    return indices, r, g, b


def values_to_rgb(values: np.ndarray, color_ranges: list,
                  background: (int, int, int) = (255, 255, 255)) -> (np.ndarray, np.ndarray, np.ndarray):

    dx = values.shape[0]
    dy = values.shape[1]
    values_flattened = values.flatten()

    r_result = np.full(shape=(dx*dy), fill_value=np.nan)
    g_result = np.full(shape=(dx*dy), fill_value=np.nan)
    b_result = np.full(shape=(dx*dy), fill_value=np.nan)

    # process each color range component separately
    for item in color_ranges:
        v_range = item['value_range']
        r_range = item['red_range']
        g_range = item['green_range']
        b_range = item['blue_range']

        # determine indices and color values (scaled according to color ranges) for all values
        # that are within the value range.
        indices, r, g, b = indices_and_color_values(values_flattened, v_range, r_range, g_range, b_range)

        r_result.put(indices, r)
        g_result.put(indices, g)
        b_result.put(indices, b)

    # reshape
    r_result = r_result.reshape((dx, dy))
    g_result = g_result.reshape((dx, dy))
    b_result = b_result.reshape((dx, dy))

    # for all remaining NaN, use background color
    r_result[np.isnan(r_result)] = background[0]
    g_result[np.isnan(g_result)] = background[1]
    b_result[np.isnan(b_result)] = background[2]

    r_result = r_result.astype(np.uint8)
    g_result = g_result.astype(np.uint8)
    b_result = b_result.astype(np.uint8)

    return r_result, g_result, b_result


def create_mp4_from_frames(frames_path: str, prefix: str, output_path: str) -> None:
    ffmpeg_path = os.environ['FFMPEG']
    result = subprocess.run([
        ffmpeg_path,
        '-r', '6',
        '-f', 'image2',
        '-i', f"{prefix}_%06d.png",
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-vcodec', 'libx264',
        '-crf', '10',
        '-pix_fmt', 'yuv420p',
        '-y',
        output_path
    ], capture_output=True, cwd=frames_path)

    if result.returncode != 0:
        raise RuntimeError(f"Error while executing ffmpeg: stdout={result.stdout} stderr={result.stderr}")
