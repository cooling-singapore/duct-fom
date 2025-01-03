import json
from typing import Union

import numpy as np
import rasterio

from rasterio.crs import CRS
from rasterio.features import rasterize


def export_as_tiff(components_or_data: Union[list[dict], np.ndarray], bbox: dict, height: int, width: int,
                   tiff_out_path: str, dtype=np.uint16) -> None:

    if isinstance(components_or_data, np.ndarray):
        data: np.ndarray = components_or_data
        data_list = [data]

    else:
        components: list[dict] = components_or_data
        data_list = [c['data'] for c in components]

    with rasterio.open(tiff_out_path, 'w+', driver='GTiff',
                       width=width,
                       height=height,
                       count=len(data_list),
                       dtype=dtype,
                       crs=CRS.from_string("EPSG:4326"),
                       transform=rasterio.transform.from_bounds(bbox['west'], bbox['south'],
                                                                bbox['east'], bbox['north'],
                                                                width, height)
                       ) as dataset:

        for i, data in enumerate(data_list, 1):
            dataset.write(data, i)


def rasterise_building_heights(logger, input_path: str, bbox: dict, shape: (int, int),
                               vv_path: str = None) -> np.ndarray:
    logger.debug(f"rasterising building heights at {input_path}...")
    with rasterio.Env():
        with open(input_path, 'r') as f:
            geojson = json.load(f)

            # rasterise the building footprint geometries
            geometries = [(feature['geometry'], int(feature['properties']['height'])) for feature in geojson['features']]
            raster = rasterize(geometries,
                               transform=rasterio.transform.from_bounds(bbox['west'], bbox['south'],
                                                                        bbox['east'], bbox['north'],
                                                                        shape[1], shape[0]),
                               out_shape=shape,
                               fill=0
                               )

            if vv_path is not None:
                logger.debug(f"exporting rasterised BH as TIFF to {vv_path}: "
                             f"shape={shape} bbox={bbox}")

                export_as_tiff(raster, bbox, shape[0], shape[1], vv_path, dtype=np.uint16)

            return raster


def rasterise_land_mask(logger, input_path: str, bbox: dict, shape: (int, int), vv_path: str = None) -> np.ndarray:
    logger.debug(f"rasterising land mask at {input_path}...")
    with rasterio.Env():
        with open(input_path, 'r') as f:
            geojson = json.load(f)

            # rasterise the building footprint geometries
            geometries = [(feature['geometry'], 1) for feature in geojson['features']]
            raster = rasterize(geometries,
                               transform=rasterio.transform.from_bounds(bbox['west'], bbox['south'],
                                                                        bbox['east'], bbox['north'],
                                                                        shape[1], shape[0]),
                               out_shape=shape,
                               fill=0
                               )

            if vv_path is not None:
                logger.debug(f"exporting rasterised land mask as TIFF to {vv_path}: "
                             f"shape={shape} bbox={bbox}")

                export_as_tiff(raster, bbox, shape[0], shape[1], vv_path, dtype=np.uint8)

            return raster
