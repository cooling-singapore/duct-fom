import abc
import datetime
import json
import logging
import shutil
import subprocess
import os
import time
import threading

import netCDF4
import numpy as np
import pyproj
import rasterio
from enum import Enum
from threading import Lock
from typing import Optional, List, Dict, Union, Tuple
from pydantic import BaseModel, Field
from pyproj import CRS
from rasterio.features import rasterize
from saas.core.helpers import generate_random_string
from saas.core.logging import Logging
from saas.dor.schemas import ProcessorDescriptor
from osgeo import gdal

import sys
sys.path.append('..')
sys.path.append('../palm_csd')

from palm_csd.palm_csd.create_driver import create_driver


logger = logging.getLogger('ucm-palm-prep')


class ExceptionContent(BaseModel):
    """
    The content of a SaaS exception.
    """
    id: str = Field(..., title="Id", description="The unique identifier of this exception.")
    reason: str = Field(..., title="Reason", description="The reason that caused this exception.")
    details: Optional[dict] = Field(title="Details", description="Supporting information about this exception.")


class ProcessorRuntimeError(Exception):
    def __init__(self, reason: str, details: dict = None):
        self._content = ExceptionContent(id=generate_random_string(16), reason=reason, details=details)

    @property
    def id(self):
        return self._content.id

    @property
    def reason(self):
        return self._content.reason

    @property
    def details(self):
        return self._content.details

    @property
    def content(self) -> ExceptionContent:
        return self._content


class ProcessorState(Enum):
    UNINITIALISED = 'uninitialised'
    BROKEN = 'broken'
    IDLE = 'idle'

    FAILED = 'failed'
    STARTING = 'starting'
    WAITING = 'waiting'
    BUSY = 'busy'
    STOPPING = 'stopping'
    STOPPED = 'stopped'


class Severity(Enum):
    DEBUG = 'debug'
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'


class JobStatus(BaseModel):
    output: List[str]
    progress: int


class ProcessorStatus(BaseModel):
    errors: List[ExceptionContent]
    state: ProcessorState
    job: Optional[JobStatus]


class ProgressListener(abc.ABC):
    @abc.abstractmethod
    def on_progress_update(self, progress: float) -> None:
        pass

    @abc.abstractmethod
    def on_output_available(self, output_name: str) -> None:
        pass

    @abc.abstractmethod
    def on_message(self, severity: Severity, message: str) -> None:
        pass


class ProcessorBase(abc.ABC):
    def __init__(self, proc_path: str) -> None:
        self._mutex = Lock()
        self._proc_path = proc_path
        self._descriptor = ProcessorDescriptor.parse_file(os.path.join(proc_path, 'descriptor.json'))
        self._state = ProcessorState.IDLE
        self._worker = None
        self._watchdog = threading.Thread(target=self._watch, daemon=True)
        self._watchdog.start()

    @property
    def proc_path(self) -> str:
        return self._proc_path

    def descriptor(self) -> ProcessorDescriptor:
        return self._descriptor

    def state(self) -> ProcessorState:
        with self._mutex:
            return self._state

    def execute(self, wd_path: str, listener: ProgressListener, logger: logging.Logger) -> None:
        with self._mutex:
            if self._state.IDLE:
                self._state = ProcessorState.BUSY
                self._worker = threading.Thread(target=self.run, kwargs={
                    'wd_path': wd_path, 'listener': listener, 'logger': logger
                }, daemon=True)
                self._worker.start()

            else:
                raise ProcessorRuntimeError(f"Processor not idle ({self._state.name}) "
                                            f"-> cannot trigger execution now. Try again later.")

    @abc.abstractmethod
    def run(self, wd_path: str, listener: ProgressListener, logger: logging.Logger) -> None:
        pass

    @abc.abstractmethod
    def cancel(self) -> None:
        pass

    def terminate(self) -> None:
        sys.exit(1)

    def _watch(self) -> None:
        while True:
            try:
                if self._worker:
                    self._worker.join()
                    with self._mutex:
                        self._state = ProcessorState.IDLE
                        self._worker = None
                time.sleep(1)

            except Exception:
                pass


##############################


class BoundingBox(BaseModel):
    west: float
    north: float
    east: float
    south: float


class Parameters(BaseModel):
    name: str
    # location: List[float]  # [x, y]
    bbox: BoundingBox
    resolution: List[int]  # [nx, ny]
    grid_dim: List[int]  # [dx, dy]
    dt_sim: float
    dd_profile: str
    ah_profile: bool


def check_environment_variables(required: list[str]) -> None:
    # check if the required environment variables are set
    missing = []
    for k in required:
        if k in os.environ:
            value = os.environ[k]
            logger.debug(f"environment: {k}={value}")

        else:
            missing.append(k)

    if len(missing) > 0:
        raise ProcessorRuntimeError(f"Environment variable(s) missing: {missing}", details={
            'required': required
        })


def load_parameters(parameters_path: str) -> Parameters:
    # parse file
    logger.info(f"load parameters: path={parameters_path}")
    parameters = Parameters.parse_file(parameters_path)

    logger.info(f"using parameters: {parameters}")
    return parameters


def coord_32648_to_4326(p_xy: (float, float)) -> (float, float):
    in_proj = pyproj.Proj('epsg:32648')
    out_proj = pyproj.Proj('epsg:4326')
    temp = pyproj.transform(in_proj, out_proj, x=p_xy[0], y=p_xy[1])
    return temp[1], temp[0]


def coord_4326_to_32648(p_xy: (float, float)) -> (float, float):
    in_proj = pyproj.Proj('epsg:4326')
    out_proj = pyproj.Proj('epsg:32648')
    return pyproj.transform(in_proj, out_proj, x=p_xy[1], y=p_xy[0])


def export_as_tiff_32648(data: np.ndarray, bbox: BoundingBox, shape: (int, int), tiff_out_path: str,
                         dtype, nodata: Union[int, float] = None) -> None:
    height = shape[0]
    width = shape[1]

    # convert bounding box to meters
    p_nw_m = coord_4326_to_32648((bbox.west, bbox.north))
    p_se_m = coord_4326_to_32648((bbox.east, bbox.south))
    bbox = BoundingBox(west=p_nw_m[0], north=p_nw_m[1], east=p_se_m[0], south=p_se_m[1])

    with rasterio.open(tiff_out_path, 'w+', driver='GTiff', width=width, height=height, count=1, dtype=dtype,
                       nodata=nodata, crs=CRS.from_string("EPSG:32648"),
                       transform=rasterio.transform.from_bounds(bbox.west, bbox.south,
                                                                bbox.east, bbox.north,
                                                                width=width, height=height)
                       ) as dataset:

        dataset.write(data, 1)


def export_as_tiff_4326(data: np.ndarray, bbox: BoundingBox, height: int, width: int, tiff_out_path: str,
                        dtype, nodata: Union[int, float] = None, flip_up: bool = False) -> None:

    if flip_up:
        data = np.flipud(data)

    with rasterio.open(tiff_out_path, 'w+', driver='GTiff', width=width, height=height, count=1, dtype=dtype,
                       nodata=nodata, crs=CRS.from_string("EPSG:4326"),
                       transform=rasterio.transform.from_bounds(bbox.west, bbox.south,
                                                                bbox.east, bbox.north,
                                                                width=width, height=height)
                       ) as dataset:

        dataset.write(data, 1)


# UrbanGeometries landcover types are the following:
# VALUE          DESCRIPTION
# -----          -----------
# soil:1         coarse
# soil:2         medium
# soil:3         medium-fine
# soil:4         fine
# soil:5         very fine
# soil:6         organic
# vegetation:1   bare soil
# vegetation:2   crops, mixed farming
# vegetation:3   short grass
# vegetation:4   evergreen needleleaf trees
# vegetation:5   deciduous needleleaf trees
# vegetation:6   evergreen broadleaf trees
# vegetation:7   deciduous broadleaf trees
# vegetation:8   tall grass
# vegetation:9   desert
# vegetation:10  tundra
# vegetation:11  irrigated crops
# vegetation:12  semi desert
# vegetation:13  ice caps and glaciers
# vegetation:14  bogs and marshes
# vegetation:15  evergreen shrubs
# vegetation:16  deciduous shrubs
# vegetation:17  mixed forest/woodland
# vegetation:18  interrupted forest
# pavement:1     asphalt/concrete mix
# pavement:2     asphalt (asphalt concrete)
# pavement:3     concrete (Portland concrete)
# pavement:4     sett
# pavement:5     paving stones
# pavement:6     cobblestone
# pavement:7     metal
# pavement:8     wood
# pavement:9     gravel
# pavement:10    fine gravel
# pavement:11    pebblestone
# pavement:12    woodchips
# pavement:13    tartan (sports)
# pavement:14    artifical turf (sports)
# pavement:15    clay (sports)
# water:1        lake
# water:2        river
# water:3        ocean
# water:4        pond
# water:5        fountain

lc_pavement_mapping = {
    'soil:1': -127,
    'soil:2': -127,
    'soil:3': -127,
    'soil:4': -127,
    'soil:5': -127,
    'soil:6': -127,
    'vegetation:1': -127,
    'vegetation:2': -127,
    'vegetation:3': -127,
    'vegetation:4': -127,
    'vegetation:5': -127,
    'vegetation:6': -127,
    'vegetation:7': -127,
    'vegetation:8': -127,
    'vegetation:9': -127,
    'vegetation:10': -127,
    'vegetation:11': -127,
    'vegetation:12': -127,
    'vegetation:13': -127,
    'vegetation:14': -127,
    'vegetation:15': -127,
    'vegetation:16': -127,
    'vegetation:17': -127,
    'vegetation:18': -127,
    'pavement:1': 1,
    'pavement:2': 2,
    'pavement:3': 3,
    'pavement:4': 4,
    'pavement:5': 5,
    'pavement:6': 6,
    'pavement:7': 7,
    'pavement:8': 8,
    'pavement:9': 9,
    'pavement:10': 10,
    'pavement:11': 11,
    'pavement:12': 12,
    'pavement:13': 13,
    'pavement:14': 14,
    'pavement:15': 15,
    'water:1': -127,
    'water:2': -127,
    'water:3': -127,
    'water:4': -127,
    'water:5': -127
}

lc_vegetation_mapping = {
    'soil:1': 1,
    'soil:2': 1,
    'soil:3': 1,
    'soil:4': 1,
    'soil:5': 1,
    'soil:6': 1,
    'vegetation:1': 1,
    'vegetation:2': 2,
    'vegetation:3': 3,
    'vegetation:4': 4,
    'vegetation:5': 5,
    'vegetation:6': 6,
    'vegetation:7': 7,
    'vegetation:8': 8,
    'vegetation:9': 9,
    'vegetation:10': 10,
    'vegetation:11': 11,
    'vegetation:12': 12,
    'vegetation:13': 13,
    'vegetation:14': 14,
    'vegetation:15': 15,
    'vegetation:16': 16,
    'vegetation:17': 17,
    'vegetation:18': 18,
    'pavement:1': -127,
    'pavement:2': -127,
    'pavement:3': -127,
    'pavement:4': -127,
    'pavement:5': -127,
    'pavement:6': -127,
    'pavement:7': -127,
    'pavement:8': -127,
    'pavement:9': -127,
    'pavement:10': -127,
    'pavement:11': -127,
    'pavement:12': -127,
    'pavement:13': -127,
    'pavement:14': -127,
    'pavement:15': -127,
    'water:1': -127,
    'water:2': -127,
    'water:3': -127,
    'water:4': -127,
    'water:5': -127
}

lc_water_mapping = {
    'soil:1': -127,
    'soil:2': -127,
    'soil:3': -127,
    'soil:4': -127,
    'soil:5': -127,
    'soil:6': -127,
    'vegetation:1': -127,
    'vegetation:2': -127,
    'vegetation:3': -127,
    'vegetation:4': -127,
    'vegetation:5': -127,
    'vegetation:6': -127,
    'vegetation:7': -127,
    'vegetation:8': -127,
    'vegetation:9': -127,
    'vegetation:10': -127,
    'vegetation:11': -127,
    'vegetation:12': -127,
    'vegetation:13': -127,
    'vegetation:14': -127,
    'vegetation:15': -127,
    'vegetation:16': -127,
    'vegetation:17': -127,
    'vegetation:18': -127,
    'pavement:1': -127,
    'pavement:2': -127,
    'pavement:3': -127,
    'pavement:4': -127,
    'pavement:5': -127,
    'pavement:6': -127,
    'pavement:7': -127,
    'pavement:8': -127,
    'pavement:9': -127,
    'pavement:10': -127,
    'pavement:11': -127,
    'pavement:12': -127,
    'pavement:13': -127,
    'pavement:14': -127,
    'pavement:15': -127,
    'water:1': 1,
    'water:2': 2,
    'water:3': 3,
    'water:4': 4,
    'water:5': 5
}

lc_patch_height_mapping = {
    'soil:1': 0,
    'soil:2': 0,
    'soil:3': 0,
    'soil:4': 0,
    'soil:5': 0,
    'soil:6': 0,
    'vegetation:1': 0,
    'vegetation:2': 2,
    'vegetation:3': 1,
    'vegetation:4': 20,
    'vegetation:5': 20,
    'vegetation:6': 20,
    'vegetation:7': 20,
    'vegetation:8': 1,
    'vegetation:9': 0,
    'vegetation:10': 0,
    'vegetation:11': 0,
    'vegetation:12': 0,
    'vegetation:13': 0,
    'vegetation:14': 3,
    'vegetation:15': 1,
    'vegetation:16': 1,
    'vegetation:17': 5,
    'vegetation:18': 3,
    'pavement:1': 0,
    'pavement:2': 0,
    'pavement:3': 0,
    'pavement:4': 0,
    'pavement:5': 0,
    'pavement:6': 0,
    'pavement:7': 0,
    'pavement:8': 0,
    'pavement:9': 0,
    'pavement:10': 0,
    'pavement:11': 0,
    'pavement:12': 0,
    'pavement:13': 0,
    'pavement:14': 0,
    'pavement:15': 0,
    'water:1': 0,
    'water:2': 0,
    'water:3': 0,
    'water:4': 0,
    'water:5': 0
}

veg_vegetation_mapping = {
    'tree:1': 1,
    'tree:2': 2,
    'tree:3': 3
}


class BuildingMapping:
    def __init__(self):
        self._next_id = 1
        self._id_mapping = {}
        self._bld_ah_profiles = {}
        self._other_ah_profiles = []

    def prune_missing_buildings(self, unique_bld_idx: List[int]) -> None:
        temp0 = {b: a for a, b in self._id_mapping.items()}
        temp1 = self._bld_ah_profiles
        self._id_mapping = {}
        self._bld_ah_profiles = {}
        for idx in unique_bld_idx:
            bld_id = temp0[idx]
            self._id_mapping[bld_id] = idx
            self._bld_ah_profiles[bld_id] = temp1[bld_id]

        # self._id_mapping = {temp0[idx]: idx for idx in unique_bld_idx}
        # self._bld_ah_profiles = {temp0[idx]: self._bld_ah_profiles[temp0[idx]] for idx in unique_bld_idx}
        # print()

    def get(self, building_id: int) -> Optional[int]:
        return self._id_mapping.get(building_id)

    def add(self, feature: dict) -> Optional[int]:
        building_id = feature['properties']['id']

        # determine AH profile as array
        conversion = {'GW': 1e6, 'MW': 1e3, 'KW': 1, 'W': 0.001}
        values = np.zeros(24)
        for key, value in feature['properties'].items():
            if key.startswith('AH_'):
                temp = key.split(':')

                # determine the hour
                h = temp[0].split('_')
                h = int(h[1])

                # convert the value into KW
                values[h] = value * conversion[temp[1]]

        # determine whether it's a building or not
        geometry = feature['geometry']
        if geometry['type'] == 'Point':
            # convert lon/lat into x/y
            point = coord_4326_to_32648(geometry['coordinates'])
            self._other_ah_profiles.append((point, values))

        else:
            self._bld_ah_profiles[building_id] = values

            # do the building id mapping
            if building_id not in self._id_mapping:
                self._id_mapping[building_id] = self._next_id
                logger.info(f"map building: '{building_id}' -> {self._id_mapping[building_id]}")
                self._next_id += 1

        return self._id_mapping.get(building_id)

    def dump(self) -> None:
        print(f"building id mapping:")
        for a, b in self._id_mapping.items():
            print(f"  {a} -> {b}")

    def number_of_buildings(self) -> int:
        return len(self._id_mapping)

    def number_of_others(self) -> int:
        return len(self._other_ah_profiles)

    def building_id_array(self) -> np.ndarray:
        unqiue_bld_idx = list(self._id_mapping.values())
        result = np.zeros(shape=(len(unqiue_bld_idx)))
        for i in range(len(unqiue_bld_idx)):
            result[i] = unqiue_bld_idx[i]
        return result

    def building_ah_array(self) -> np.ndarray:
        result = np.zeros(shape=(24, self.number_of_buildings()))
        for idx, building_id in enumerate(self._id_mapping.keys()):
            if building_id in self._bld_ah_profiles:
                profile = self._bld_ah_profiles[building_id]
                result[:, idx] = profile[:24]

        return result

    def others_ah_array(self) -> np.ndarray:
        result = np.zeros(shape=(24, self.number_of_others()))
        for idx, item in enumerate(self._other_ah_profiles):
            profile = item[1]
            result[:, idx] = profile[:24]

        return result

    def others_x_location(self) -> np.ndarray:
        result = np.zeros(self.number_of_others())
        for idx, item in enumerate(self._other_ah_profiles):
            result[idx] = item[0][0]
        return result

    def others_y_location(self) -> np.ndarray:
        result = np.zeros(self.number_of_others())
        for idx, item in enumerate(self._other_ah_profiles):
            result[idx] = item[0][1]
        return result


def rasterise_landcover(input_path: str, bbox: BoundingBox, shape: (int, int), vv_paths: Dict[str, str]) -> None:
    logger.debug(f"rasterising land-cover information to {vv_paths}...")

    with rasterio.Env():
        with open(input_path, 'r') as f:
            # load features
            geojson = json.load(f)
            features = geojson['features']
            
            # determine transform
            transform = rasterio.transform.from_bounds(bbox.west, bbox.south, bbox.east, bbox.north,
                                                       width=shape[1], height=shape[0])
            
            # create geometry to value mappings
            geometries_pavement = [
                (f['geometry'], lc_pavement_mapping[f['properties']['landcover_type']]) for f in features
            ]

            geometries_vegetation = [
                (f['geometry'], lc_vegetation_mapping[f['properties']['landcover_type']]) for f in features
            ]

            geometries_water = [
                (f['geometry'], lc_water_mapping[f['properties']['landcover_type']]) for f in features
            ]

            geometries_patch_height = [
                (f['geometry'], lc_patch_height_mapping[f['properties']['landcover_type']]) for f in features
            ]

            # rasterise the geometries
            pavement = rasterize(geometries_pavement, transform=transform, out_shape=shape, fill=-127)
            vegetation = rasterize(geometries_vegetation, transform=transform, out_shape=shape, fill=-127)
            water = rasterize(geometries_water, transform=transform, out_shape=shape, fill=-127)
            patch_height = rasterize(geometries_patch_height, transform=transform, out_shape=shape, fill=-127)
            empty_int = np.full(fill_value=-9999, shape=shape, dtype=np.int16)
            empty_float = np.full(fill_value=-9999.9, shape=shape, dtype=np.float32)
            zt = np.full(fill_value=0.0, shape=shape, dtype=np.float32)

            # export the raster as GeoTIFF
            export_as_tiff_32648(vegetation, bbox, shape, vv_paths['vegetation'], dtype=np.int16, nodata=-127)
            export_as_tiff_32648(patch_height, bbox, shape, vv_paths['patch_height'], dtype=np.float32, nodata=-127)
            export_as_tiff_32648(water, bbox, shape, vv_paths['water'], dtype=np.int16, nodata=-127)
            export_as_tiff_32648(pavement, bbox, shape, vv_paths['pavement'], dtype=np.int16, nodata=-127)
            export_as_tiff_32648(empty_int, bbox, shape, vv_paths['empty_int'], dtype=np.int16, nodata=-127)
            export_as_tiff_32648(empty_float, bbox, shape, vv_paths['empty_float'], dtype=np.float32, nodata=-127)
            export_as_tiff_32648(zt, bbox, shape, vv_paths['zt'], dtype=np.float32, nodata=-9999)


def rasterise_buildings(input_path: str, bbox: BoundingBox, shape: (int, int),
                        vv_paths: Dict[str, str]) -> BuildingMapping:
    logger.debug(f"rasterising building information to {vv_paths}...")

    id_mapping = BuildingMapping()

    with rasterio.Env():
        with open(input_path, 'r') as f:
            # load buildings and add to id mapping
            geojson = json.load(f)

            # create geometry to value mappings
            # TODO: need actual mapping based on building type
            geometries_heights = []
            geometries_ids = []
            geometries_types = []
            for feature in geojson['features']:
                # add to id mapping (returns None if it's not a building)
                palm_bld_id = id_mapping.add(feature)

                if palm_bld_id is not None:
                    geometries_heights.append((feature['geometry'], int(feature['properties']['height'])))
                    geometries_ids.append((feature['geometry'], palm_bld_id))
                    geometries_types.append((feature['geometry'], 5))

            # rasterise the geometries
            transform = rasterio.transform.from_bounds(bbox.west, bbox.south, bbox.east, bbox.north,
                                                       width=shape[1], height=shape[0])

            raster_heights = rasterize(geometries_heights, transform=transform, out_shape=shape, fill=0)
            raster_ids = rasterize(geometries_ids, transform=transform, out_shape=shape, fill=0)
            raster_types = rasterize(geometries_types, transform=transform, out_shape=shape, fill=0)

            # export the raster as as GeoTIFF
            export_as_tiff_32648(raster_heights, bbox, shape, vv_paths['bld_heights'], dtype=np.uint16, nodata=0)
            export_as_tiff_32648(raster_ids, bbox, shape, vv_paths['bld_ids'], dtype=np.uint16, nodata=0)
            export_as_tiff_32648(raster_types, bbox, shape, vv_paths['bld_types'], dtype=np.uint16, nodata=0)

            # determine list of unique building ids that are actually present in the rasterised datasets
            # and prune the building id mapping
            unique_bld_ids = np.unique(raster_ids).tolist()
            unique_bld_ids.pop(0)
            id_mapping.prune_missing_buildings(unique_bld_ids)

    return id_mapping


def rasterise_vegetation(input_path: str, bbox: BoundingBox, shape: (int, int), vv_paths: Dict[str, str]) -> None:
    logger.debug(f"rasterising vegetation (trees) information to {vv_paths}...")

    with rasterio.Env():
        with open(input_path, 'r') as f:
            # load features
            geojson = json.load(f)
            features = geojson['features']

            # determine transform
            transform = rasterio.transform.from_bounds(bbox.west, bbox.south, bbox.east, bbox.north,
                                                       width=shape[1], height=shape[0])

            # create geometry to value mappings
            geometries_trees_type = [
                # map the vegetation type to the PALM type - or use 1 as default if type not found
                (f['geometry'], veg_vegetation_mapping.get(f['properties']['vegetation_type'], 1)) for f in features
            ]
            # rasterise the geometries
            trees_type = rasterize(geometries_trees_type, transform=transform, out_shape=shape, fill=-9999)

            # export the raster as GeoTIFF
            export_as_tiff_32648(trees_type, bbox, shape, vv_paths['trees_type'], dtype=np.float32, nodata=-9999)


def create_coordinate_tiffs(vv_paths: Dict[str, str], shape: Tuple[int, int], bbox: BoundingBox) -> Tuple[float, float]:
    cell_width_deg = (bbox.east - bbox.west) / shape[1]
    cell_height_deg = (bbox.north - bbox.south) / shape[0]

    # determine lon/lat in degrees and meters for 2 corners
    p_xy_nw_d = (bbox.west + 0.5*cell_width_deg, bbox.north - 0.5 * cell_height_deg)
    p_xy_se_d = (bbox.east - 0.5*cell_width_deg, bbox.south + 0.5 * cell_height_deg)
    p_xy_nw_m = coord_4326_to_32648(p_xy_nw_d)
    p_xy_se_m = coord_4326_to_32648(p_xy_se_d)

    # determine lon/lat matrix in degrees and meters (based on cell centre)
    lat_deg = np.zeros(shape=shape, dtype=np.float32)
    lon_deg = np.zeros(shape=shape, dtype=np.float32)
    lat_meters = np.zeros(shape=shape, dtype=np.float32)
    lon_meters = np.zeros(shape=shape, dtype=np.float32)
    for j in range(shape[0]):
        y_d = p_xy_nw_d[1] + j * (p_xy_se_d[1] - p_xy_nw_d[1]) / shape[0]
        y_m = p_xy_nw_m[1] + j * (p_xy_se_m[1] - p_xy_nw_m[1]) / shape[0]
        for i in range(shape[1]):
            x_d = p_xy_nw_d[0] + i * (p_xy_se_d[0] - p_xy_nw_d[0]) / shape[1]
            x_m = p_xy_nw_m[0] + i * (p_xy_se_m[0] - p_xy_nw_m[0]) / shape[1]

            lat_deg[j, i] = y_d
            lon_deg[j, i] = x_d
            lat_meters[j, i] = y_m
            lon_meters[j, i] = x_m

    export_as_tiff_32648(lat_deg, bbox, shape, vv_paths['lat_deg'], dtype=np.float32)
    export_as_tiff_32648(lon_deg, bbox, shape, vv_paths['lon_deg'], dtype=np.float32)
    export_as_tiff_32648(lat_meters, bbox, shape, vv_paths['lat_meters'], dtype=np.float32)
    export_as_tiff_32648(lon_meters, bbox, shape, vv_paths['lon_meters'], dtype=np.float32)

    return lon_deg[0][0], lat_deg[0][0]


def create_netcdf_files(all_tiff: Dict[str, str], nc_path: str) -> Dict[str, str]:
    # iterate over the tiffs and convert into netcdf
    all_netcdf: Dict[str, str] = {}
    for key, tiff_path in all_tiff.items():
        filename = os.path.basename(tiff_path).replace('.tiff', '.nc')
        netcdf_path = os.path.join(nc_path, filename)
        gdal.Translate(netcdf_path, tiff_path, format='NetCDF')
        all_netcdf[key] = netcdf_path

    return all_netcdf


def create_file_using_template(template_path: str, destination_path: str, mapping: Dict[str, str]) -> None:
    # read the template and replace placeholders with content
    with open(template_path, "rt") as f:
        # read and replace
        data = f.read()
        for k, v in mapping.items():
            data = data.replace(k, v)

    # write the data to the destination
    with open(destination_path, "wt") as f:
        f.write(data)


def extract_static_driver_variables(static_driver_path: str, destination_path: str) -> None:
    with netCDF4.Dataset(static_driver_path, 'r') as f:
        # determine bounding box
        lat = f.variables['lat']
        lon = f.variables['lon']
        bounding_box = BoundingBox(west=np.min(lon), east=np.max(lon), south=np.min(lat), north=np.max(lat))

        # dump selected variables as geotiffs
        for var_name in ['buildings_2d', 'building_id', 'building_type']:
            data = f.variables[var_name]
            tiff_out_path = os.path.join(destination_path, f'{var_name}.tiff')
            height = data.shape[0]
            width = data.shape[1]
            export_as_tiff_4326(data, bounding_box, height, width, tiff_out_path, data.dtype, flip_up=True)


def create_static_driver(parameters: Parameters, input_path: str, output_path: str,
                         template_path: str) -> Tuple[Tuple[float, float], BuildingMapping]:
    # create visual validation and netcdf folders
    vv_path = os.path.join(output_path, 'tiff-contents')
    nc_path = os.path.join(output_path, 'netcdf-contents')
    sd_path = os.path.join(output_path, 'static-driver-contents')
    for path in [vv_path, nc_path, sd_path]:
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    # determine boundingx box and shape of area of interest
    bbox = parameters.bbox
    shape = (parameters.grid_dim[1], parameters.grid_dim[0])  # (dy, dx)

    # rasterise land-cover data
    lc_path = os.path.join(input_path, 'landcover')
    vv_paths_lc = {
        'vegetation': os.path.join(vv_path, 'lulc_vegetation.tiff'),
        'patch_height': os.path.join(vv_path, 'lulc_patch_height.tiff'),
        'water': os.path.join(vv_path, 'lulc_water.tiff'),
        'pavement': os.path.join(vv_path, 'lulc_pavement.tiff'),
        'empty_int': os.path.join(vv_path, 'lulc_empty_int.tiff'),
        'empty_float': os.path.join(vv_path, 'lulc_empty_float.tiff'),
        'zt': os.path.join(vv_path, 'lulc_zt.tiff')
    }
    rasterise_landcover(lc_path, bbox, shape, vv_paths_lc)

    # rasterise building data -> height, id, type
    bld_path = os.path.join(input_path, 'buildings')
    vv_paths_bld = {
        'bld_heights': os.path.join(vv_path, 'buildings_heights.tiff'),
        'bld_ids': os.path.join(vv_path, 'buildings_ids.tiff'),
        'bld_types': os.path.join(vv_path, 'buildings_types.tiff')
    }
    building_id_mapping = rasterise_buildings(bld_path, bbox, shape, vv_paths_bld)

    # rasterise vegetation (tree) data
    veg_path = os.path.join(input_path, 'vegetation')
    vv_paths_veg = {
        'trees_type': os.path.join(vv_path, 'lulc_trees_type.tiff')
    }
    rasterise_vegetation(veg_path, bbox, shape, vv_paths_veg)

    # generate lon/lat in deg/meters tiffs
    vv_paths_lonlat = {
        'lat_deg': os.path.join(vv_path, 'coords_lat_deg.tiff'),
        'lon_deg': os.path.join(vv_path, 'coords_lon_deg.tiff'),
        'lat_meters': os.path.join(vv_path, 'coords_lat_meters.tiff'),
        'lon_meters': os.path.join(vv_path, 'coords_lon_meters.tiff'),
    }
    origin = create_coordinate_tiffs(vv_paths_lonlat, shape, bbox)

    # convert tiffs into netcdf files
    all_tiff: Dict[str, str] = {}
    all_tiff.update(vv_paths_lc)
    all_tiff.update(vv_paths_bld)
    all_tiff.update(vv_paths_veg)
    all_tiff.update(vv_paths_lonlat)
    all_netcdf: Dict[str, str] = create_netcdf_files(all_tiff, nc_path)

    # create the palm config file
    config_path = os.path.join(output_path, 'palm_csd.config')
    mapping = {
        '###OUTPUT_PATH###': output_path,
        '###OUTPUT_FILE_OUT###': 'static_driver.nc',
        '###DOMAIN_ROOT_NX###': '383',
        '###DOMAIN_ROOT_NY###': '383',
        '###INPUT_FILE_PATH###': nc_path,
        '###INPUT_FILE_X_UTM###': os.path.basename(all_netcdf['lon_meters']),
        '###INPUT_FILE_Y_UTM###': os.path.basename(all_netcdf['lat_meters']),
        '###INPUT_FILE_LON_DEG###': os.path.basename(all_netcdf['lon_deg']),
        '###INPUT_FILE_LAT_DEG###': os.path.basename(all_netcdf['lat_deg']),
        '###INPUT_FILE_ZT###': os.path.basename(all_netcdf['zt']),
        '###INPUT_FILE_BLD_HEIGHT###': os.path.basename(all_netcdf['bld_heights']),
        '###INPUT_FILE_BLD_ID###': os.path.basename(all_netcdf['bld_ids']),
        '###INPUT_FILE_BLD_TYPE###': os.path.basename(all_netcdf['bld_types']),
        '###INPUT_FILE_VEGETATION###': os.path.basename(all_netcdf['vegetation']),
        '###INPUT_FILE_PAVEMENT###': os.path.basename(all_netcdf['pavement']),
        '###INPUT_FILE_WATER###': os.path.basename(all_netcdf['water']),
        '###INPUT_FILE_TREES_TYPE###': os.path.basename(all_netcdf['trees_type']),
        '###INPUT_FILE_PATCH_HEIGHT###': os.path.basename(all_netcdf['patch_height']),
        '###INPUT_FILE_EMPTY_FLOAT###': os.path.basename(all_netcdf['empty_float'])
    }
    config_template_path = os.path.join(template_path, 'palm_csd.config.template')
    create_file_using_template(config_template_path, config_path, mapping)

    # create the static driver
    config_path = os.path.join(output_path, 'palm_csd.config')
    create_driver(config_path)

    # extract various variables from the static driver
    static_driver_path = os.path.join(output_path, 'static_driver.nc_root')
    extract_static_driver_variables(static_driver_path, sd_path)

    return origin, building_id_mapping


def generate_ah_file(output_path: str, building_mapping: BuildingMapping,
                     origin: Tuple[float, float], origin_date_time: str) -> None:

    n_buildings = building_mapping.number_of_buildings()
    n_others = building_mapping.number_of_others()

    with netCDF4.Dataset(output_path, 'w', format='NETCDF4') as dataset:
        # Optional global attributes
        # --------------------------
        dataset.title = 'Example PALM ah driver'
        dataset.author = 'PALM user'
        dataset.institution = 'Institution'
        dataset.comment = 'Testing setup'
        dataset.creation_date = datetime.datetime.utcnow().strftime('%y-%m-%d %H:%M:%S %z')
        dataset.history = ''
        dataset.keywords = 'example, PALM-4U'
        dataset.license = ''
        dataset.palm_version = ''
        dataset.references = ''
        dataset.source = ''
        dataset.version = '1'

        # Mandatory global attributes
        # ---------------------------
        dataset.Conventions = 'CF-1.7'
        dataset.origin_lon = origin[0]  # Used to initialize Coriolis parameter
        dataset.origin_lat = origin[1]  # (overwrite initialization_parameters)
        dataset.origin_time = origin_date_time
        dataset.origin_x = 0
        dataset.origin_y = 0
        dataset.origin_z = 0.0
        dataset.rotation_angle = 0.0

        # Define coordinates
        # ------------------
        dataset.createDimension('building_id', n_buildings)
        building_id = dataset.createVariable('building_id', 'i4', ('building_id',))
        building_id.long_name = 'id of buildings that emit anthropogenic heat'
        building_id.units = '1'
        building_id[:] = building_mapping.building_id_array()  # np.arange(1, n_buildings+1, 1)

        dataset.createDimension('point_id', n_others)
        point_id = dataset.createVariable('point_id', 'i4', ('point_id',))
        point_id.long_name = 'id of points that emit anthropogenic heat'
        point_id.units = '1'
        point_id[:] = np.arange(1, n_others+1, 1)

        dataset.createDimension('time', 24)
        time = dataset.createVariable('time', 'f4', ('time',))
        time.long_name = 'time'
        time.standard_name = 'time'
        time.units = 'seconds'
        time.axis = 'T'
        time[:] = np.arange(0, 24, 1) * 3600.0

        # Define variables
        # -----------------
        nc_building_ah = dataset.createVariable('building_ah', 'f8', ('time', 'building_id'), fill_value=-9999)
        nc_building_ah.long_name = "anthropogenic heat emission from building roof"
        nc_building_ah.units = "kWh"
        nc_building_ah[:, :] = building_mapping.building_ah_array()

        nc_point_ah = dataset.createVariable('point_ah', 'f8', ('time', 'point_id'), fill_value=-9999)
        nc_point_ah.long_name = "anthropogenic heat emission from a specific ground tile"
        nc_point_ah.units = "kWh"
        nc_point_ah[:, :] = building_mapping.others_ah_array()

        nc_point_x = dataset.createVariable('point_x', 'f8', ('point_id',), fill_value=-9999)
        nc_point_x.long_name = "x coordinate from point source (UTM coordinates)"
        nc_point_x.units = "m"
        nc_point_x[:] = building_mapping.others_x_location()

        nc_point_y = dataset.createVariable('point_y', 'f8', ('point_id',), fill_value=-9999)
        nc_point_y.long_name = "y coordinate from point source (UTM coordinates)"
        nc_point_y.units = "m"
        nc_point_y[:] = building_mapping.others_y_location()


def create_vv_package(wd_path: str) -> None:
    # create the package
    package_path = os.path.join(wd_path, 'vv-package')
    logger.info(f"create vv-package at {package_path}")

    result = subprocess.run(['tar', 'czf', package_path, 'tiff-contents', 'netcdf-contents', 'static-driver-contents',
                             'palm_csd.config'],
                            cwd=wd_path, capture_output=True)
    if result.returncode != 0:
        raise ProcessorRuntimeError("Creating vv-package failed", details={
            'stdout': result.stdout.decode('utf-8'),
            'stderr': result.stderr.decode('utf-8')
        })


def create_run_package(wd_path: str, templates_path: str, parameters: Parameters, origin_date_time: str,
                       dt_warmup: float = 7200.0) -> None:
    # create the contents folder
    package_contents_path = os.path.join(wd_path, 'run-package-contents')
    if os.path.exists(package_contents_path):
        shutil.rmtree(package_contents_path)
    os.makedirs(package_contents_path)

    # extract template files into contents folder
    shutil.copy(os.path.join(templates_path, 'package_contents.tar.gz'),
                os.path.join(package_contents_path, 'package_contents.tar.gz'))
    result = subprocess.run(['tar', 'xzf', 'package_contents.tar.gz'], cwd=package_contents_path, capture_output=True)
    if result.returncode != 0:
        raise ProcessorRuntimeError("Extracting package_contents.tar.gz failed", details={
            'stdout': result.stdout.decode('utf-8'),
            'stderr': result.stderr.decode('utf-8')
        })

    # create p3d and p3dr files
    create_file_using_template(os.path.join(templates_path, 'p3d.template'),
                               os.path.join(package_contents_path, 'NAME_p3d'),
                               {
                                   '###INIT_ACTIONS###': 'read_from_file',
                                   '###SOUTH_WEST_LON###': str(parameters.bbox.west),
                                   '###SOUTH_WEST_LAT###': str(parameters.bbox.south),
                                   '###ORIGIN_DATE_TIME###': origin_date_time,
                                   '###END_TIME###': f'{(parameters.dt_sim + dt_warmup):.1f}',
                                   '###AH_FLAG###': '' if parameters.ah_profile else '! '
                               })
    create_file_using_template(os.path.join(templates_path, 'p3d.template'),
                               os.path.join(package_contents_path, 'NAME_p3dr'),
                               {
                                   '###INIT_ACTIONS###': 'read_restart_data',
                                   '###SOUTH_WEST_LON###': str(parameters.bbox.west),
                                   '###SOUTH_WEST_LAT###': str(parameters.bbox.south),
                                   '###ORIGIN_DATE_TIME###': origin_date_time,
                                   '###END_TIME###': f'{(parameters.dt_sim + dt_warmup):.1f}',
                                   '###AH_FLAG###': '' if parameters.ah_profile else '! '
                               })

    # move static driver
    static_driver_path = os.path.join(wd_path, 'static_driver.nc_root')
    os.rename(static_driver_path, os.path.join(package_contents_path, 'NAME_static'))

    # move dynamic driver
    os.rename(os.path.join(package_contents_path, f'NAME_dynamic.{parameters.dd_profile}'),
              os.path.join(package_contents_path, 'NAME_dynamic'))

    # move ah file
    filenames = ['NAME_dynamic', 'NAME_static', 'NAME_p3d', 'NAME_p3dr', 'NAME_rlw', 'NAME_rsw']
    if parameters.ah_profile:
        ah_file_path = os.path.join(wd_path, 'ah.nc')
        os.rename(ah_file_path, os.path.join(package_contents_path, 'NAME_ah'))
        filenames.append('NAME_ah')

    # create the package
    package_path = os.path.join(wd_path, 'palm-run-package')
    logger.info(f"create palm-run-package at {package_path}")
    result = subprocess.run(['tar', 'czf', package_path, *filenames],
                            cwd=package_contents_path, capture_output=True)
    if result.returncode != 0:
        raise ProcessorRuntimeError("Creating palm-run-package failed", details={
            'stdout': result.stdout.decode('utf-8'),
            'stderr': result.stderr.decode('utf-8')
        })

    # delete run package contents folder
    shutil.rmtree(package_contents_path)


class PalmPrepProcessor(ProcessorBase):
    def __init__(self, proc_path: str) -> None:
        super().__init__(proc_path)

        self._is_cancelled = False

    def run(self, wd_path: str, callback: ProgressListener, logger: logging.Logger) -> None:
        def check_if_cancelled_and_pub_progress(progress: int) -> None:
            callback.on_progress_update(progress)
            if self._is_cancelled:
                raise ProcessorRuntimeError(f"cancelled -> exiting now.")

        check_if_cancelled_and_pub_progress(1)

        # load parameters
        callback.on_message(Severity.INFO, 'Loading parameters.')
        parameters_path = os.path.join(wd_path, "parameters")
        parameters = load_parameters(parameters_path)
        check_if_cancelled_and_pub_progress(2)

        # create the static driver
        callback.on_message(Severity.INFO, 'Creating static driver...')
        templates_path = os.path.join('..', 'templates')
        origin, building_id_mapping = create_static_driver(parameters, wd_path, wd_path, templates_path)
        check_if_cancelled_and_pub_progress(30)

        origin_date_time = '2020-07-01 22:00:00 +08'

        # create the AH file
        if parameters.ah_profile:
            callback.on_message(Severity.INFO, 'Generating AH file...')
            ah_output_path = os.path.join(wd_path, 'ah.nc')
            generate_ah_file(ah_output_path, building_id_mapping, origin, origin_date_time)
        else:
            callback.on_message(Severity.INFO, 'Skipping generation of AH file.')
        check_if_cancelled_and_pub_progress(50)

        # create the palm-run-package
        callback.on_message(Severity.INFO, 'Creating palm run package...')
        create_run_package(wd_path, templates_path, parameters, origin_date_time)
        callback.on_output_available('palm-run-package')
        check_if_cancelled_and_pub_progress(80)

        # create vv-package
        callback.on_message(Severity.INFO, 'Creating visual validation package...')
        create_vv_package(wd_path)
        callback.on_output_available('vv-package')
        callback.on_progress_update(100)

        # indicate we are done
        callback.on_message(Severity.INFO, 'Done (palm-prep).')

    def cancel(self) -> None:
        self._is_cancelled = True


if __name__ == '__main__':
    # create the logger
    Logging.initialise(logging.INFO)
    logger = Logging.get(__name__)

    # get the proc path
    proc_path = os.getcwd()
    logger.info(f"Assuming processor located at '{proc_path}'")

    # get the working directory path
    wd_path = sys.argv[1]
    if not os.path.isdir(wd_path):
        logger.error(f"Working directory at '{wd_path}' does not exist!")
        sys.exit(-1)
    logger.info(f"Using working directory located at '{wd_path}'")

    class LegacyListener(ProgressListener):
        def on_progress_update(self, progress: float) -> None:
            print(f"trigger:progress:{int(progress)}")
            sys.stdout.flush()

        def on_output_available(self, output_name: str) -> None:
            print(f"trigger:output:{output_name}")
            sys.stdout.flush()

        def on_message(self, severity: Severity, message: str) -> None:
            print(f"trigger:message:{severity.value}:{message}")
            sys.stdout.flush()

    # create the processor object and run the
    callback = LegacyListener()
    try:
        logger.info(f"Attempting to run the Palm4U preprocessor...'")
        proc = PalmPrepProcessor(proc_path)
        proc.run(wd_path, callback, logger)
        logger.info("Done!")
        sys.exit(0)

    except ProcessorRuntimeError as e:
        logger.error(f"Exception {e.id}: {e.reason}\ndetails: {e.details}")
        callback.on_message(Severity.ERROR, e.reason)

        sys.exit(-2)
