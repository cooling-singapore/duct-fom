import abc
import bisect
import datetime
import glob
import itertools
import json
import logging
import shutil
import string
import threading
import sys
import os
import time
import subprocess
from contextlib import contextmanager

import rasterio
import numpy as np
from enum import Enum
from threading import Lock
from typing import Optional, List, Union

from pydantic import BaseModel, Field
from rasterio import CRS
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from saas.core.helpers import generate_random_string
from saas.core.logging import Logging
from saas.dor.schemas import ProcessorDescriptor

logger = logging.getLogger('ucm-wrf-prep')


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

class Parameters(BaseModel):
    class DateTime(BaseModel):
        date: str
        time: str

    class Settings(BaseModel):
        pbs_project_id: Optional[str]
        pbs_queue: Optional[str]

    name: str
    t_from: DateTime
    t_to: DateTime
    frc_urb1: float
    frc_urb2: float
    frc_urb3: float
    frc_urb4: float
    frc_urb5: float
    frc_urb6: float
    frc_urb7: float
    frc_urb8: float
    frc_urb9: float
    frc_urb10: float
    settings: Optional[Settings]


class BoundingBox(BaseModel):
    north: float
    east: float
    south: float
    west: float


# define some defaults
default_height_limit = 300  # [m] -> 300m is the height limit
default_p_top_requested = 5000
default_z_levels = [
    *list(range(0, 140, 20)),
    *list(range(140, 200, 30)),
    *list(range(200, 400, 100)),
    *list(range(400, 1000, 200)),
    *list(range(1000, 6000, 500)),
    *list(range(6000, 19000, 1000))
]


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


def rm_mk_dir(dir_path: str) -> None:
    if os.path.isdir(dir_path):
        logger.debug(f"directory at {dir_path} already exists -> deleting...")
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


def mk_symlink(target_path: str, link_path: str) -> None:
    logger.debug(f"create symbolic link: {target_path} -> {link_path}")
    if os.path.exists(link_path):
        os.remove(link_path)
    os.symlink(target_path, link_path)


def create_file_using_template(template_path: str, destination_path: str, mapping: dict) -> None:
    # read the template and replace placeholders with content
    with open(template_path, "rt") as f:
        # read and replace
        data = f.read()
        for k, v in mapping.items():
            data = data.replace(str(k), str(v))

        # write the data to the destination
        with open(destination_path, "wt") as f:
            f.write(data)


def export_as_tiff(data: np.ndarray, tiff_out_path: str, bbox: BoundingBox, flip_vertically: bool, dtype) -> None:
    if flip_vertically:
        data = np.copy(data)
        data = np.flipud(data)

    with rasterio.open(tiff_out_path, 'w+', driver='GTiff', width=data.shape[1], height=data.shape[0],
                       count=1, dtype=dtype, crs=CRS().from_string("EPSG:4326"),
                       transform=rasterio.transform.from_bounds(bbox.west, bbox.south, bbox.east, bbox.north,
                                                                data.shape[1], data.shape[0])
                       ) as dataset:
        dataset.write(data, 1)


def reproject_map(input_path: str, resampling: Resampling, no_data: int, width: int, height: int,
                  bbox: BoundingBox) -> np.ndarray:

    with rasterio.Env():
        with rasterio.open(input_path) as src:
            # calculate transform
            crs = CRS().from_string("EPSG:4326")
            dst_transform, dst_width, dst_height = calculate_default_transform(
                crs, crs, width, height, dst_width=width, dst_height=height,
                top=bbox.north, bottom=bbox.south, left=bbox.west, right=bbox.east
            )

            # set properties for output
            dst_kwargs = src.meta.copy()
            dst_kwargs.update(
                {"crs": crs, "transform": dst_transform, "width": dst_width, "height": dst_height, "nodata": no_data}
            )

            # determine shape and bounding box
            dst_shape = (dst_height, dst_width)

            # re-project source to destination
            source = src.read(1)
            dst = np.zeros(dst_shape)
            reproject(
                source=source, destination=dst,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=dst_transform, dst_crs=crs, dst_nodata=no_data,
                resampling=resampling
            )

            return dst


def inject_geodata(wd_path: str, vv_dir_path: str, n_domains: int, nx: int, ny: int, nz: int, cell_area: int,
                   height_levels: list, lon_precision: int = 5, lat_precision: int = 7) -> None:

    def determine_height_levels_and_weights(height: int, height_levels: list) -> (int, int, float, float):
        # determine the two height levels k0, k1 that are closest to the height of interest
        k0 = -1
        k1 = -1
        for i in range(len(height_levels) - 1):
            k0 = i
            k1 = i + 1
            if height_levels[k0] <= height < height_levels[k1]:
                break

        # determine the difference in height dh (in meters) between the two levels k0, k1
        dh = height_levels[k1] - height_levels[k0]  # [m]

        # determine the weights w0, w1 based on how close the height is to the two height levels
        w0 = 1.0 - (height - height_levels[k0]) / dh
        w1 = 1.0 - (height_levels[k1] - height) / dh

        return k0, k1, w0, w1

    def prepare_ah(ah_input_name: str, heights: list) -> dict:
        # make the folder for the archive contents
        ah_folder_path = os.path.join(wd_path, f"{ah_input_name}.unpacked")
        shutil.rmtree(ah_folder_path, ignore_errors=True)
        os.mkdir(ah_folder_path)
        logger.info(f"[INJECT] unpacking AH profiles of '{ah_input_name}' to '{ah_folder_path}'.")

        # copy the archive
        input_path0 = os.path.join(wd_path, ah_input_name)
        input_path1 = os.path.join(ah_folder_path, ah_input_name)
        shutil.copy(input_path0, input_path1)

        # unpack the archive
        result = subprocess.run(['tar', 'xzf', ah_input_name], cwd=ah_folder_path)
        if result.returncode != 0:
            raise ProcessorRuntimeError(f"Unpacking AH archive {ah_input_name} failed", details={
                'stdout': result.stdout.decode('utf-8'),
                'stderr': result.stderr.decode('utf-8')
            })

        # remove the archive
        os.remove(input_path1)

        # make a list of files
        ah_files = []
        for filename in os.listdir(ah_folder_path):
            # ignore certain files
            if filename.startswith('.'):
                logger.warning(f"[INJECT] ignoring file '{filename}' in AH archive '{ah_input_name}'")
                continue

            # is it a file?
            file_path = os.path.join(ah_folder_path, filename)
            if os.path.isfile(file_path):
                ah_files.append(file_path)

        logger.debug(f"[INJECT] found {len(ah_files)} AH files: {ah_files}")

        # parse each AH file and sort features by height level
        result = {}
        for input_path in ah_files:
            logger.info(f"[INJECT] parsing {input_path}")

            # read geojson input file
            with open(input_path, 'r') as f:
                collection = json.load(f)

            # process the geojson content
            for feature in collection['features']:
                # determine the closest two height levels (k0, k1) and determine corresponding weights (w0, w1)
                height = int(feature['properties']['height:m'])
                k0, k1, w0, w1 = determine_height_levels_and_weights(height, heights)

                # fix whitespaces in AH properties
                for key in list(feature['properties'].keys()):
                    if key != 'AH_type' and key.startswith('AH_') and ' ' in key:
                        fixed_key = key.replace(' ', '')
                        feature['properties'][fixed_key] = feature['properties'].pop(key)

                # determine AH values for each hour of the day in Watts (convert accordingly) and scale by w0/w1
                for it in range(24):
                    if f"AH_{it}:W" in feature['properties']:
                        ah = feature['properties'][f"AH_{it}:W"]
                    elif f"AH_{it}:KW" in feature['properties']:
                        ah = feature['properties'][f"AH_{it}:KW"] * 1e3
                    elif f"AH_{it}:MW" in feature['properties']:
                        ah = feature['properties'][f"AH_{it}:MW"] * 1e6
                    else:
                        raise ProcessorRuntimeError(f"Encountered unsupported/missing AH data: {feature['properties']}")

                    ah0 = ah * w0
                    ah1 = ah * w1
                    feature['properties'][f"AH_{k0}:{it}"] = ah0
                    feature['properties'][f"AH_{k1}:{it}"] = ah1

                # add to the k0 collection
                if k0 not in result:
                    result[k0] = {
                        "type": "FeatureCollection",
                        "features": [feature]
                    }
                else:
                    result[k0]['features'].append(feature)

                # add to the k1 collection
                if k1 not in result:
                    result[k1] = {
                        "type": "FeatureCollection",
                        "features": [feature]
                    }
                else:
                    result[k1]['features'].append(feature)

        return result

    def grid_overlap(feature: dict, bbox: BoundingBox, d_lat: float, d_lon: float) -> list:
        if feature['geometry']['type'] == 'LineString':
            p0 = feature['geometry']['coordinates'][0]
            p1 = feature['geometry']['coordinates'][1]

            y0, x0 = int((bbox.north - p0[1]) / d_lat), int((p0[0] - bbox.west) / d_lon)
            y1, x1 = int((bbox.north - p1[1]) / d_lat), int((p1[0] - bbox.west) / d_lon)

            # how many steps?
            ay = abs(y1 - y0)
            ax = abs(x1 - x0)
            n = max(ay, ax)

            result = []
            if n == 0:
                result.append((y0, x0))

            elif n == 1:
                result.append((y0, x0))
                result.append((y1, x1))

            else:
                # determine lon/lat steps size
                dlon = (p1[0] - p0[0]) / n
                dlat = (p1[1] - p0[1]) / n
                for i in range(0, n+1, 1):
                    lat = p0[1] + i * dlat
                    lon = p0[0] + i * dlon

                    y = int((bbox.north - lat) / d_lat)
                    x = int((lon - bbox.west) / d_lon)

                    result.append((y, x))

            return result

        elif feature['geometry']['type'] == 'Point':
            p = feature['geometry']['coordinates']
            y, x = int((bbox.north - p[1]) / d_lat), int((p[0] - bbox.west) / d_lon)
            return [(y, x)]

        else:
            raise ProcessorRuntimeError(f"Unsupported geometry type: '{feature['geometry']['type']}'")

    def rasterise_ah(prepared_ah_data: dict, bbox: BoundingBox) -> dict:
        d_lat = (bbox.north - bbox.south) / ny
        d_lon = (bbox.east - bbox.west) / nx

        with rasterio.Env():
            result = {}
            for k, collection in prepared_ah_data.items():
                stack = []
                for t in range(24):
                    key = f"{k}:{t}"
                    raster = np.zeros(shape=(ny, nx), dtype=np.float32)

                    for f in collection['features']:
                        items = grid_overlap(f, bbox, d_lat, d_lon)
                        ah = f['properties'][f"AH_{key}"] / len(items)
                        if ah > 0:
                            for i in items:
                                y = i[0]
                                x = i[1]

                                if 0 <= y < ny and 0 <= x < nx:
                                    raster[y][x] += ah

                    # convert [W] -> [W/m2]
                    raster /= cell_area

                    # flip vertically and stack
                    raster = np.flipud(raster)
                    stack.append(raster)

                result[k] = np.stack(stack)

            return result

    # make vv geogrid path
    vv_geogrid_path = os.path.join(vv_dir_path, 'geogrid')
    rm_mk_dir(vv_geogrid_path)

    for d in range(1, n_domains+1):
        geo_em_path = os.path.join(wd_path, f"geo_em.d{d:02d}.nc")
        with open_netcdf(geo_em_path, 'a') as nc_file:
            # source: https://www2.mmm.ucar.edu/wrf/users/docs/user_guide_v4/v4.2/users_guide_chap3.html
            # extract the bounding box using 'corner_lats' and 'corner_lons"
            lat = nc_file.__getattribute__('corner_lats')
            lon = nc_file.__getattribute__('corner_lons')
            bbox = BoundingBox(
                west=round(float(lon[14 - 1]), lon_precision),
                north=round(float(lat[14 - 1]), lat_precision),
                east=round(float(lon[16 - 1]), lon_precision),
                south=round(float(lat[16 - 1]), lat_precision)
            )

            # extract the LU/GF variables
            lu = nc_file.variables['LU_INDEX']
            gf = nc_file.variables['GREENFRAC']
            sh_ext = nc_file.variables['SH_EXT']
            lh_ext = nc_file.variables['LH_EXT']
            logger.info(f"[INJECT] [D{d:02d}] shapes: LU={lu.shape} GF={gf.shape} "
                        f"SH_EXT={sh_ext.shape} LH_EXT={lh_ext.shape}")

            # reference shapes:
            # 2023-08-25 11:03:58.423 [INFO] [__main__] [inject_geodata] [D04] LU shape=(1, 129, 210)
            # 2023-08-25 11:03:58.423 [INFO] [__main__] [inject_geodata] [D04] GF shape=(1, 12, 129, 210)
            # 2023-08-26 11:57:14.826 [INFO] [__main__] [inject_geodata] [D04] SH_EXT shape=(1, 264, 129, 210)
            # 2023-08-26 11:57:14.826 [INFO] [__main__] [inject_geodata] [D04] LH_EXT shape=(1, 264, 129, 210)

            # store the original LU/GF maps
            export_as_tiff(lu[0], os.path.join(vv_geogrid_path, f"original_LU.d{d:02d}.tiff"), bbox, True, np.uint8)
            export_as_tiff(gf[0][0], os.path.join(vv_geogrid_path, f"original_GF.d{d:02d}.tiff"), bbox, True, np.float32)

            # is this the domain of interest?
            if d == n_domains:
                height = lu[0].shape[0]
                width = lu[0].shape[1]

                # reproject the LCZ map to match the original LU map
                lcz_path = os.path.join(wd_path, 'lcz-map')
                lcz_vv_path = os.path.join(vv_geogrid_path, 'reprojected-lcz.tiff')
                lcz = reproject_map(lcz_path, Resampling.nearest, 0, width, height, bbox)
                export_as_tiff(lcz, lcz_vv_path, bbox, False, np.uint8)

                # reproject the VF map to match the original GF map
                vegfra_path = os.path.join(wd_path, 'vegfra-map')
                vegfra_vv_path = os.path.join(vv_geogrid_path, 'reprojected-vegfra.tiff')
                vegfra = reproject_map(vegfra_path, Resampling.average, -1, width, height, bbox)
                export_as_tiff(vegfra, vegfra_vv_path, bbox, False, np.float32)

                # merge original LU_INDEX into LCZ to fill missing values in LCZ (if any)
                lcz = np.flipud(lcz)
                idx = (lcz == 0)
                lcz[idx] = lu[0][idx]
                export_as_tiff(lcz, os.path.join(vv_geogrid_path, 'repaired-lcz.tiff'), bbox, True, np.uint8)

                # merge original GREENFRAC into VEGFRA to fill missing values in VEGFRA (if any)
                vegfra = np.flipud(vegfra)
                idx = (vegfra < 0)
                vegfra[idx] = gf[0][0][idx]
                export_as_tiff(vegfra, os.path.join(vv_geogrid_path, 'repaired-vegfra.tiff'), bbox, True, np.float32)

                # prepare and rasterise AH data
                prepared_sh = prepare_ah('sh-profile', height_levels)
                prepared_lh = prepare_ah('lh-profile', height_levels)
                sh = rasterise_ah(prepared_sh, bbox)
                lh = rasterise_ah(prepared_lh, bbox)

                # inject repaired LCZ into LU
                lu[0, :, ] = lcz

                # inject repaired VEGFRA into GF
                for m in range(12):
                    gf[0, m, :, ] = vegfra

                # inject SH_EXT
                for k, profile in sh.items():
                    for it in range(24):
                        iz = k * 24 + it
                        sh_ext[0, iz, :] = profile[it]

                # inject LH_EXT
                for k, profile in lh.items():
                    for it in range(24):
                        iz = k * 24 + it
                        lh_ext[0, iz, :] = profile[it]

        if d == n_domains:
            # print the post-injection vv GeoTIFFs
            with open_netcdf(geo_em_path, 'r') as nc_file:
                # extract the bounding box using 'corner_lats' and 'corner_lons"
                lat = nc_file.__getattribute__('corner_lats')
                lon = nc_file.__getattribute__('corner_lons')
                bbox = BoundingBox(
                    west=round(float(lon[14 - 1]), lon_precision),
                    north=round(float(lat[14 - 1]), lat_precision),
                    east=round(float(lon[16 - 1]), lon_precision),
                    south=round(float(lat[16 - 1]), lat_precision)
                )

                # extract the LU/GF variables
                lu = nc_file.variables['LU_INDEX']
                gf = nc_file.variables['GREENFRAC']
                export_as_tiff(lu[0], os.path.join(vv_geogrid_path, f"injected_LU.d{d:02d}.tiff"), bbox, True, np.uint8)
                export_as_tiff(gf[0][0], os.path.join(vv_geogrid_path, f"injected_GF.d{d:02d}.tiff"), bbox, True, np.float32)

                # export SH_EXT and LH_EXT for all layer/time that have non-zero values
                sh_ext = nc_file.variables['SH_EXT']
                lh_ext = nc_file.variables['LH_EXT']
                for k in range(0, nz):
                    for t in range(0, 24):
                        iz = k*24 + t

                        sh_sum = np.sum(sh_ext[0][iz])
                        if sh_sum > 0:
                            filename = f"injected_SH_EXT_d{d:02d}_k{k:02d}_t{t:02d}.tiff"
                            export_as_tiff(sh_ext[0][iz], os.path.join(vv_geogrid_path, filename), bbox, True, np.uint16)

                        lh_sum = np.sum(lh_ext[0][iz])
                        if lh_sum > 0:
                            filename = f"injected_LH_EXT_d{d:02d}_k{k:02d}_t{t:02d}.tiff"
                            export_as_tiff(lh_ext[0][iz], os.path.join(vv_geogrid_path, filename), bbox, True, np.uint16)


def eta_calc(p_top_requested: int, z_levs: np.ndarray) -> (list, list):
    # WRF Base State Atmosphere Variables
    p00 = 100000       # Surface pressure, Pa
    t00 = 301          # Sea level temperature, K
    t0  = 300          # Base state potential temperature, K
    a   = 50           # Base state lapse rate 1000 - 300 hPa, K
    mub = p00 - p_top_requested

    # Constants
    r_d  = 287         # Dry gas constant
    cp   = 1004        # Specific heat of dry air at const P.
    cvpm = -717/1004   # -cv/cp
    g    = 9.81        # gravity

    # WRF Standard Eta Levels
    znw = np.array([1.000000, 0.993000, 0.983000, 0.970000, 0.954000, 0.934000, 0.909000,
                    0.880000, 0.844022, 0.808045, 0.772067, 0.736089, 0.671281, 0.610883,
                    0.554646, 0.502330, 0.453708, 0.408567, 0.366699, 0.327910, 0.292015,
                    0.258837, 0.228210, 0.199974, 0.173979, 0.150082, 0.128148, 0.108049,
                    0.089663, 0.072875, 0.057576, 0.043663, 0.031039, 0.019611, 0.009292,
                    0.000000])

    # znu       is the eta value on the mass grid points
    # dnw       is the change in eta between eta levels
    # p         is the base state pressure on the mass grid points
    # t         is the base state temperature on the mass grid points
    # t_init    is the base state potential temperature perturbation on the mass grid points
    # alb       is the inverse base state density on the mass grid points

    znu = np.zeros(shape=(znw.shape[0]-1))
    dnw = np.zeros(shape=(znw.shape[0]-1))
    p = np.zeros(shape=(znw.shape[0]-1))
    t = np.zeros(shape=(znw.shape[0]-1))
    t_init = np.zeros(shape=(znw.shape[0]-1))
    alb = np.zeros(shape=(znw.shape[0]-1))
    for k in range(znw.shape[0] - 1):
        znu[k] = (znw[k] + znw[k+1]) * 0.5
        dnw[k] = znw[k+1] - znw[k]
        p[k] = znu[k] * mub + p_top_requested
        t[k] = t00 + a * np.log(p[k] / p00)
        t_init[k] = t[k] * ((p00 / p[k]) ** (r_d / cp)) - t0
        alb[k] = (r_d / p00) * (t_init[k] + t0) * ((p[k] / p00) ** cvpm)

    # print(f"znu: {znu}")
    # print(f"dnw: {dnw}")
    # print(f"p: {p}")
    # print(f"t: {t}")
    # print(f"t_init: {t_init}")
    # print(f"alb: {alb}")

    # This loops solves the hydrostatic equation using base state
    # geopotential to find the model top geopotential.

    phb = np.zeros(shape=(znw.shape[0]))
    for k in range(1, len(znw)):
        phb[k] = phb[k - 1] - dnw[k - 1] * mub * alb[k - 1]
    # print(f"phb: {phb}")

    # Solve for model top on the w grid points in z coords, m.
    # not used?? --> ztop = phb(length(phb))./g

    eta = np.zeros(shape=(z_levs.shape[0]))
    eta[0] = 1.000

    # dz         is thickness between z levels
    # pw         is the base state pressure on the w grid points
    # tw         is the base state temperature on the w grid points
    # t_initw    is the base state potential temperature perturbation on the w grid points
    # albw       is the inverse base state density on the w grid points

    dz = np.zeros(shape=(z_levs.shape[0]-1))
    pw = np.zeros(shape=(z_levs.shape[0]-1))
    tw = np.zeros(shape=(z_levs.shape[0]-1))
    t_initw = np.zeros(shape=(z_levs.shape[0]-1))
    albw = np.zeros(shape=(z_levs.shape[0]-1))

    for k in range(z_levs.shape[0] - 1):
        dz[k] = z_levs[k+1] - z_levs[k]
        pw[k] = eta[k] * mub + p_top_requested
        tw[k] = t00 + a * np.log(pw[k] / p00)
        t_initw[k] = tw[k] * ((p00 / pw[k]) ** (r_d / cp)) - t0
        albw[k] = (r_d / p00) * (t_initw[k] + t0) * ((pw[k] / p00) ** cvpm)
        eta[k+1] = eta[k] - ((dz[k] * g) / (mub * albw[k]))

    eta[eta.shape[0]-1] = 0.0

    # print(f"dz: {dz}")
    # print(f"pw: {pw}")
    # print(f"tw: {tw}")
    # print(f"t_initw: {t_initw}")
    # print(f"albw: {albw}")
    # print(f"eta: {eta}")

    # Section 3.  Calculate model levels on z coords from our new eta levs as a
    # visual test.  If the distribution looks good use the eta values in the
    # WRF namelist.

    # This loop calculates the base state atmosphere at the calculated
    # eta levels for use in the hydrostatic equation below.

    # znue       is the eta value on the mass grid points
    # dnwe       is the change in eta between eta levels
    # pe         is the base state pressure on the mass grid points
    # te         is the base state temperature on the mass grid points
    # t_inite    is the base state potential temperature perturbation on the mass grid points
    # albe       is the inverse base state density on the mass grid points

    znue = np.zeros(shape=(eta.shape[0]-1))
    dnwe = np.zeros(shape=(eta.shape[0]-1))
    pe = np.zeros(shape=(eta.shape[0]-1))
    te = np.zeros(shape=(eta.shape[0]-1))
    t_inite = np.zeros(shape=(eta.shape[0]-1))
    albe = np.zeros(shape=(eta.shape[0]-1))

    for k in range(eta.shape[0] - 1):
        znue[k] = (eta[k] + eta[k + 1]) * 0.5
        dnwe[k] = eta[k + 1] - eta[k]
        pe[k] = znue[k] * mub + p_top_requested
        te[k] = t00 + a * np.log(pe[k] / p00)
        t_inite[k] = te[k] * ((p00 / pe[k]) ** (r_d / cp)) - t0
        albe[k] = (r_d / p00) * (t_inite[k] + t0) * ((pe[k] / p00) ** cvpm)

    # print(f"znue: {znue}")
    # print(f"dnwe: {dnwe}")
    # print(f"pe: {pe}")
    # print(f"te: {te}")
    # print(f"t_inite: {t_inite}")
    # print(f"albe: {albe}")

    # This loops solves the hydrostatic equation using base state
    # geopotential to find the model to in z coords.

    # phbe is the base state geopotential on the w grid points from the
    # calculated eta values

    phbe = np.zeros(shape=(eta.shape[0]))
    for k in range(1, eta.shape[0]):
        phbe[k] = phbe[k - 1] - dnwe[k - 1] * mub * albe[k - 1]
    # print(f"phbe: {phbe}")

    # Convert geopotential to heights in z, m.
    heights = phbe / g
    # print(f"heights: {heights}")

    return eta.tolist(), heights.tolist()


def generate_eta_levels(wd_path: str, p_top_requested: int, z_levels: List[int],
                        height_limit: int) -> (list, int, list):

    # determine ETA levels and heights (convert between lists and ndarrays for input and output)
    z_levels = np.array(z_levels)
    eta, heights = eta_calc(p_top_requested, z_levels)
    eta: List[float] = list(eta)  # only here to satisfy PyCharm...

    # determine nz, i.e., the number of layers within a given height limit (in meters). basically, identify
    # the number layers that are relevant for the urban area. the highest layer should be at the top of the
    # urban environment. the default height limit is 300m which results in 11 layers. IMPORTANT: this value
    # is hard-coded in WRF. changing it, requires recompiling WRF. eventually, it should be a parameter WRF.
    nz = bisect.bisect([int(h) for h in heights], height_limit)

    # write height levels to file
    height_conversion_path = os.path.join(wd_path, 'height_conversion.txt')
    with open(height_conversion_path, 'w') as f:
        f.write("Index\tETA Level\tHeight (in meters)\n")
        for i in range(len(eta)):
            f.write(f"{i}\t{eta[i]}\t{heights[i]}\n")

    return eta, nz, heights


@contextmanager
def open_netcdf(path: str, mode: str):
    v = os.environ['USE_NETCDF_VERSION']
    if v == '3':
        import scipy
        file = scipy.io.netcdf_file(path, mode, mmap=False)
        try:
            yield file
        finally:
            file.close()

    elif v == '4':
        import netCDF4
        file = netCDF4.Dataset(path, mode)
        try:
            yield file
        finally:
            file.close()

    else:
        raise ProcessorRuntimeError(f"Invalid value for 'USE_NETCDF_VERSION': {v}")


def run_exe(wd_path: str, exe_name: str, env_name: str, label: str, parameters: Parameters,
            ncpus: str = "1", mem: str = "16GB", walltime: str = "01:00:00") -> None:

    # link executable
    exe_link_path = os.path.join(wd_path, exe_name)
    if os.path.isfile(exe_link_path):
        os.remove((exe_link_path))
    os.symlink(os.environ[env_name], exe_link_path)

    # run executable
    if os.environ['USE_PBS'] == 'yes':
        # pbs_execute(wd_path, parameters, label, f"./{exe_name}")
        # check if the template exists
        template_path = os.path.join(os.environ['MODEL_PATH'], f"job.template-{os.environ['TEMPLATE_VERSION']}")
        if not os.path.isfile(template_path):
            raise ProcessorRuntimeError(f"Missing input: job template not found at {template_path}")

        # delete related files (if they exist)
        job_filename = f"job.{parameters.name}.{label}"
        log_filename = f"log.{parameters.name}.{label}"
        exitcode_filename = f"exitcode.{parameters.name}.{label}"
        job_path = os.path.join(wd_path, job_filename)
        log_path = os.path.join(wd_path, log_filename)
        exitcode_path = os.path.join(wd_path, exitcode_filename)
        for path in [job_path, log_path, exitcode_path]:
            if os.path.isfile(path):
                os.remove(path)

        # create the PBS job file
        create_file_using_template(template_path, job_path, {
            "###QUEUE###": parameters.settings.pbs_queue,
            "###NAME###": f"{parameters.name}.{label}",
            "###PROJECT_ID###": parameters.settings.pbs_project_id,
            "###NCPUS###": ncpus,
            "###MEM###": mem,
            "###WALLTIME###": walltime,
            "###LOG_PATH###": log_path,
            "###WD_PATH###": wd_path,
            "###COMMAND###": f"./{exe_name}",
            "###EXITCODE_FILE###": exitcode_filename
        })

        # submit
        logger.info(f"submitting pbs job for {label}")
        result = subprocess.run(['qsub', job_filename], cwd=wd_path, capture_output=True)
        if result.returncode != 0:
            raise ProcessorRuntimeError(f"Submitting command '{label}' failed.", details={
                'stdout': result.stdout.decode('utf-8'),
                'stderr': result.stderr.decode('utf-8')
            })

        # wait until exitcode file is available
        while True:
            if os.path.isfile(exitcode_path):
                logger.info(f"exitcode file found at {exitcode_path}")
                break
            else:
                logger.info(f"waiting for exitcode file...")
                time.sleep(5)

        # read the exit code
        with open(exitcode_path, 'r') as f:
            line = f.readline()
            logger.info(f"exitcode file content: {line}")

            exitcode = int(line)
            if exitcode != 0:
                raise ProcessorRuntimeError(f"Non-zero exitcode when running command {label}")

    else:
        result = subprocess.run([f'./{exe_name}'], cwd=wd_path, capture_output=True)
        if result.returncode != 0:
            raise ProcessorRuntimeError(f"Running {exe_name} failed", details={
                'stdout': result.stdout.decode('utf-8'),
                'stderr': result.stderr.decode('utf-8')
            })


def run_geogrid(wd_path: str, parameters: Parameters, simulate_only: bool = False) -> (int, int, int, int):
    def extract_value(line: str, idx: int = None) -> Union[str, List[str]]:
        line = line.replace('\n', '')
        fields = line.split('=')
        fields = fields[1].split(',')
        return fields if idx is None else fields[idx]

    def generate_ahext_index(ah_code: str) -> str:
        index_folder_path = os.path.join(wd_path, ah_code)
        rm_mk_dir(index_folder_path)

        template_path = os.path.join(os.environ['MODEL_PATH'], 'index.ahext.template')
        target_path = os.path.join(index_folder_path, 'index')
        logger.info(f"[GEOGRID] create {ah_code}/index: {template_path} -> {target_path}")
        create_file_using_template(template_path, target_path, {
            '###DESCRIPTION###': ah_code
        })

        return index_folder_path

    def generate_lulc_index() -> str:
        index_folder_path = os.path.join(wd_path, 'LULC')
        rm_mk_dir(index_folder_path)

        template_path = os.path.join(os.environ['MODEL_PATH'], 'index.lulc.template')
        target_path = os.path.join(index_folder_path, 'index')
        logger.info(f"[GEOGRID] create LULC/index: {template_path} -> {target_path}")
        create_file_using_template(template_path, target_path, {
            '###DESCRIPTION###': 'LULC'
        })

        return index_folder_path

    n_domains = None
    nx, ny = None, None
    dx, dy, pgr = None, None, None

    # parse namelist.wps.template
    wps_template_path = os.path.join(os.environ['MODEL_PATH'], 'namelist.wps.template')
    with open(wps_template_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'max_dom' in line:
                n_domains = int(extract_value(line, 0))

            elif 'e_we' in line:
                nx = int(extract_value(line, -1)) - 1  # smallest domains -> last item in the list

            elif 'e_sn' in line:
                ny = int(extract_value(line, -1)) - 1  # smallest domains -> last item in the list

            elif 'parent_grid_ratio' in line:
                fields = extract_value(line)
                pgr = 1.0
                for v in fields:  # smallest domains -> iterate through all ratios to calculate pgr
                    pgr /= float(v)

            elif 'dx' in line:
                dx = int(extract_value(line, 0))

            elif 'dy' in line:
                dy = int(extract_value(line, 0))

    # calculate the area of one cell in square meters for the smallest domain
    dx *= pgr
    dy *= pgr
    cell_area = dx * dy

    logger.info(f"[GEOGRID] number of domains: {n_domains}")
    logger.info(f"[GEOGRID] grid dimension (nx ny): {nx} {ny}")
    logger.info(f"[GEOGRID] cell area: {cell_area} [m^2]")

    # create the LULC indices -> this is done so the NUM_LANDCAT variable will be increased to 40. otherwise
    # the real.exe stage will fail. there is probably a better way of doing this. there is no data at this point
    # (i.e., values will be zero). actual data will be injected during the injection stage.
    lulc_path = generate_lulc_index()

    # create the AH data indices -> this is done so the SH_EXT and LH_EXT variables will be added to
    # the geo_em* files. there is no data at this point (i.e., values will be zero). actual data will
    # be injected during the injection stage.
    sh_ext_path = generate_ahext_index('SH_EXT')
    lh_ext_path = generate_ahext_index('LH_EXT')

    # create geogrid path
    geogrid_path = os.path.join(wd_path, 'geogrid')
    rm_mk_dir(geogrid_path)

    # create GEOGRID.TBL
    geogrid_template_path = os.path.join(os.environ['MODEL_PATH'], 'GEOGRID.TBL.template')
    target_path = os.path.join(geogrid_path, 'GEOGRID.TBL')
    logger.info(f"[GEOGRID] create GEOGRID.TBL: {geogrid_template_path} -> {target_path}")
    create_file_using_template(geogrid_template_path, target_path, {
        '###LULC_PATH###': lulc_path,
        '###SH_EXT_PATH###': sh_ext_path,
        '###LH_EXT_PATH###': lh_ext_path
    })

    # determine from/to timestamps
    datetime_from = "{}_{}".format(parameters.t_from.date, parameters.t_from.time)
    datetime_to = "{}_{}".format(parameters.t_to.date, parameters.t_to.time)
    logger.info(f"[GEOGRID] using timestamps: from={datetime_from} to={datetime_to}")

    # check if data path exists
    data_path = os.path.join(os.environ['DATA_PATH'], 'static')
    if not os.path.isdir(data_path):
        raise ProcessorRuntimeError(f"Path to static data not found at {data_path}")

    # create namelist.wps
    target_path = os.path.join(wd_path, 'namelist.wps')
    logger.info(f"[GEOGRID] create namelist.wps: {wps_template_path} -> {target_path}")
    create_file_using_template(wps_template_path, target_path, {
        "###DATE_FROM###": datetime_from,
        "###DATE_TO###": datetime_to,
        "###STATIC_PATH###": data_path
    })

    # run geogrid.exe
    logger.info(f"[GEOGRID] running...")
    if not simulate_only:
        run_exe(wd_path, 'geogrid.exe', 'GEOGRID_EXE', 'geogrid', parameters)

    logger.info(f"[GEOGRID] done")
    return n_domains, nx, ny, cell_area


def run_ungrib(wd_path: str, parameters: Parameters, force_download: bool = False, force_ungrib: bool = False) -> None:
    logger.info(f"[UNGRIB] force download: {force_download}")
    logger.info(f"[UNGRIB] force ungrib: {force_ungrib}")

    # check if data path exists
    ungribbed_path = os.path.join(os.environ['DATA_PATH'], 'dynamic', 'ungribbed')
    if not os.path.isdir(ungribbed_path):
        os.makedirs(ungribbed_path, exist_ok=True)
    logger.info(f"[UNGRIB] using unbribbed path at {ungribbed_path}")

    # determine the from and to timestamps
    date_from = parameters.t_from.date.split('-')
    time_from = parameters.t_from.time.split(':')
    date_to = parameters.t_to.date.split('-')
    time_to = parameters.t_to.time.split(':')
    t0 = datetime.datetime(int(date_from[0]), int(date_from[1]), int(date_from[2]), int(time_from[0]), 0, 0)
    t1 = datetime.datetime(int(date_to[0]), int(date_to[1]), int(date_to[2]), int(time_to[0]), 0, 0)
    logger.info(f"[UNGRIB] required time period: from={date_from} to={date_to}")
    logger.info(f"[UNGRIB] t0={t0.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"[UNGRIB] t1={t1.strftime('%Y-%m-%d %H:%M:%S')}")

    # check if the required 'FILE:yyyy-mm-dd_hh' files already exist
    do_ungrib = force_ungrib
    ti = datetime.datetime(t0.year, t0.month, t0.day, t0.hour, 0, 0)
    while (t1 - ti).total_seconds() >= 0:
        # determine paths
        file_path = os.path.join(ungribbed_path, f"FILE:{ti.year:04d}-{ti.month:02d}-{ti.day:02d}_{ti.hour:02d}")
        grib2_name = f"gdas1.fnl0p25.{ti.year:04d}{ti.month:02d}{ti.day:02d}{ti.hour:02d}.f00.grib2"
        grib2_path = os.path.join(os.environ['DATA_PATH'], 'dynamic', grib2_name)
        grib2_url = f'https://data.rda.ucar.edu/ds083.3/{ti.year:04d}/{ti.year:04d}{ti.month:02d}/{grib2_name}'

        # determine whether to download and ungrib
        file_found = os.path.isfile(file_path)
        grib2_found = os.path.isfile(grib2_path)
        do_download = not grib2_found or force_download
        do_ungrib = do_ungrib or do_download or not file_found
        logger.info(f"[UNGRIB] {ti.year:04d}-{ti.month:02d}-{ti.day:02d}_{ti.hour:02d}: "
                    f"FILE found={file_found} GRIB2 found={grib2_found} -> download: {do_download}")

        # download the file?
        if do_download:
            while True:
                # delete the temp file if it already exists
                temp_path = os.path.join(wd_path, grib2_name)
                if os.path.isfile(temp_path):
                    os.remove(temp_path)

                # download file
                result = subprocess.run(['wget', grib2_url], cwd=wd_path, capture_output=True)
                if result.returncode != 0:
                    logger.error(f"Failed to download {grib2_name}"
                                 f"\nstdout={result.stdout.decode('utf-8')}"
                                 f"\nstderr={result.stderr.decode('utf-8')} -> trying again in 60 seconds...")
                    time.sleep(60)
                else:
                    break

            # move file
            logger.info(f"[UNGRIB] move {temp_path} -> {grib2_path}")
            shutil.move(temp_path, grib2_path)

        ti += datetime.timedelta(hours=6)

    # do we need to run ungrib?
    if do_ungrib:
        logger.info(f"[UNGRIB] running ungrib required: YES")

        # create symbolic link to Vtable
        vtable_target_path = os.path.join(os.environ['MODEL_PATH'], 'Vtable.GFS')
        vtable_link_path = os.path.join(wd_path, 'Vtable')
        mk_symlink(vtable_target_path, vtable_link_path)

        # create symbolic links to GRIB files
        suffixes = [''.join(combination) for combination in itertools.product(string.ascii_uppercase, repeat=3)]
        ti = datetime.datetime(int(date_from[0]), int(date_from[1]), int(date_from[2]), int(time_from[0]), 0, 0)
        while (t1 - ti).total_seconds() >= 0:
            # do we still have suffixes?
            if not suffixes:
                raise ProcessorRuntimeError("Not enough GRIB file suffixes.")

            # determine the filename and path, and create link. example: gdas1.fnl0p25.2020012718.f00.grib2
            target_filename = f"gdas1.fnl0p25.{ti.year:04d}{ti.month:02d}{ti.day:02d}{ti.hour:02d}.f00.grib2"
            target_path = os.path.join(os.environ['DATA_PATH'], 'dynamic', target_filename)
            link_path = os.path.join(wd_path, f"GRIBFILE.{suffixes.pop(0)}")
            mk_symlink(target_path, link_path)

            ti += datetime.timedelta(hours=6)

        # run ungrib.exe
        logger.info(f"[UNGRIB] running...")
        run_exe(wd_path, 'ungrib.exe', 'UNGRIB_EXE', 'ungrib', parameters)

        # move the output to the data directory
        for filename in [f for f in os.listdir(wd_path) if f.startswith('FILE')]:
            src_path = os.path.join(wd_path, filename)
            dst_path = os.path.join(ungribbed_path, filename)
            if os.path.exists(dst_path):
                os.remove(dst_path)

            if not os.path.islink(src_path):
                logger.info(f"[UNGRIB] moving file: {src_path} -> {dst_path}")
                shutil.move(src_path, dst_path)
            else:
                logger.info(f"[UNGRIB] skip moving symlink: {src_path}")

    else:
        logger.info(f"[UNGRIB] running ungrib required: NO")

    # create symbolic links
    for filename in [f for f in os.listdir(ungribbed_path) if f.startswith('FILE')]:
        target_path = os.path.join(ungribbed_path, filename)
        link_path = os.path.join(wd_path, filename)
        mk_symlink(target_path, link_path)


def run_metgrid(wd_path: str, parameters: Parameters) -> int:
    # create metgrid path
    metgrid_path = os.path.join(wd_path, 'metgrid')
    rm_mk_dir(metgrid_path)

    # create GEOGRID.TBL
    template_path = os.path.join(os.environ['MODEL_PATH'], 'METGRID.TBL.template')
    target_path = os.path.join(metgrid_path, 'METGRID.TBL')
    create_file_using_template(template_path, target_path, {})

    # run metgrid.exe
    logger.info(f"[METGRID] running metgrid ...")
    run_exe(wd_path, 'metgrid.exe', 'METGRID_EXE', 'metgrid', parameters)

    # run ncdump
    prefix = os.path.join(wd_path, 'met_em.d01')
    met_em_files = glob.glob(f"{prefix}*")
    grep_out_path = os.path.join(wd_path, 'grep.num_metgrid_levels.out')
    logger.debug(f"[METGRID] running ncdump on {met_em_files[0]}")
    result = subprocess.run(['bash', '-c', f"ncdump -h {met_em_files[0]} | grep num_metgrid_levels > {grep_out_path}"],
                            cwd=wd_path, capture_output=True)
    if result.returncode != 0:
        raise ProcessorRuntimeError("ncdump failed", details={
            'stdout': result.stdout.decode('utf-8'),
            'stderr': result.stderr.decode('utf-8')
        })

    # extract num_metgrid_levels
    with open(grep_out_path, "r") as f:
        lines = f.read()
        lines = lines.split('\n')
        for line in lines:
            if 'num_metgrid_levels =' in line:
                line = line.split()
                num_metgrid_levels = int(line[2])
                logger.info(f"[METGRID] using num_metgrid_levels = {num_metgrid_levels}")
                return num_metgrid_levels

    raise ProcessorRuntimeError("Could not extract num_metgrid_levels value.")


def run_real(wd_path: str, vv_dir_path: str, parameters: Parameters, num_metgrid_levels: int, n_domains: int, nz: int,
             p_top_requested: int, eta_levels: List[float]) -> None:

    def update_bem_parameters(urbparm_path: str, bem_parameters_path: str) -> None:
        logger.info(f"[REAL] updating {urbparm_path} with BEM parameters at {bem_parameters_path}...")

        # parse the BEM parameters
        updates = {}
        with open(bem_parameters_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if ':' not in line:
                    continue

                temp = line.split(':')

                # strip leading/trailing white spaces in variable name and values
                key = temp[0].strip()
                value = temp[1].strip()

                updates[key] = value

        logger.info(f"[REAL] found {len(updates)} BEM parameter updates for {' '.join(updates.keys())}")

        # read all lines from the URBPARM.TBL file
        with open(urbparm_path, 'r') as f:
            lines = f.readlines()

        # update the lines (if necessary)
        new_lines = []
        for line in lines:
            # strip leading/trailing white spaces
            line = line.strip()

            # is the line not a comments and defines a variable?
            if not line.startswith('#') and ':' in line:
                # extract the variable
                temp = line.split(':')
                key = temp[0].strip()

                # do we have an update for this variable?
                if key in updates:
                    values = updates.pop(key)
                    line = f"{key}: {values}"
                    logger.info(f"[REAL] updating {key}: '{temp[1]}' -> '{values}'")

            # append the line
            new_lines.append(line)

        # do we have any BEM parameter updates left?
        if len(updates) > 0:
            logger.info(f"[REAL] found {len(updates)} BEM parameters not present in the template: {' '.join(updates.keys())}")
            for key, value in updates.items():
                logger.info(f"adding {key}: {value}")
                new_lines.append(f"{key}: {value}")

        # write new URBPARM.TBL file
        with open(urbparm_path, 'w') as f:
            for line in new_lines:
                f.write(f"{line}\n")

    # determine from/to timestamps
    date_from = parameters.t_from.date.split('-')
    time_from = parameters.t_from.time.split(':')
    date_to = parameters.t_to.date.split('-')
    time_to = parameters.t_to.time.split(':')

    # create namelist.input
    template_path = os.path.join(os.environ['MODEL_PATH'], f"namelist.input.template-{os.environ['TEMPLATE_VERSION']}")
    target_path = os.path.join(wd_path, 'namelist.input')
    logger.info(f"[REAL] create namelist.input: {template_path} -> {target_path}")
    create_file_using_template(template_path, target_path, {
        "###START_YYYY###": date_from[0],
        "###START_mm###": date_from[1],
        "###START_dd###": date_from[2],
        "###START_HH###": time_from[0],
        "###START_MM###": time_from[1],
        "###START_SS###": time_from[2],
        "###END_YYYY###": date_to[0],
        "###END_mm###": date_to[1],
        "###END_dd###": date_to[2],
        "###END_HH###": time_to[0],
        "###END_MM###": time_to[1],
        "###END_SS###": time_to[2],
        "###NUM_METGRID_LEVELS###": str(num_metgrid_levels),
        "###P_TOP_REQUESTED###": str(p_top_requested),
        "###E_VERT###": str(len(eta_levels)),
        "###ETA_LEVELS###": ','.join([str(v) for v in eta_levels])
    })

    # create URBPARM.TBL
    template_path = os.path.join(os.environ['MODEL_PATH'], 'URBPARM.TBL.template')
    target_path = os.path.join(wd_path, 'URBPARM.TBL')
    create_file_using_template(template_path, target_path, {
        "###FRC_URB1###": parameters.frc_urb1,
        "###FRC_URB2###": parameters.frc_urb2,
        "###FRC_URB3###": parameters.frc_urb3,
        "###FRC_URB4###": parameters.frc_urb4,
        "###FRC_URB5###": parameters.frc_urb5,
        "###FRC_URB6###": parameters.frc_urb6,
        "###FRC_URB7###": parameters.frc_urb7,
        "###FRC_URB8###": parameters.frc_urb8,
        "###FRC_URB9###": parameters.frc_urb9,
        "###FRC_URB10###": parameters.frc_urb10
    })

    # inject BEM parameters into URBPARM.TBL
    bem_parameters_path = os.path.join(wd_path, 'bem-parameters')
    update_bem_parameters(target_path, bem_parameters_path)

    # run real.exe
    logger.info(f"running real ...")
    run_exe(wd_path, 'real.exe', 'REAL_EXE', 'real', parameters)

    # make vv real path
    vv_real_path = os.path.join(vv_dir_path, 'real')
    rm_mk_dir(vv_real_path)

    # print the post-injection vv GeoTIFFs
    for d in range(1, n_domains+1):
        wrfinput_path = os.path.join(wd_path, f"wrfinput_d{d:02d}")
        with open_netcdf(wrfinput_path, 'r') as nc_file:
            # extract the bounding box using 'XLAT' and 'XLON"
            lat = nc_file.variables['XLAT'][0]
            lon = nc_file.variables['XLONG'][0]
            ny = lat.shape[0]
            nx = lat.shape[1]
            bbox = BoundingBox(west=lon[0][0], north=lat[ny - 1][nx - 1], east=lon[ny - 1][nx - 1], south=lat[0][0])

            # extract the LU/GF variables
            lu = nc_file.variables['LU_INDEX']
            vf = nc_file.variables['VEGFRA']
            export_as_tiff(lu[0], os.path.join(vv_real_path, f"LU.d{d:02d}.tiff"), bbox, True, np.uint8)
            export_as_tiff(vf[0], os.path.join(vv_real_path, f"VF.d{d:02d}.tiff"), bbox, True, np.float32)

            # export SH_EXT and LH_EXT for all layer/time that have non-zero values
            sh_ext = nc_file.variables['SH_EXT']
            lh_ext = nc_file.variables['LH_EXT']
            for k in range(0, nz):
                for t in range(0, 24):
                    iz = k*24 + t

                    sh_sum = np.sum(sh_ext[0][iz])
                    if sh_sum > 0:
                        filename = f"SH_EXT_d{d:02d}_k{k:02d}_t{t:02d}.tiff"
                        export_as_tiff(sh_ext[0][iz], os.path.join(vv_real_path, filename), bbox, True, np.uint16)

                    lh_sum = np.sum(lh_ext[0][iz])
                    if lh_sum > 0:
                        filename = f"LH_EXT_d{d:02d}_k{k:02d}_t{t:02d}.tiff"
                        export_as_tiff(lh_ext[0][iz], os.path.join(vv_real_path, filename), bbox, True, np.uint16)


def create_wrf_run_package(wd_path: str) -> None:
    # first copy missing files here
    for filename in ['LANDUSE.TBL', 'VEGPARM.TBL', 'MPTABLE.TBL']:
        src_path = os.path.join(os.environ['MODEL_PATH'], filename)
        dst_path = os.path.join(wd_path, filename)
        shutil.copy(src_path, dst_path)

    # create the package
    package_path = os.path.join(wd_path, 'wrf-run-package')
    logger.info(f"create wrf-run-package at {package_path}")
    result = subprocess.run([
        'tar', 'czf', package_path,
        'wrfbdy_d01', 'wrfinput_d01', 'wrfinput_d02', 'wrfinput_d03', 'wrfinput_d04',
        'LANDUSE.TBL', 'URBPARM.TBL', 'VEGPARM.TBL', 'MPTABLE.TBL', 'namelist.input'
    ], cwd=wd_path, capture_output=True)
    if result.returncode != 0:
        raise ProcessorRuntimeError("Creating vv-package failed", details={
            'stdout': result.stdout.decode('utf-8'),
            'stderr': result.stderr.decode('utf-8')
        })


def create_vv_package(wd_path: str) -> None:
    # collect the items
    vv_contents_path = os.path.join(wd_path, 'vv-contents')
    items = os.listdir(vv_contents_path)

    # create the package
    package_path = os.path.join(wd_path, 'vv-package')
    logger.info(f"create vv-package at {package_path}")

    result = subprocess.run(['tar', 'czf', package_path, *items], cwd=vv_contents_path, capture_output=True)
    if result.returncode != 0:
        raise ProcessorRuntimeError("Creating vv-package failed", details={
            'stdout': result.stdout.decode('utf-8'),
            'stderr': result.stderr.decode('utf-8')
        })


# def perform_sanity_checks(wd_path: str, n_domains: int, nx: int, ny: int, nz: int, cell_area: float) -> None:
#     logger.info(f"perform sanity checks...")
#     for d in range(1, n_domains+1):
#         # read the input file and get the LU_INDEX data set
#         wrfinput_path = os.path.join(wd_path, f"wrfinput_d{d:02d}")
#         with open_netcdf(wrfinput_path, 'r') as nc_file:
#             # print available variables
#             available = [v for v in nc_file.variables]
#             logger.info(f"variables available in d{d:02d}: {available}")
#
#             # get the variables we need
#             lat_ext = nc_file.variables['XLAT'][0]
#             lon_ext = nc_file.variables['XLONG'][0]
#             dny = lat_ext.shape[0]
#             dnx = lat_ext.shape[1]
#
#             # get the UL/LR corner coordinates
#             logger.info(f"d{d:02d} UL -> [0,0]=({lat_ext[0][0]:.6f} {lon_ext[0][0]:.4f})")
#             logger.info(f"d{d:02d} LR -> [{dny},{dnx}]=({lat_ext[dny-1][dnx-1]:.6f} {lon_ext[dny-1][dnx-1]:.4f})")
#
#             # do we have SH_EXT and LH_EXT variables?
#             if 'SH_EXT' not in available or 'LH_EXT' not in available:
#                 raise RuntimeError(f"Missing variables SH_EXT and/or LH_EXT in d{d:02d}")
#             logger.info(f"found SH_EXT and LH_EXT? YES")
#
#             # do the dimension make sense?
#             sh_ext = nc_file.variables['SH_EXT']
#             lh_ext = nc_file.variables['LH_EXT']
#             logger.info(f"shapes of SH_EXT and LH_EXT: {sh_ext.shape} {lh_ext.shape}")
#
#             if d == n_domains:
#                 nh = nz * 24
#                 if sh_ext[0].shape != (nh, ny, nx) or lh_ext[0].shape != (nh, ny, nx):
#                     raise ProcessorRuntimeError(f"Unexpected shapes for SH_EXT and/or LH_EXT in d{d:02d}: "
#                                                 f"sh_ext.shape={sh_ext.shape} lh_et.shape={lh_ext.shape}")
#                 logger.info(f"shape of d{d:02d} SH_EXT and LH_EXT match (_, {nh}, {ny}, {nx})? YES")
#
#             else:
#                 # values of SH_EXT and LH_EXT should be zero for all domains, except the smallest domain
#                 sh_total = 0
#                 lh_total = 0
#                 for tt in range(sh_ext.shape[0]):
#                     sh_total = float(np.sum(sh_ext[tt]))
#                     lh_total = float(np.sum(lh_ext[tt]))
#
#                 if sh_total != 0 or lh_total != 0:
#                     raise ProcessorRuntimeError(f"Non-zero AH values found for in d{d:02d}: "
#                                                 f"sh_total={sh_total} lh_total={lh_total}")
#                 logger.info(f"total SH/LH equals for d{d:02d} to zero? YES")
#
#     # closer analysis of SH_EXT and LH_EXT in smallest domain
#     geoinput_path = os.path.join(wd_path, f"geo_em.d{n_domains:02d}.nc")
#     with open_netcdf(geoinput_path, 'r') as nc_file0:
#         sh_ext0 = nc_file0.variables['SH_EXT']
#         lh_ext0 = nc_file0.variables['LH_EXT']
#         logger.info(f"shape of SH_EXT / LH_EXT (geo_em.d{n_domains:02d}.nc): {sh_ext0.shape} / {lh_ext0.shape}")
#
#         # determine sum of all AH values
#         total_sh0 = float(np.sum(sh_ext0[0]))  # [W/m^2]
#         total_lh0 = float(np.sum(lh_ext0[0]))  # [W/m^2]
#
#         # convert to annual AH in [ktoe]
#         annual_sh0 = convert_wsqm_to_ktoe(total_sh0, cell_area) * 365  # [ktoe/year]
#         annual_lh0 = convert_wsqm_to_ktoe(total_lh0, cell_area) * 365  # [ktoe/year]
#         logger.info(f"annualised SH/LH (geo_em.d{n_domains:02d}.nc): {annual_sh0} {annual_lh0}")
#
#     wrfinput_path = os.path.join(wd_path, f"wrfinput_d{n_domains:02d}")
#     with open_netcdf(wrfinput_path, 'r') as nc_file1:
#         sh_ext1 = nc_file1.variables['SH_EXT']
#         lh_ext1 = nc_file1.variables['LH_EXT']
#         logger.info(f"shape of SH_EXT / LH_EXT (wrfinput_d{n_domains:02d}): {sh_ext1.shape} / {lh_ext1.shape}")
#
#         # determine sum of all AH values
#         total_sh1 = float(np.sum(sh_ext1[0]))  # [W/m^2]
#         total_lh1 = float(np.sum(lh_ext1[0]))  # [W/m^2]
#
#         # convert to annual AH in [ktoe]
#         annual_sh1 = convert_wsqm_to_ktoe(total_sh1, cell_area) * 365  # [ktoe/year]
#         annual_lh1 = convert_wsqm_to_ktoe(total_lh1, cell_area) * 365  # [ktoe/year]
#         logger.info(f"annualised SH/LH (wrfinput_d{n_domains:02d}): {annual_sh1} {annual_lh1}")
#



class WRFPrepProcessor(ProcessorBase):
    def __init__(self, proc_path: str) -> None:
        super().__init__(proc_path)

        self._is_cancelled = False

    def run(self, wd_path: str, callback: ProgressListener, logger: logging.Logger) -> None:
        def check_if_cancelled_and_pub_progress(progress: int) -> None:
            callback.on_progress_update(progress)
            if self._is_cancelled:
                raise ProcessorRuntimeError(f"cancelled -> exiting now.")

        # determine the processor directory
        proc_path = os.getcwd()
        logger.info(f"begin executing ucm-wrf/proc_prep: wd_path={wd_path} proc_path={proc_path}")
        check_if_cancelled_and_pub_progress(1)

        # check environment variables
        check_environment_variables(['DATA_PATH', 'WPS_DIR', 'WRF_DIR', 'MODEL_PATH', 'GEOGRID_EXE',
                                     'UNGRIB_EXE', 'METGRID_EXE', 'REAL_EXE', 'USE_NETCDF_VERSION', 'USE_PBS'])

        # load parameters
        callback.on_message(Severity.INFO, 'Loading parameters.')
        parameters_path = os.path.join(wd_path, "parameters")
        parameters = load_parameters(parameters_path)
        logger.info(f"parameters: {parameters}")
        check_if_cancelled_and_pub_progress(2)

        # create folders for keeping vv data
        callback.on_message(Severity.INFO, 'Creating folders for visual validation data...')
        vv_dir_path = os.path.join(wd_path, 'vv-contents')
        rm_mk_dir(vv_dir_path)
        check_if_cancelled_and_pub_progress(3)

        # determine ETA levels
        callback.on_message(Severity.INFO, 'Generating ETA levels and height conversion...')
        eta_levels, nz, height_levels = generate_eta_levels(wd_path, default_p_top_requested, default_z_levels,
                                                            default_height_limit)
        logger.info(f"height levels: {height_levels}")
        logger.info(f"ETA levels: {eta_levels}")
        check_if_cancelled_and_pub_progress(5)

        # perform geogrid stage
        callback.on_message(Severity.INFO, 'Executing geogrid stage...')
        n_domains, nx, ny, cell_area = run_geogrid(wd_path, parameters)
        check_if_cancelled_and_pub_progress(20)

        # inject geodata
        callback.on_message(Severity.INFO, 'Injecting geodata stage...')
        inject_geodata(wd_path, vv_dir_path, n_domains, nx, ny, nz, cell_area, height_levels)
        check_if_cancelled_and_pub_progress(35)

        # perform ungrib stage
        callback.on_message(Severity.INFO, 'Executing ungrib stage...')
        run_ungrib(wd_path, parameters, force_download=False, force_ungrib=False)
        check_if_cancelled_and_pub_progress(50)

        # perform metgrid stage
        callback.on_message(Severity.INFO, 'Executing metgrid stage...')
        num_metgrid_levels = run_metgrid(wd_path, parameters)
        check_if_cancelled_and_pub_progress(65)

        # perform real stage
        callback.on_message(Severity.INFO, 'Executing real stage...')
        run_real(wd_path, vv_dir_path, parameters, num_metgrid_levels, n_domains, nz,
                 default_p_top_requested, eta_levels)
        check_if_cancelled_and_pub_progress(80)

        # create the wrf run package
        callback.on_message(Severity.INFO, 'Creating WRF run package...')
        create_wrf_run_package(wd_path)
        callback.on_output_available('wrf-run-package')
        check_if_cancelled_and_pub_progress(90)

        # create the visual validation package
        callback.on_message(Severity.INFO, 'Creating visual validation package...')
        create_vv_package(wd_path)
        callback.on_output_available('vv-package')
        check_if_cancelled_and_pub_progress(95)

        # we are done
        logger.info(f"done.")
        callback.on_message(Severity.INFO, "Done (ucmwrf-prep)")
        check_if_cancelled_and_pub_progress(100)

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
        logger.info(f"Attempting to run the WRF pre-processor...'")
        proc = WRFPrepProcessor(proc_path)
        proc.run(wd_path, callback, logger)
        logger.info("Done!")
        sys.exit(0)

    except ProcessorRuntimeError as e:
        logger.error(f"Exception {e.id}: {e.reason}\ndetails: {e.details}")
        callback.on_message(Severity.ERROR, e.reason)

        sys.exit(-2)
