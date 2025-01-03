import abc
import datetime
import logging
import re
import shutil
import threading
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Optional, List, Tuple

import sys

import os
import time
import subprocess
import h5py
import netCDF4
import numpy as np
import rasterio
from _datetime import timedelta

from pydantic import BaseModel, Field
from pyproj import CRS
from saas.core.helpers import generate_random_string, get_timestamp_now
from saas.core.logging import Logging
from saas.dor.schemas import ProcessorDescriptor
from scipy.io import netcdf_file

logger = Logging.get('ucm-palm-sim')

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
    north: float
    east: float
    south: float
    west: float


class Parameters(BaseModel):
    name: str
    datetime_offset: str
    resolution: List[int]


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


def get_simulation_progress(job_progress_path: str) -> int:
    # is there a progress file?
    if not os.path.isfile(job_progress_path):
        logger.info(f"no progress file found at '{job_progress_path}'")
        return 0

    # read contents of the PROGRESS file
    # CBD_TEST_DUCT
    #  0.30  0.30
    with open(job_progress_path, 'r') as f:
        lines = f.readlines()
        temp = lines[1].strip().split()

        progress = float(temp[0])
        progress = int(100 * progress)
        return progress


def check_and_extract(f: netcdf_file, variable: str, expected_units: Optional[str] = None) -> np.ndarray:
    # check if the variable exists
    if variable not in f.variables:
        raise ProcessorRuntimeError(f"Variable '{variable}' not found in {f.filename}")

    v = f.variables[variable]
    units = v.units.decode('utf-8')

    # check if the units match
    if expected_units and not units == expected_units:
        raise ProcessorRuntimeError(f"Unexpected units '{units}' for '{variable}'.", details={
            'expected_units': expected_units,
            'units': units
        })

    return v.data


def kelvin_to_celsius(t: np.ndarray) -> np.ndarray:
    return t - 273.15


def extract_datetime_offset(parameters: Parameters) -> datetime:
    offset = time.strptime(parameters.datetime_offset, "%Y-%m-%d %H:%M:%S")
    offset = datetime(*offset[:6])
    return offset


def write_geotiff(data: np.ndarray, tiff_out_path: str, width: int, height: int, bbox: BoundingBox, dtype, nodata=None) -> None:
    with rasterio.open(tiff_out_path, 'w+', driver='GTiff',
                       width=width,
                       height=height,
                       count=1,
                       dtype=dtype,
                       nodata=nodata,
                       crs=CRS.from_string("EPSG:4326"),
                       transform=rasterio.transform.from_bounds(bbox.west, bbox.south, bbox.east, bbox.north,
                                                                width=width, height=height)
                       ) as dataset:
        dataset.write(data, 1)


def create_vv_output(data: np.ndarray, timestamps: List[int], parameters: Parameters, bounding_box: BoundingBox,
                     vv_output_path: str, vv_output_name: str, dtype, z_levels=None, nodata=None) -> List[str]:
    # determine the datetime offset
    t_offset = extract_datetime_offset(parameters)

    n_times = data.shape[0]

    # iterate over all times and create a GeoTIFF
    filenames = []
    if len(data.shape) == 3:
        height = data.shape[1]
        width = data.shape[2]

        for idx in range(n_times):
            # get the date and time
            t_idx = t_offset + timedelta(seconds=timestamps[idx])
            t_idx = t_idx.strftime("D%y%m%d_T%H%M%S")

            # determine the filename
            filename = f"{vv_output_name}_{t_idx}.tiff"
            filenames.append(filename)

            # write the GeoTIFF
            tiff_out_path = os.path.join(vv_output_path, filename)
            write_geotiff(data[idx], tiff_out_path, width, height, bounding_box, dtype, nodata=nodata)

    elif len(data.shape) == 4:
        height = data.shape[2]
        width = data.shape[3]

        # write GeoTIFF for selected Z levels
        z_levels = z_levels if z_levels else [0]
        for z_idx in z_levels:
            for idx in range(n_times):
                # get the date and time
                t_idx = t_offset + timedelta(seconds=timestamps[idx])
                t_idx = t_idx.strftime("D%y%m%d_T%H%M%S")

                # determine the filename
                filename = f"{vv_output_name}_{t_idx}_Z{z_idx}.tiff"
                filenames.append(filename)

                subset = data[:, z_idx, :, :]

                tiff_out_path = os.path.join(vv_output_path, filename)
                write_geotiff(subset[idx], tiff_out_path, width, height, bounding_box, dtype, nodata=nodata)

    else:
        raise ProcessorRuntimeError(f"Unexpected shape dimensionality {data.shape} for dataset '{vv_output_name}'")

    return filenames


def add_hdf5_output(f_out: h5py.File, name: str, unit: str, data: np.ndarray, timestamps: List[int],
                    parameters: Parameters, bounding_box: BoundingBox, z_slice: Tuple[int, int] = None) -> None:
    bounding_box = [bounding_box.south, bounding_box.north, bounding_box.west, bounding_box.east]

    # is the data 2D or 3D?
    shape = data.shape
    if len(shape) == 3:
        # create the output data set
        data_set = f_out.create_dataset(name, data=data, track_times=False)
        data_set.attrs['unit'] = unit
        data_set.attrs['shape'] = data.shape
        data_set.attrs['timestamps'] = timestamps
        data_set.attrs['bounding_box'] = bounding_box

    elif len(shape) == 4:
        # truncate the z-dimension?
        z_res = parameters.resolution[2]
        heights = [i * z_res for i in range(shape[1])]

        if z_slice:
            data = data[:, slice(z_slice[0], z_slice[1]), :, :]
            shape = data.shape
            heights = heights[z_slice[0]:z_slice[1]]

        # create the output data set
        data_set = f_out.create_dataset(name, data=data, track_times=False)
        data_set.attrs['unit'] = unit
        data_set.attrs['shape'] = shape
        data_set.attrs['timestamps'] = timestamps
        data_set.attrs['heights'] = heights
        data_set.attrs['bounding_box'] = bounding_box

    else:
        raise ProcessorRuntimeError(f"Unexpected shape dimensionality {shape} for dataset '{name}'")


def create_vv_output_package(wd_path: str, filenames: List[str], remove_individual_files: bool = True) -> None:
    # archive them
    package_path = os.path.join(wd_path, 'vv-package')
    result = subprocess.run(['tar', 'czf', package_path, *filenames], capture_output=True, cwd=wd_path)
    if result.returncode != 0:
        raise ProcessorRuntimeError(f"Failed to create VV package at {package_path}", details={
            'filenames': filenames,
            'stdout': result.stdout.decode('utf-8'),
            'stderr': result.stderr.decode('utf-8')
        })

    if remove_individual_files:
        # delete the files
        for f in filenames:
            os.remove(os.path.join(wd_path, f))


def convert_to_abs_timestamps(relative_timestamps: List[int], offset: str) -> List[int]:
    temp = offset.split(' ')
    date = temp[0].split('-')
    time = temp[1].split(':')
    t = datetime(int(date[0]), int(date[1]), int(date[2]), int(time[0]), int(time[1]), int(time[2]))

    timestamps = [t + timedelta(seconds=int(dt)) for dt in relative_timestamps]
    timestamps = [timestamp.strftime("%Y%m%d%H%M00") for timestamp in timestamps]
    timestamps = [int(timestamp) for timestamp in timestamps]
    return timestamps


def postprocess(raw_output_path: str, parameters: Parameters, bounding_box: BoundingBox, wd_path: str,
                callback: Optional[ProgressListener] = None, dt_warmup: int = 2) -> None:

    av_xy_path = os.path.join(raw_output_path, f"{parameters.name}_av_xy.000.nc")
    av_3d_path = os.path.join(raw_output_path, f"{parameters.name}_av_3d.000.nc")
    vv_filenames = []

    # create the HDF5 output file
    hdf5_out_path = os.path.join(wd_path, 'climatic-variables')
    with h5py.File(hdf5_out_path, "w") as f_out:
        # process the 3D file
        with netcdf_file(av_3d_path, 'r', mmap=True) as nc_file:
            # time (26,) seconds
            v_time = check_and_extract(nc_file, 'time', 'seconds')
            v_time = v_time[dt_warmup-1:]  # (26,) -> (24,)
            rel_timestamps: List[int] = [int(t) for t in v_time]
            abs_timestamps: List[int] = convert_to_abs_timestamps(rel_timestamps, parameters.datetime_offset)

            # wspeed (26, 242, 384, 384) m/s
            v_wspeed = check_and_extract(nc_file, 'wspeed', 'm/s')
            v_wspeed = v_wspeed[dt_warmup-1:]  # (26, 242, 384, 384) -> (24, 242, 384, 384)

            # wdir (26, 242, 384, 384) degree
            v_wdir = check_and_extract(nc_file, 'wdir', 'degree')
            v_wdir = v_wdir[dt_warmup-1:]  # (26, 242, 384, 384) -> (24, 242, 384, 384)

            # rh (26, 242, 384, 384) %
            v_rh = check_and_extract(nc_file, 'rh', '%')
            v_rh = v_rh[dt_warmup-1:]  # (26, 242, 384, 384) -> (24, 242, 384, 384)

            # ta (26, 242, 384, 384) degree_
            v_ta = check_and_extract(nc_file, 'ta', 'degree_')
            v_ta = v_ta[dt_warmup-1:]  # (26, 242, 384, 384) -> (24, 242, 384, 384)

            # add datasets to HDF5 file. only up to 300m
            # z_slice = (0, int(math.ceil(300 / parameters.resolution[2])))
            z_slice = (0, 2)  # TODO: change back to the above line when done testing
            add_hdf5_output(f_out, 'wind_speed', 'm/s', v_wspeed, abs_timestamps, parameters, bounding_box, z_slice)
            add_hdf5_output(f_out, 'wind_direction', '˚', v_wdir, abs_timestamps, parameters, bounding_box, z_slice)
            add_hdf5_output(f_out, 'relative_humidity', '%', v_rh, abs_timestamps, parameters, bounding_box, z_slice)
            add_hdf5_output(f_out, 'air_temperature', '˚C', v_ta, abs_timestamps, parameters, bounding_box, z_slice)

            # # create visual validation output
            vv_filenames.extend(create_vv_output(v_wspeed, rel_timestamps, parameters, bounding_box, wd_path, 'ws',
                                                 dtype=np.float32, z_levels=[1], nodata=-9999))
            vv_filenames.extend(create_vv_output(v_wdir, rel_timestamps, parameters, bounding_box, wd_path, 'wd',
                                                 dtype=np.int16, z_levels=[1], nodata=-9999))
            vv_filenames.extend(create_vv_output(v_rh, rel_timestamps, parameters, bounding_box, wd_path, 'rh',
                                                 dtype=np.float32, z_levels=[1], nodata=-9999))
            vv_filenames.extend(create_vv_output(v_ta, rel_timestamps, parameters, bounding_box, wd_path, 'ta',
                                                 dtype=np.float32, z_levels=[1], nodata=-9999))

        # process the XY file
        with netcdf_file(av_xy_path, 'r', mmap=True) as nc_file:
            # time (26,) seconds
            v_time = check_and_extract(nc_file, 'time', 'seconds')
            v_time = v_time[dt_warmup-1:]  # (26,) -> (24,) seconds
            rel_timestamps: List[int] = [int(t) for t in v_time]
            abs_timestamps: List[int] = convert_to_abs_timestamps(rel_timestamps, parameters.datetime_offset)

            # tsurf*_xy (26, 1, 384, 384) K
            v_ts = check_and_extract(nc_file, 'tsurf*_xy', 'K')
            v_ts = v_ts[dt_warmup-1:]  # (26, 1, 384, 384) -> (24, 1, 384, 384)
            v_ts = kelvin_to_celsius(v_ts)
            v_ts = np.squeeze(v_ts)

            # bio_pet*_xy (26, 1, 384, 384) degree_
            v_pet = check_and_extract(nc_file, 'bio_pet*_xy', 'degree_')
            v_pet = v_pet[dt_warmup-1:]  # (26, 1, 384, 384) -> (24, 1, 384, 384)
            v_pet = np.squeeze(v_pet)

            # add datasets to HDF5 file
            add_hdf5_output(f_out, 'surface_temperature', '˚C', v_ts, abs_timestamps, parameters, bounding_box)
            add_hdf5_output(f_out, 'pet', '˚C', v_pet, abs_timestamps, parameters, bounding_box)

            # create visual validation output
            vv_filenames.extend(
                create_vv_output(v_ts, rel_timestamps, parameters, bounding_box, wd_path, 'ts', dtype=np.int8)
            )
            vv_filenames.extend(
                create_vv_output(v_pet, rel_timestamps, parameters, bounding_box, wd_path, 'pet', dtype=np.int8)
            )

    # create the visual validation package
    package_path = os.path.join(wd_path, 'vv-package')
    result = subprocess.run(['tar', 'czf', package_path, *vv_filenames], capture_output=True, cwd=wd_path)
    if result.returncode != 0:
        raise ProcessorRuntimeError(f"Failed to create VV package at {package_path}", details={
            'filenames': vv_filenames,
            'stdout': result.stdout.decode('utf-8'),
            'stderr': result.stderr.decode('utf-8')
        })

    if callback:
        callback.on_output_available('climatic-variables')
        callback.on_output_available('vv-package')


def check_palmrun_logs(log_path: str) -> None:
    def read_text_block(lines: List[str]) -> Optional[str]:
        if len(lines) == 0:
            return None

        # determine the number of spaces
        n_spaces = len(lines[0]) - len(lines[0].lstrip())

        # read the next n lines that match number of spaces
        block = []
        while len(lines) > 0:
            line = lines.pop(0)
            n = len(line) - len(line.lstrip())
            if n < n_spaces:
                break

            line = line[n_spaces:]
            line = line.rstrip()
            if len(line) != 0:
                line = line+' ->' if line.startswith('ID') else line
                block.append(line)

        result = ' '.join(block)
        return result

    with open(log_path, 'r') as f:
        lines = f.readlines()
        while len(lines) > 0:
            line = lines.pop(0)
            if '+++ error message ---' in line:
                message = read_text_block(lines)
                raise ProcessorRuntimeError('Error during simulation run', details={
                    'message': message
                })


def extract_bounding_box(static_driver_path: str) -> BoundingBox:
    with netCDF4.Dataset(static_driver_path, 'r') as f:
        lat = f.variables['lat']
        lon = f.variables['lon']
        bounding_box = BoundingBox(west=np.min(lon), east=np.max(lon), south=np.min(lat), north=np.max(lat))
        return bounding_box


class PalmSimProcessor(ProcessorBase):
    def __init__(self, logger, proc_path: str) -> None:
        super().__init__(proc_path)

        self._logger = logger
        self._is_cancelled = False

    def extract_package(self, package_path: str, destination_path: str) -> BoundingBox:
        self._logger.info(f"extracting package at {package_path} -> {destination_path}")

        # create the destination folder
        if os.path.isdir(destination_path):
            self._logger.warning(f"folder {destination_path} already exists -> deleting.")
            shutil.rmtree(destination_path)
        os.makedirs(destination_path)

        # move package
        if not os.path.isfile(package_path):
            raise ProcessorRuntimeError(f"Package at '{package_path}' does not exist.")
        shutil.copy(package_path, os.path.join(destination_path, 'package.tar.gz'))

        # unpack the package
        result = subprocess.run(['tar', 'xzf', "package.tar.gz"], cwd=destination_path, capture_output=True)
        if result.returncode != 0:
            raise ProcessorRuntimeError(f"Failed to unpack data", details={
                'package_path': package_path,
                'destination_path': destination_path,
                'stdout': result.stdout.decode('utf-8'),
                'stderr': result.stderr.decode('utf-8'),
                'is_cancelled': self._is_cancelled
            })

        # rename the files
        suffixes = ['dynamic', 'p3d', 'p3dr', 'rlw', 'rsw', 'static', 'ah']
        for suffix in suffixes:
            path0 = os.path.join(destination_path, f"NAME_{suffix}")
            if os.path.exists(path0):
                path1 = os.path.join(destination_path, f"{self._parameters.name}_{suffix}")
                self._logger.info(f"renaming {path0} -> {path1}")
                os.rename(path0, path1)

        # delete the package.tar.gz
        os.remove(os.path.join(destination_path, 'package.tar.gz'))

        # extract the bounding box from the static driver
        static_driver_path = os.path.join(destination_path, f"{self._parameters.name}_static")
        return extract_bounding_box(static_driver_path)

    def submit_simulation(self, palm_build_path: str) -> None:
        # create submit script
        submit_script_path = os.path.join(palm_build_path, f'submit_{self._parameters.name}.sh')
        with open(submit_script_path, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write(f'palmrun -c default -r {self._parameters.name} -a \"d3#\" -X256 -T128 -t 86400 -b -B -v\n')
            f.write('exit $?\n')

        # make script executable
        result = subprocess.run(['chmod', 'u+x', submit_script_path], cwd=palm_build_path, capture_output=True)
        if result.returncode != 0:
            raise ProcessorRuntimeError(f"Failed to make submit script executable", details={
                'palm_build_path': palm_build_path,
                'stdout': result.stdout.decode('utf-8'),
                'stderr': result.stderr.decode('utf-8'),
                'is_cancelled': self._is_cancelled
            })

        # run submit script
        result = subprocess.run([submit_script_path], cwd=palm_build_path, capture_output=True)
        if result.returncode != 0:
            raise ProcessorRuntimeError(f"Failed to submit job", details={
                'palm_build_path': palm_build_path,
                'stdout': result.stdout.decode('utf-8'),
                'stderr': result.stderr.decode('utf-8'),
                'is_cancelled': self._is_cancelled
            })

        # remove submit script
        os.remove(submit_script_path)

    def get_job_status(self) -> (Optional[str], Optional[str]):
        # check the status by running qstat
        result = subprocess.run(['qstat', '-x'], capture_output=True)
        if result.returncode != 0:
            raise ProcessorRuntimeError("Running qstat failed", details={
                'stdout': result.stdout.decode('utf-8'),
                'stderr': result.stderr.decode('utf-8'),
                'is_cancelled': self._is_cancelled
            })

        # extract the status and look for something like this:
        # Job id                 Name             User              Time Use S Queue
        # ---------------------  ---------------- ----------------  -------- - -----
        # 2333413.pbs101         CBD_TEST_DUCT.1* aydt              1002:53* R q5
        queued = []
        finished = []
        running = []
        self._logger.info(f"qstat at {get_timestamp_now()}:")
        for line in result.stdout.decode('utf-8').split('\n'):
            # skip header lines
            if line.startswith('Job id') or line.startswith('---'):
                continue

            # split the line into components
            line = re.sub(r'\s+', ' ', line)
            temp = line.split()
            if len(temp) < 6:
                continue

            # parse only the relevant lines
            if temp[1].startswith(self._parameters.name):
                # determine job info and seq id and find the highest sequence id
                job_id = temp[0]
                pattern = temp[1]
                status = temp[4]

                # according to man pages, status can be the following:
                # B  Array job has at least one subjob running
                # E  Job is exiting after having run
                # F  Job is finished
                # H  Job is held
                # M  Job was moved to another server
                # Q  Job is queued
                # R  Job is running
                # S  Job is suspended
                # T  Job is being moved to new location
                # U  Cycle-harvesting job is suspended due to keyboard activity
                # W  Job is waiting for its submitter-assigned start time to be reached
                # X  Subjob has completed execution or has been deleted

                if status == 'Q':
                    queued.append((job_id, pattern))
                elif status == 'R':
                    running.append((job_id, pattern))
                elif status == 'F':
                    finished.append((job_id, pattern))
                else:
                    self._logger.warning(f"ignoring job with unexpected status '{status}': {temp}")

        self._logger.info(f"queued={queued} running={running} finished={finished}")

        # queued but no running jobs?
        if queued and not running:
            return 'Q', queued[0][1]

        # only finished jobs?
        if not queued and not running and finished:
            return 'F', None

        # no queued but one running job?
        if not queued and len(running) == 1:
            return 'R', running[0][1]

        self._logger.warning(f"Unexpected qstat situation: queued={queued} running={running} finished={finished}")
        return None, None

    def find_matching_subdirectories(self, base_folder: str, pattern: str) -> List[str]:
        # remove the asterisk (if any)
        if pattern[-1] == '*':
            new_pattern = pattern[:-1]
            self._logger.info(f"removing asterisk: {pattern} -> {new_pattern}")
            pattern = new_pattern

        # identify all matching subdirectories
        result = []
        for item in os.listdir(base_folder):
            if item.startswith(pattern):
                path = os.path.join(base_folder, item)
                if os.path.isdir(path):
                    result.append(path)

        return result

    def run(self, wd_path: str, callback: ProgressListener, logger: logging.Logger) -> None:
        def check_if_cancelled_and_pub_progress(progress: int) -> None:
            callback.on_progress_update(progress)
            if self._is_cancelled:
                raise ProcessorRuntimeError(f"cancelled -> exiting now.")

        check_if_cancelled_and_pub_progress(1)

        # check environment variables
        check_environment_variables(['PALM_PATH'])

        # determine paths
        palm_build_path = os.path.join(os.environ['PALM_PATH'], 'build')
        palm_jobs_path = os.path.join(palm_build_path, 'JOBS')
        palm_tmp_path = os.path.join(palm_build_path, 'tmp')

        # load parameters
        callback.on_message(Severity.INFO, 'Loading parameters.')
        parameters_path = os.path.join(wd_path, "parameters")
        self._parameters = load_parameters(parameters_path)
        check_if_cancelled_and_pub_progress(2)

        # unpack the network data packages
        callback.on_message(Severity.INFO, 'Unpacking run package contents.')
        package_path = os.path.join(wd_path, 'palm-run-package')
        job_path = os.path.join(palm_jobs_path, self._parameters.name)
        job_input_path = os.path.join(job_path, 'INPUT')
        bounding_box = self.extract_package(package_path, job_input_path)

        # copy user code
        callback.on_message(Severity.INFO, 'Copy user code for building AH.')
        user_code_path = os.path.join('..', 'templates', 'USER_CODE')
        job_user_code_path = os.path.join(job_path, 'USER_CODE')
        shutil.copytree(user_code_path, job_user_code_path, dirs_exist_ok=True)

        # submit simulation
        callback.on_message(Severity.INFO, 'Submitting simulation job.')
        self.submit_simulation(palm_build_path)

        # wait for simulation to start
        callback.on_message(Severity.INFO, 'Waiting for simulation job to start...')
        while not self._is_cancelled:
            time.sleep(30)

            # get the current status
            status, pattern = self.get_job_status()
            if status is None:
                continue

            self._logger.info(f"status={status} pattern={pattern}")

            if status == 'R':
                # determine the specific job path
                matching_subdirectories = self.find_matching_subdirectories(palm_tmp_path, pattern)
                if len(matching_subdirectories) == 1:
                    job_seq_path = matching_subdirectories[0]
                    job_progress_path = os.path.join(job_seq_path, 'PROGRESS')
                    simulation_progress = get_simulation_progress(job_progress_path)
                    callback.on_message(Severity.INFO, 'Waiting for simulation job to finish...')

                    # scale sim progress 0..100% to proc process between 2..90%
                    callback.on_progress_update(2 + 0.88*simulation_progress)
                else:
                    self._logger.info(f"unexpected matching subdirectories: {matching_subdirectories}")

            elif status == 'F':
                callback.on_message(Severity.INFO, 'Simulation job finished.')
                break

        check_if_cancelled_and_pub_progress(90)

        # check the logs
        job_logs_path = os.path.join(job_path, 'LOG_FILES')
        for f in os.listdir(job_logs_path):
            if self._parameters.name in f:
                log_path = os.path.join(job_logs_path, f)
                check_palmrun_logs(log_path)

        # post-processing
        callback.on_message(Severity.INFO, 'Post-processing...')
        raw_output_path = os.path.join(job_path, 'OUTPUT')
        postprocess(raw_output_path, self._parameters, bounding_box, wd_path, callback)

        check_if_cancelled_and_pub_progress(99)

        # cleaning up
        callback.on_message(Severity.INFO, 'Cleaning up...')
        removable = [job_path]
        for f in os.listdir(palm_tmp_path):
            if self._parameters.name in f:
                removable.append(os.path.join(palm_tmp_path, f))

        # delete items
        for f in removable:
            logger.info(f"deleting {f}")
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)

        # indicate we are done
        callback.on_progress_update(100)
        callback.on_message(Severity.INFO, 'Done.')

    def cancel(self) -> None:
        # check the status by running qstat
        result = subprocess.run(['qstat', '-x'], capture_output=True)
        if result.returncode != 0:
            raise ProcessorRuntimeError("Running qstat failed", details={
                'stdout': result.stdout.decode('utf-8'),
                'stderr': result.stderr.decode('utf-8'),
                'is_cancelled': self._is_cancelled
            })

        # extract the status and look for something like this:
        # Job id                 Name             User              Time Use S Queue
        # ---------------------  ---------------- ----------------  -------- - -----
        # 2333413.pbs101         CBD_TEST_DUCT.1* aydt              1002:53* R q5
        unfinished = []
        for line in result.stdout.decode('utf-8').split('\n'):
            # skip header lines
            if line.startswith('Job id') or line.startswith('---'):
                continue

            # split the line into components
            line = re.sub(r'\s+', ' ', line)
            temp = line.split()
            if len(temp) < 6:
                continue

            # check if the name matches
            if temp[1].startswith(self._parameters.name):
                # all job's that haven't finished need to be cancelled
                job_id = temp[0]
                status = temp[4]
                if status != 'F':
                    unfinished.append(job_id)
        logger.info(f"found unfinished jobs for '{self._parameters.name}': {unfinished}")

        # cancel all the unfinished jobs
        for job_id in unfinished:
            result = subprocess.run(['qdel', job_id], capture_output=True)
            if result.returncode != 0:
                logger.info(f"cancelling job {job_id}...failed: "
                            f"stdout={result.stdout.decode('utf-8')} "
                            f"stderr={result.stderr.decode('utf-8')}")
            else:
                logger.info(f"cancelling job {job_id}...done.")

        self._is_cancelled = True


if __name__ == '__main__':
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
        logger.info(f"Attempting to run the Palm4U simulation...'")
        proc = PalmSimProcessor(logger, proc_path)
        proc.run(wd_path, callback, logger)
        logger.info("Done!")
        sys.exit(0)

    except ProcessorRuntimeError as e:
        logger.error(f"Exception {e.id}: {e.reason}\ndetails: {e.details}")
        callback.on_message(Severity.ERROR, e.reason)

        sys.exit(-2)
