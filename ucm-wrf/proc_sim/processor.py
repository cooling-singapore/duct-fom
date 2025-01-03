import abc
import datetime
import glob
import logging
import re
import shutil
import threading
import sys
import os
import time
import subprocess
from contextlib import contextmanager

import h5py
import rasterio
import numpy as np
from enum import Enum
from threading import Lock
from typing import Optional, List, Union

import wrf
from pydantic import BaseModel, Field
from rasterio import CRS
from saas.core.helpers import generate_random_string
from saas.core.logging import Logging
from saas.dor.schemas import ProcessorDescriptor

from wbgt import wbgt, calc_cza, calc_fdir

logger = logging.getLogger('ucm-wrf-sim')


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
    class Settings(BaseModel):
        pbs_project_id: str
        pbs_queue: str

    name: str
    settings: Optional[Settings]


class BoundingBox(BaseModel):
    north: float
    east: float
    south: float
    west: float


class Dimensions(BaseModel):
    width: int
    height: int


class KillSwitch:
    def __init__(self) -> None:
        self._id = None
        self._attempt = 0

    def set(self, id: Union[int, str]) -> None:
        self._id = id
        if os.environ['USE_PBS'] == 'yes':
            logger.warning(f"kill switch set for PBS job {id}")
        else:
            logger.warning(f"kill switch set for process {id}")

    @property
    def activated(self) -> bool:
        return self._attempt > 0

    def activate(self) -> None:
        if self._id is None:
            logger.warning(f"kill switch activated without process/job id -> ignoring...")

        else:
            # determine the SIGTERM/SIGKILL commands
            if os.environ['USE_PBS'] == 'yes':
                sigterm = ['qdel', f'{self._id}']
                sigkill = ['qdel', '-W', 'force', f'{self._id}']

            else:
                sigterm = ['kill', f'{self._id}']
                sigkill = ['kill', '-9', f'{self._id}']

            if self._attempt == 0:
                logger.warning(f"kill switch activated: attempt={self._attempt} -> sending SIGTERM to {self._id}")
                result = subprocess.run(sigterm, capture_output=True)
                if result.returncode != 0:
                    logger.error(f"sending SIGTERM/SIGKILL failed: "
                                 f"stdout={result.stdout.decode('utf-8')} "
                                 f"stderr={result.stderr.decode('utf-8')}")
                else:
                    self._attempt += 1

            else:
                logger.warning(f"kill switch activated: attempt={self._attempt} -> sending SIGKILL to {self._id}")
                result = subprocess.run(sigkill, capture_output=True)
                if result.returncode != 0:
                    logger.error(f"sending SIGTERM/SIGKILL failed: "
                                 f"stdout={result.stdout.decode('utf-8')} "
                                 f"stderr={result.stderr.decode('utf-8')}")
                else:
                    self._attempt += 1


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


def prepare_run_folder(wd_path: str, parameters: Parameters) -> str:
    # do we have a scratch path?
    if 'SCRATCH_PATH' in os.environ and os.environ['SCRATCH_PATH'] != '':
        scratch_path = os.environ['SCRATCH_PATH']
    else:
        scratch_path = wd_path

    # make a copy of the wrf/run directory in the scratch home
    run_path0 = os.path.join(os.environ['WRF_DIR'], 'run')
    run_path1 = os.path.join(scratch_path, f'run.{parameters.name}')
    if os.path.isdir(run_path1):
        logger.info(f"making copy of wrf/run folder: {run_path0} -> {run_path1} (EXISTING -> delete first)")
        shutil.rmtree(run_path1)
    else:
        logger.info(f"making copy of wrf/run folder: {run_path0} -> {run_path1}")
    shutil.copytree(run_path0, run_path1)

    # copy the executables
    for exe in ['ndown.exe', 'real.exe', 'tc.exe', 'wrf.exe']:
        exe_path0 = os.path.join(os.environ['WRF_DIR'], 'main', exe)
        exe_path1 = os.path.join(run_path1, exe)
        shutil.copy(exe_path0, exe_path1)

    # copy the information file into the run folder to support debugging
    information_path0 = os.path.join(wd_path, 'information')
    information_path1 = os.path.join(run_path1, 'information.json')
    shutil.copy(information_path0, information_path1)

    # copy the run package contents into the run directory
    run_package_path0 = os.path.join(wd_path, 'wrf-run-package')
    run_package_path1 = os.path.join(run_path1, 'wrf-run-package.tar.gz')
    logger.info(f"copy wrf-run-package contents into wrf/run: {run_package_path0} -> {run_package_path1}")
    if not os.path.isfile(run_package_path0):
        raise ProcessorRuntimeError(f"Missing input: run package not found at {run_package_path0}")
    shutil.copy(run_package_path0, run_package_path1)

    # extract the run package contents
    logger.info(f"extracting run package contents into wrf/run at {run_path1}...")
    result = subprocess.run(['tar', 'xzf', 'wrf-run-package.tar.gz'], capture_output=True, cwd=run_path1)
    if result.returncode != 0:
        raise ProcessorRuntimeError(f"Extracting run package at {run_package_path1} failed", details={
            'stdout': result.stdout.decode('utf-8'),
            'stderr': result.stderr.decode('utf-8')
        })

    logger.info(f"delete run package at {run_package_path1}")
    os.remove(run_package_path1)

    return run_path1


def extract_simulation_period(namelist_path: str) -> (int, int):
    # start_year = 2020, 2020, 2020, 2020
    # start_month = 03, 03, 03, 03
    # start_day = 13, 13, 13, 13
    # start_hour = 22, 22, 22, 22
    # start_minute = 32, 32, 32, 32
    # start_second = 42, 42, 42, 42
    # end_year = 2020, 2020, 2020, 2020
    # end_month = 03, 03, 03, 03
    # end_day = 14, 14, 14, 14
    # end_hour = 00, 00, 00, 00
    # end_minute = 00, 00, 00, 00
    # end_second = 00, 00, 00, 00

    values = {}
    for a in ['start', 'end']:
        for b in ['year', 'month', 'day', 'hour', 'minute', 'second']:
            values[f"{a}_{b}"] = None

    with open(namelist_path, 'r') as f:
        for line in f.readlines():
            line = line.split()
            for key in values:
                if key in line:
                    values[key] = int(line[-1])

    t_from = int(datetime.datetime(
        values['start_year'], values['start_month'], values['start_day'],
        values['start_hour'], values['start_minute'], values['start_second']).timestamp())

    t_end = int(datetime.datetime(
        values['end_year'], values['end_month'], values['end_day'],
        values['end_hour'], values['end_minute'], values['end_second']).timestamp())

    return t_from, t_end


def run_exe(wd_path: str, exe_name: str, label: str, parameters: Parameters, kill_switch: KillSwitch,
            ncpus: int = 56, mem: str = "64GB", walltime: str = "23:50:00") -> None:

    # run executable
    if os.environ['USE_PBS'] == 'yes':
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
            "###NCPUS###": str(ncpus),
            "###MEM###": mem,
            "###WALLTIME###": walltime,
            "###LOG_PATH###": log_path,
            "###WD_PATH###": wd_path,
            "###COMMAND###": f"mpirun -np {ncpus} ./{exe_name}",
            "###EXITCODE_FILE###": exitcode_filename
        })

        # submit
        logger.info(f"[RUN_EXE:PBS] submitting job for {label}")
        result = subprocess.run(['qsub', job_filename], cwd=wd_path, capture_output=True)
        if result.returncode != 0:
            raise ProcessorRuntimeError(f"Submitting command '{label}' failed.", details={
                'stdout': result.stdout.decode('utf-8'),
                'stderr': result.stderr.decode('utf-8')
            })

        else:
            job_id = result.stdout.decode('utf-8').strip()
            logger.info(f"[RUN_EXE:PBS] job id: {job_id}")
            kill_switch.set(job_id)

        # wait until exitcode file is available
        logger.info(f"[RUN_EXE:PROC] waiting for job to finish...")
        while True:
            # was the kill switch activated?
            if kill_switch.activated:
                return

            if os.path.isfile(exitcode_path):
                logger.info(f"[RUN_EXE:PBS] exitcode file found at {exitcode_path}")
                break
            else:
                time.sleep(5)

        # read the exit code
        with open(exitcode_path, 'r') as f:
            line = f.readline()
            logger.info(f"[RUN_EXE:PBS] exitcode file content: {line}")

            exitcode = int(line)
            if exitcode != 0:
                raise ProcessorRuntimeError(f"Non-zero exitcode ({exitcode}) when running command {label}")

    else:
        # determine number of processes
        ncpus = min(os.cpu_count(), ncpus)
        ncpus = ncpus - (ncpus % 2)
        logger.info(f"[RUN_EXE:PROC] number of processes: {ncpus} out of {os.cpu_count()}")

        # start the process
        logger.info(f"[RUN_EXE:PROC] starting process for {label}")
        process = subprocess.Popen(['mpirun', '-np', ncpus, f'./{exe_name}'], cwd=wd_path)

        # obtain the PID and set the killswitch
        logger.info(f"[RUN_EXE:PROC] process id: {process.pid}")
        kill_switch.set(process.pid)

        # wait for the process to finish
        logger.info(f"[RUN_EXE:PROC] waiting for process to return...")
        process.wait()

        if process.returncode != 0:
            raise ProcessorRuntimeError(f"Non-zero exitcode ({process.returncode}) when running command {label}")


def has_run_successfully_completed(run_path: str, t_start: int, t_end: int, p_start: int, p_end: int,
                                   callback: ProgressListener = None) -> bool:
    def wrf_timestamp_to_seconds(timestamp: str) -> int:
        # example: 2020-03-14_00:00:00
        timestamp = timestamp.split('_')
        date = timestamp[0].split('-')
        time = timestamp[1].split(':')
        timestamp = datetime.datetime(int(date[0]), int(date[1]), int(date[2]),
                                      int(time[0]), int(time[1]), int(time[2])).timestamp()
        return int(timestamp)

    # look for something like this: 'd01 2020-01-01_00:15:18 wrf: SUCCESS COMPLETE WRF'
    # get all files that match the pattern and use the first one
    rsl_file = os.path.join(run_path, 'rsl.error.0000')

    # do we have any files to begin with? if the simulation hasn't started yet, then we don't
    if not os.path.isfile(rsl_file):
        return False

    # if WRF successfully completed the run then all RSL output files should contain the message,
    # so just pick one file and search for it.
    latest_timing = None
    success = False
    with open(rsl_file, 'r') as f:
        for line in f.readlines():
            # check for successful completion...
            if 'SUCCESS COMPLETE WRF' in line:
                success = True

            # but also check for issues that we cannot recover from
            elif '---- ERROR: ' in line or 'SIGSEGV, segmentation fault occurred' in line:
                raise ProcessorRuntimeError("Error encountered during WRF run. See logs for more details.")

            # check for latest timing message. example:
            # Timing for main: time 2020-03-14_00:00:00 on domain   3:    2.68046 elapsed seconds
            elif 'Timing for main' in line:
                line = line.split(' ')
                latest_timing = line[4]

    # update the progress
    if latest_timing is not None:
        t = wrf_timestamp_to_seconds(latest_timing)
        p = int(p_start + ((t - t_start) / (t_end - t_start))*(p_end - p_start))
        if callback:
            callback.on_progress_update(p)
        logger.info(f"progress: {p}%")

    return success


def collect_wrf_files(run_path: str, prefix: str) -> (dict, list):
    # get all files that match the pattern
    output_files = {}
    files = glob.glob(os.path.join(run_path, f"{prefix}_*"))
    for file in files:
        # split filename into its components
        fields = os.path.basename(file).split('_')
        domain = fields[1]
        date = fields[2]
        time = fields[3]
        timestamp = int(re.sub('[^0-9]', '', f"{date}{time}"))

        if timestamp not in output_files:
            output_files[timestamp] = {}

        output_files[timestamp][domain] = file
        output_files[timestamp]['date'] = date.split('-')
        output_files[timestamp]['time'] = time.split(':')

    # now get all the unique timestamps in order
    timestamps = sorted([*output_files.keys()])

    return output_files, timestamps


def collect_wrfout_files(run_path: str) -> (dict, list):
    return collect_wrf_files(run_path, 'wrfout')


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


def extract_combined_dataset_specs(domain: str, c: dict, wrfout_files: dict, wrfout_timestamps: list) -> dict:
    # determine the reference bounding box, shape and timestamps of the available data set for this component
    ref_shape = None
    ref_timestamps = []
    ref_bounding_box = None
    for wrfout_t in wrfout_timestamps:
        # open the netcdf file
        with open_netcdf(wrfout_files[wrfout_t][domain], 'r') as nc_file:
            # determine min/max latitude and longitude
            if os.environ['USE_NETCDF_VERSION'] == "3":
                lat = nc_file.variables['XLAT'].data
                lon = nc_file.variables['XLONG'].data
            else:
                lat = nc_file.variables['XLAT']
                lon = nc_file.variables['XLONG']

            bounding_box = [np.min(lat), np.max(lat), np.min(lon), np.max(lon)]
            # logger.info(f"wrfout_files[{wrfout_t}][{domain}] bounding_box: {bounding_box}")

            # do we already have the bounding box? if so, do some checks...
            if ref_bounding_box is None:
                ref_bounding_box = bounding_box
                logger.debug(f"bounding box: {ref_bounding_box}")

            elif ref_bounding_box != bounding_box:
                raise ProcessorRuntimeError(f"Mismatching bounding box encountered: "
                                            f"ref={ref_bounding_box} mismatch={bounding_box} wrfout_t={wrfout_t} ")

            # add on to the reference timestamps
            times = nc_file.variables['Times']
            for t in times:
                t = "".join(c.decode('utf-8') for c in t)
                t = int(re.sub('[^0-9]', '', t))
                ref_timestamps.append(t)

            # get the variable (T2 in case of WBTG for purpose of obtaining the shape)
            if c['name'] == 'wet_bulb_globe_temperature':
                variable = nc_file.variables['T2'][0]
            else:
                variable = wrf.getvar(nc_file, c['variable'])
                variable = variable if c['index'] is None else variable[c['index']]

            # do we already have the shape? if so, do some checks and update
            ref_shape = (times.shape[0], *variable.shape) if ref_shape is None else \
                (ref_shape[0] + times.shape[0], *variable.shape)
            logger.debug(f"updated shape: ref={ref_shape} timesteps={times.shape[0]}")

    return {
        'shape': ref_shape,
        'timestamps': ref_timestamps,
        'timezone': 'UTC',
        'bounding_box': ref_bounding_box
    }


def calculate_wbgt(t: str, lat, lon, t2, rh2, psfc, speed, swdown) -> np.ndarray:
    # determine year month day and hour
    t = ''.join([char.decode('utf-8') for char in t])
    year = int(t[0:4])
    month = int(t[5:7])
    day = int(t[8:10])
    hour = int(t[11:13])

    ny = lat.shape[0]
    nx = lat.shape[1]

    result = np.zeros_like(t2)
    for j in range(ny):
        for i in range(nx):
            lat_ji = lat[j, i]
            lon_ji = lon[j, i]

            # make sure cza and fdir is calculated before wbgt
            cza = calc_cza(lat_ji, lon_ji, year, month, day, hour)
            fdir = calc_fdir(year, month, day, lat_ji, lon_ji, swdown[j, i], cza)

            # wbgt(year, mon, day, lat, lon, solar, cza, fdir, pres, Tair, relhum, speed)
            estimate = wbgt(year=year, month=month, day=day, hr=hour, lat=lat_ji, lon=lon_ji,
                            solar=swdown[j, i], cza=cza, fdir=fdir, pres=psfc[j, i] * 0.01, Tair=t2[j, i] - 273.15,
                            relhum=rh2[j, i], speed=speed[j, i])
            result[j, i] = estimate

    return result


def create_combined_dataset(domain: str, c: dict, wrfout_files: dict,
                            wrfout_timestamps: list, shape: tuple) -> np.ndarray:
    # determine the complete data set (i.e., concatenating individual data)
    c_data = np.zeros(shape=shape, dtype=np.float32)
    c_idx = 0
    for wrfout_t in wrfout_timestamps:
        # open the netcdf file
        with open_netcdf(wrfout_files[wrfout_t][domain], 'r') as nc_file:
            # in case of WBGT, do the calculation manually
            if c['name'] == 'wet_bulb_globe_temperature':
                lat = nc_file.variables['XLAT'][:]
                lon = nc_file.variables['XLONG'][:]
                t2 = nc_file.variables['T2'][:]
                q2 = nc_file.variables['Q2'][:]
                psfc = nc_file.variables['PSFC'][:]
                uwnd = nc_file.variables['U10'][:]
                vwnd = nc_file.variables['V10'][:]
                swdown = nc_file.variables['SWDOWN']
                speed = np.sqrt(np.square(uwnd) + np.square(vwnd))

                # calculate relative humidity from temperature, pressure and mixing ratio constants for relative
                # humidity calculation
                svp1 = 611.2
                svp2 = 17.67
                svp3 = 29.65
                svpt0 = 273.15
                eps = 0.622
                rh2 = 1.E2 * (psfc * q2 / (q2 * (1. - eps) + eps)) / (svp1 * np.exp(svp2 * (t2 - svpt0) / (t2 - svp3)))

                # iterate over all timesteps
                times = nc_file.variables['Times']
                for idx in range(times.shape[0]):
                    variable = calculate_wbgt(times[idx], lat[idx], lon[idx], t2[idx],
                                              rh2[idx], psfc[idx], speed[idx], swdown[idx])

                    # copy data
                    np.copyto(c_data[c_idx], variable)
                    c_idx += 1

            # otherwise use the WRF Python diagnostics
            else:
                # iterate over all timesteps
                times = nc_file.variables['Times']
                for idx in range(times.shape[0]):
                    # get variable
                    # source: https://wrf-python.readthedocs.io/en/latest/diagnostics.html
                    variable = wrf.getvar(nc_file, c['variable'], timeidx=idx)
                    variable = variable if c['index'] is None else variable[c['index']]

                    # copy data
                    np.copyto(c_data[c_idx], variable)
                    c_idx += 1

    # perform conversion?
    if c['conversion'] is not None:
        c_data = c['conversion'](c_data)

    return c_data


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


def determine_uhi_masks(lu_path: str, wrfrun_vv_path: str) -> (np.ndarray, np.ndarray, BoundingBox, Dimensions):
    def dfs(matrix, visited, i, j, label):
        stack = [(i, j)]
        cluster_size = 0
        while stack:
            x, y = stack.pop()
            if visited[x, y] != 0:
                continue
            visited[x, y] = label
            cluster_size += 1
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < matrix.shape[0] and 0 <= new_y < matrix.shape[1]:
                    if matrix[new_x, new_y] == 1 and visited[new_x, new_y] == 0:
                        stack.append((new_x, new_y))
        return cluster_size

    def largest_cluster(matrix):
        visited = np.zeros_like(matrix, dtype=np.int32)
        label = 1
        max_cluster = 0
        max_label = 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] == 1 and visited[i, j] == 0:
                    cluster_size = dfs(matrix, visited, i, j, label)
                    if cluster_size > max_cluster:
                        max_cluster = cluster_size
                        max_label = label
                    label += 1
        mask = (visited == max_label).astype(np.int32)
        return mask

    with rasterio.Env():
        with rasterio.open(lu_path) as src:
            lu = src.read(1)

            # determine bounding box
            bbox = src.bounds
            bbox = BoundingBox(west=bbox.left, east=bbox.right, north=bbox.top, south=bbox.bottom)

            # determine nature mask (LCZ A-B, i.e., values 1-2)
            ref = np.zeros(shape=lu.shape)
            ref[(lu >= 1) & (lu <= 2)] = 1

            # identify the largest ref/nature cluster
            ref = largest_cluster(ref)

            # determine urban mask (LCZ 1-10 i.e., values 31-40)
            urban = np.zeros(shape=lu.shape)
            urban[(lu >= 31) & (lu <= 40)] = 1

            ref_tiff_path = os.path.join(wrfrun_vv_path, 'uhi_mask_ref.tiff')
            export_as_tiff(ref, ref_tiff_path, bbox, False, np.int8)

            urban_tiff_path = os.path.join(wrfrun_vv_path, 'uhi_mask_urban.tiff')
            export_as_tiff(urban, urban_tiff_path, bbox, False, np.int8)

            ref = np.flipud(ref)
            urban = np.flipud(urban)

        return ref, urban, bbox, Dimensions(height=lu.shape[0], width=lu.shape[1])


def process_output_data(domain: str, run_path: str, lu_path: str, hdf5_path: str,
                        vv_path, components: List[dict]) -> None:
    # collect WRF output files
    wrfout_files, wrfout_timestamps = collect_wrfout_files(run_path)
    logger.debug(f"[POSTPROCESS] wrfout files found for timestamps: {[*wrfout_files.keys()]}")

    # determine the reference area for UHI calculation
    ref_mask, urban_mask, _, _ = determine_uhi_masks(lu_path, vv_path)

    # write the HDF5 and VV output
    logger.info(f"[POSTPROCESS] writing HDF5 output to {hdf5_path}...")
    with h5py.File(hdf5_path, "w") as f:
        for c in components:
            # extract information that describes the combined data set
            specs = extract_combined_dataset_specs(domain, c, wrfout_files, wrfout_timestamps)
            logger.info(f"[POSTPROCESS] {c['variable']} specs: shape={specs['shape']} "
                        f"timestamps={specs['timestamps']} bounding_box={specs['bounding_box']}")

            # create the combined data set
            c_data = create_combined_dataset(domain, c, wrfout_files, wrfout_timestamps, specs['shape'])
            # logger.info(f"[POSTPROCESS] {c['variable']} data: shape={c_data.shape} content=\n{c_data}")

            # is it the UHI component? if so, then correct the temperature values by the nature ref temps
            if c['name'] == '2m_air_temperature_uhi':
                for idx, t in enumerate(specs['timestamps']):
                    # calculate the average air temperature (AT) in nature
                    masked_data = c_data[idx][ref_mask == 1]
                    avg_ref_at = np.mean(masked_data)

                    c_data[idx][urban_mask == 0] = 999
                    c_data[idx][urban_mask == 1] -= avg_ref_at

            # create the output data set
            dataset = f.create_dataset(c['name'], data=c_data, track_times=False)
            dataset.attrs['unit'] = c['unit']
            dataset.attrs['shape'] = specs['shape']
            dataset.attrs['timestamps'] = specs['timestamps']
            dataset.attrs['timezone'] = specs['timezone']
            dataset.attrs['bounding_box'] = specs['bounding_box']

            # determine bounding box
            bbox = BoundingBox(
                south=specs['bounding_box'][0], north=specs['bounding_box'][1],
                west=specs['bounding_box'][2], east=specs['bounding_box'][3]
            )

            # write vv tiffs
            for idx, t in enumerate(specs['timestamps']):
                filename = f"{c['name']}_{t}UTC.tiff"
                logger.info(f"[POSTPROCESS] write {filename}")
                tiff_out_path = os.path.join(vv_path, filename)
                export_as_tiff(c_data[idx], tiff_out_path, bbox, True, np.float32)


def postprocess(run_path: str, wd_path: str, callback: ProgressListener = None) -> None:
    # prepare the vv-contents directory
    vv_contents_path = os.path.join(wd_path, 'vv-contents')
    logger.info(f"[POSTPROCESS] create vv-contents path at {vv_contents_path}")
    os.makedirs(vv_contents_path, exist_ok=True)

    # copy the input vv package
    vv_package_path0 = os.path.join(wd_path, 'wrf-prep-vv-package')
    vv_package_path1 = os.path.join(wd_path, 'vv-contents', 'wrf-prep-vv-package')
    shutil.copy(vv_package_path0, vv_package_path1)

    # unpack the wrf-prep-vv-package and delete the archive
    result = subprocess.run(['tar', 'xzf', 'wrf-prep-vv-package'], cwd=vv_contents_path, capture_output=True)
    if result.returncode != 0:
        raise ProcessorRuntimeError(f"Failed to unpack wrf-prep-vv-package")
    os.remove(vv_package_path1)

    # create the wrfrun vv path
    wrfrun_vv_path = os.path.join(vv_contents_path, 'wrfrun')
    logger.info(f"[POSTPROCESS] create vv-wrfrun path at {wrfrun_vv_path}")
    os.makedirs(wrfrun_vv_path, exist_ok=True)

    # post-process the near-surface climate output
    lu_path = os.path.join(vv_contents_path, 'real', 'LU.d04.tiff')
    d04_nsc_path = os.path.join(wd_path, 'd04-near-surface-climate')
    components = [
        {
            'name': '2m_air_temperature',
            'variable': 'T2',
            'index': None,
            'unit': '˚C',
            'conversion': lambda x: x - 273.15
        },
        {
            'name': '2m_air_temperature_uhi',
            'variable': 'T2',
            'index': None,
            'unit': 'Δ˚C',
            'conversion': lambda x: x - 273.15,
        },
        {
            'name': '2m_relative_humidity',
            'variable': 'rh2',
            'index': None,
            'unit': '%',
            'conversion': None
        },
        {
            'name': 'wet_bulb_globe_temperature',
            'variable': None,
            'index': None,
            'unit': '˚C',
            'conversion': None
        },
        {
            'name': '10m_wind_speed',
            'variable': 'wspd_wdir10',
            'index': 0,
            'unit': 'm/s',
            'conversion': None
        },
        {
            'name': '10m_wind_direction',
            'variable': 'wspd_wdir10',
            'index': 1,
            'unit': '˚',
            'conversion': None
        }
    ]
    process_output_data("d04", run_path, lu_path, d04_nsc_path, wrfrun_vv_path, components)
    if callback:
        callback.on_output_available('d04-near-surface-climate')

    # collect the log files and move them to the vv folder
    vv_logs_path = os.path.join(vv_contents_path, 'logs')
    os.makedirs(vv_logs_path, exist_ok=True)
    logger.info(f"[POSTPROCESS] collecting WRF run logs at {vv_logs_path}")
    for prefix in ["log.*", "namelist*", "rsl.*"]:
        prefixed_path = os.path.join(run_path, prefix)
        for file_path0 in glob.glob(prefixed_path):
            file_path1 = os.path.join(vv_logs_path, os.path.basename(file_path0))
            shutil.copy(file_path0, file_path1)

    # make a copy of the 'information' file in the vv folder
    information_path0 = os.path.join(wd_path, 'information')
    information_path1 = os.path.join(vv_contents_path, 'information.json')
    shutil.copy(information_path0, information_path1)

    # create vv-package
    items = os.listdir(vv_contents_path)
    vv_package_path = os.path.join(wd_path, 'vv-package')
    logger.info(f"[POSTPROCESS] creating vv-package at {vv_package_path}")
    result = subprocess.run(['tar', 'czf', vv_package_path, *items], cwd=vv_contents_path, capture_output=True)
    if result.returncode != 0:
        raise ProcessorRuntimeError(f"Creating vv-package failed: "
                                    f"stdout={result.stdout.decode('utf-8')} "
                                    f"stderr={result.stderr.decode('utf-8')}")

    if callback:
        callback.on_output_available('vv-package')


class WRFSimProcessor(ProcessorBase):
    def __init__(self, proc_path: str, ext_run_path: str = None) -> None:
        super().__init__(proc_path)

        self._ext_run_path = ext_run_path
        self._is_cancelled = False
        self._kill_switch = KillSwitch()

    def run(self, wd_path: str, callback: ProgressListener, logger: logging.Logger) -> None:
        def check_if_cancelled_and_pub_progress(progress: int) -> None:
            callback.on_progress_update(progress)
            if self._is_cancelled:
                raise ProcessorRuntimeError(f"cancelled -> exiting now.")

        # determine the processor directory
        proc_path = os.getcwd()
        logger.info(f"begin executing ucm-wrf/proc_sim: wd_path={wd_path} proc_path={proc_path} "
                    f"ext_run_path={self._ext_run_path}")
        check_if_cancelled_and_pub_progress(1)

        # check environment variables
        check_environment_variables(['WRF_DIR', 'USE_NETCDF_VERSION', 'USE_PBS'])

        # load parameters
        callback.on_message(Severity.INFO, 'Loading parameters...')
        parameters_path = os.path.join(wd_path, "parameters")
        parameters = load_parameters(parameters_path)
        logger.info(f"using parameters: {parameters}")
        check_if_cancelled_and_pub_progress(2)

        # create folders for keeping vv data
        callback.on_message(Severity.INFO, 'Creating folders for visual validation data...')
        vv_dir_path = os.path.join(wd_path, 'vv-contents')
        rm_mk_dir(vv_dir_path)
        check_if_cancelled_and_pub_progress(3)

        # do we have an external run folder?
        if self._ext_run_path is None:
            # prepare the run folder
            callback.on_message(Severity.INFO, 'Prepare simulation run folder...')
            run_path = prepare_run_folder(wd_path, parameters)
            logger.info(f"using run_path: {run_path}")
            check_if_cancelled_and_pub_progress(4)

        else:
            run_path = os.path.join(wd_path, 'run')
            mk_symlink(self._ext_run_path, run_path)
            logger.info(f"using run_path: {run_path} -> {self._ext_run_path} EXTERNAL!")
            check_if_cancelled_and_pub_progress(4)

        # extract start/end simulation time
        callback.on_message(Severity.INFO, 'Extracting simulation period...')
        namelist_path = os.path.join(run_path, 'namelist.input')
        t_start, t_end = extract_simulation_period(namelist_path)
        logger.info(f"using start/end time: {t_start} -> {t_end}")
        check_if_cancelled_and_pub_progress(5)

        # we only start our own simulation if we don't observe an external run path
        if self._ext_run_path is None:
            # start simulation
            callback.on_message(Severity.INFO, 'Running WRF...')
            thread = threading.Thread(target=run_exe, kwargs={
                'wd_path': run_path,  'exe_name': 'wrf.exe',  'label': 'wrf',
                'parameters': parameters, 'kill_switch': self._kill_switch,
                'ncpus': 56, 'mem': "64GB", 'walltime': "23:50:00"
            })
            thread.start()

        # wait for simulation to be done
        while not has_run_successfully_completed(run_path, t_start, t_end, 6, 90, callback):
            time.sleep(30)

        # perform post-processing
        callback.on_message(Severity.INFO, 'Post-processing simulation results...')
        postprocess(run_path, wd_path, callback)
        check_if_cancelled_and_pub_progress(98)

        # clean-up
        callback.on_message(Severity.INFO, 'Cleaning up...')
        shutil.rmtree(run_path, ignore_errors=True)
        check_if_cancelled_and_pub_progress(99)

        # we are done
        callback.on_message(Severity.INFO, "Done (ucmwrf-sim)")
        check_if_cancelled_and_pub_progress(100)

    def cancel(self) -> None:
        self._is_cancelled = True
        self._kill_switch.activate()


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
        logger.error(f"Working directory at {wd_path} does not exist!")
        sys.exit(-1)
    logger.info(f"Using working directory located at {wd_path}")

    # do we have an already existing run folder?
    if len(sys.argv) >= 3:
        run_path = sys.argv[2]
        if not os.path.isdir(run_path):
            logger.error(f"Run folder at {run_path} does not exist")
            sys.exit(-1)
        logger.info(f"Using existing run folder at {run_path}")
    else:
        run_path = None

    class LegacyListener(ProgressListener):
        def on_progress_update(self, progress: float) -> None:
            print(f"trigger:progress:{int(progress)}")
            sys.stdout.flush()

        def on_output_available(self, output_name: str) -> None:
            print(f"trigger:output:{output_name}")
            sys.stdout.flush()

        def on_message(self, severity: Severity, message: str) -> None:
            logger.info(f"msg[{severity}] {message}")
            print(f"trigger:message:{severity.value}:{message}")
            sys.stdout.flush()

    # create the processor object and run the
    callback = LegacyListener()
    try:
        logger.info(f"Attempting to run the WRF simulator...'")
        proc = WRFSimProcessor(proc_path, ext_run_path=run_path)
        proc.run(wd_path, callback, logger)
        logger.info("Done!")
        sys.exit(0)

    except ProcessorRuntimeError as e:
        logger.error(f"Exception {e.id}: {e.reason}\ndetails: {e.details}")
        callback.on_message(Severity.ERROR, e.reason)

        sys.exit(-2)
