import shutil
import subprocess
import sys

import json
import os
import tarfile
import time
import traceback
from typing import List, Tuple, Dict, Optional, Callable

import gmsh
import logging

import h5py
import numpy as np
import pyvista as pv
import pyproj
import rasterio

from pydantic import BaseModel
from rasterio.transform import from_origin
from saas.core.logging import Logging
from saas.sdk.adapter import print_progress, print_output
from shapely.errors import TopologicalError
from shapely.geometry.geo import shape, mapping
from shapely.geometry.polygon import Polygon
from shapely.ops import transform
from scipy.interpolate import griddata

logger = Logging.get('ucm-scout-prep', level=logging.DEBUG)


class BlockMeshDict(BaseModel):
    min_coords: Tuple[float, float, float]
    max_coords: Tuple[float, float, float]
    buffer_min_coords: Tuple[float, float, float]
    buffer_max_coords: Tuple[float, float, float]
    offsets: Tuple[float, float]

    def location_in_mesh(self) -> str:
        mid = tuple((min_c + max_c) / 2 for min_c, max_c in zip(self.min_coords, self.max_coords))
        return f"({mid[0]:.6f} {mid[1]:.6f} {mid[2]:.6f})"

    def minx(self) -> float:
        return self.min_coords[0]

    def miny(self) -> float:
        return self.min_coords[1]

    def minz(self) -> float:
        return self.min_coords[2]

    def maxx(self) -> float:
        return self.max_coords[0]

    def maxy(self) -> float:
        return self.max_coords[1]

    def maxz(self) -> float:
        return self.max_coords[2]

    def buffer_minx(self) -> float:
        return self.buffer_min_coords[0]

    def buffer_miny(self) -> float:
        return self.buffer_min_coords[1]

    def buffer_minz(self) -> float:
        return self.buffer_min_coords[2]

    def buffer_maxx(self) -> float:
        return self.buffer_max_coords[0]

    def buffer_maxy(self) -> float:
        return self.buffer_max_coords[1]

    def buffer_maxz(self) -> float:
        return self.buffer_max_coords[2]

    def offset_x(self) -> float:
        return self.offsets[0]

    def offset_y(self) -> float:
        return self.offsets[1]


def replace_in_file(file_path: str, mapping: Dict[str, str]) -> None:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    for key, value in mapping.items():
        content = content.replace(key, value)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def load_buildings(geojson_path: str, xy_buffer: float, z_buffer: float) -> Tuple[List[dict], BlockMeshDict, Callable]:
    # Read GeoJSON
    with open(geojson_path) as f:
        geojson_data = json.load(f)

    features = geojson_data.get('features', [])
    if not features:
        return [], (0, 0), (0, 0)

    # WGS84 (lat/lon)
    wgs84 = pyproj.CRS('EPSG:4326')

    # Compute bounding box of features
    minx = miny = float('inf')
    maxx = maxy = float('-inf')
    maxz = minz = 0.0
    for feat in features:
        geom = shape(feat['geometry'])
        bxmin, bymin, bxmax, bymax = geom.bounds
        minx = min(minx, bxmin)
        miny = min(miny, bymin)
        maxx = max(maxx, bxmax)
        maxy = max(maxy, bymax)
        maxz = max(maxz, feat['properties']['height'])

    # Approximate center in lon/lat
    center_lon = (minx + maxx) / 2
    center_lat = (miny + maxy) / 2

    # UTM zone from center
    utm_zone = int((center_lon + 180) / 6) + 1
    is_northern = center_lat >= 0

    utm_crs = pyproj.CRS.from_dict({
        'proj': 'utm',
        'zone': utm_zone,
        'south': not is_northern
    })

    # Coordinate transformers
    to_utm = pyproj.Transformer.from_crs(wgs84, utm_crs, always_xy=True).transform
    to_wgs84 = pyproj.Transformer.from_crs(utm_crs, wgs84, always_xy=True).transform

    # Reproject features to UTM GeoJSON
    reprojected_features: List[dict] = []
    for feat in features:
        geom = shape(feat['geometry'])
        geom_utm = transform(to_utm, geom)
        new_feat = {
            'type': 'Feature',
            'geometry': mapping(geom_utm),
            'properties': feat.get('properties', {}),
        }
        # Optionally preserve 'id' if present
        if 'id' in feat:
            new_feat['id'] = feat['id']
        reprojected_features.append(new_feat)

    # Re-compute x/y bounding box of features
    minx = miny = float('inf')
    maxx = maxy = float('-inf')
    for feat in reprojected_features:
        geom = shape(feat['geometry'])
        bxmin, bymin, bxmax, bymax = geom.bounds
        minx = min(minx, bxmin)
        miny = min(miny, bymin)
        maxx = max(maxx, bxmax)
        maxy = max(maxy, bymax)

    # determine deltas
    delta_x = maxx - minx
    delta_y = maxy - miny
    delta_z = maxz - minz

    # determine the offset
    offset_x = minx - xy_buffer
    offset_y = miny - xy_buffer

    # determine min/max XYZ
    minx = xy_buffer
    miny = xy_buffer
    minz = 0
    maxx = minx + delta_x
    maxy = miny + delta_y
    maxz = minz + delta_z

    # determine the buffer min/max XYZ
    buffer_minx = 0
    buffer_miny = 0
    buffer_minz = 0
    buffer_maxx = delta_x + 2*xy_buffer
    buffer_maxy = delta_y + 2*xy_buffer
    buffer_maxz = delta_z + z_buffer

    # Inverse-transform bounding boxes to WGS84
    def bbox_to_wgs84(minx, miny, maxx, maxy):
        lon_min, lat_min = to_wgs84(minx + offset_x, miny + offset_y)
        lon_max, lat_max = to_wgs84(maxx + offset_x, maxy + offset_y)
        return (lon_min, lat_min, lon_max, lat_max)

    feature_bbox_deg = bbox_to_wgs84(minx, miny, maxx, maxy)
    buffer_bbox_deg = bbox_to_wgs84(buffer_minx, buffer_miny, buffer_maxx, buffer_maxy)

    print(f"Feature bounding box: "
          f"west={feature_bbox_deg[0]:.6f}, east={feature_bbox_deg[2]:.6f}, "
          f"south={feature_bbox_deg[1]:.6f}, north={feature_bbox_deg[3]:.6f}")

    print(f"Buffered bounding box: "
          f"west={buffer_bbox_deg[0]:.6f}, east={buffer_bbox_deg[2]:.6f}, "
          f"south={buffer_bbox_deg[1]:.6f}, north={buffer_bbox_deg[3]:.6f}")

    # Return GeoJSON features, and BlockMeshDict
    return reprojected_features, BlockMeshDict(
        min_coords=(minx, miny, minz), max_coords=(maxx, maxy, maxz),
        buffer_min_coords=(buffer_minx, buffer_miny, buffer_minz),
        buffer_max_coords=(buffer_maxx, buffer_maxy, buffer_maxz),
        offsets=(offset_x, offset_y)
    ), to_wgs84


def simplify_geometries(
        features: List[dict], tolerance: int = 5, to_wgs84: Optional[Callable] = None,
        geojson_path: Optional[str] = None
) -> List[Tuple[Polygon, float]]:

    total_poly_points = 0
    feature_poly_points = []

    simplified_features: List[dict] = []
    simplified_polygons: List[Tuple[Polygon, float]] = []
    for feature in features:
        geometry = feature['geometry']
        properties = feature['properties']

        # determine height information
        height = float(properties.get('height', 0))
        if height == 0:
            print(f"Feature {properties.get('id', 'unknown')}/{properties.get('name', 'unknown')} has no height -> skipping")
            continue

        # in case of multi-polygon: just take first polygon
        if geometry['type'] == 'MultiPolygon':
            geometry = {
                'type': 'Polygon',
                'coordinates': geometry['coordinates'][0]
            }

        # get the shapely geometry
        try:
            polygon_geom = shape(geometry)
        except (ValueError, TopologicalError) as e:
            print(f"Invalid geometry for feature {properties.get('id', 'unknown')} -> skipping")
            continue

        # we should be dealing with a polygon
        if not isinstance(polygon_geom, Polygon):
            print(f"Feature {properties.get('id', 'unknown')} is not a Polygon -> skipping")
            continue

        # Simplify geometry here (tolerance value in coordinate units)
        simplified_polygon: Polygon = polygon_geom.simplify(tolerance, preserve_topology=True)
        exterior_coords = list(simplified_polygon.exterior.coords)

        # generate simplified GeoJSON features
        simplified_feature = {
            "type": "Feature",
            "geometry": mapping(transform(to_wgs84, simplified_polygon)),
            "properties": properties
        }
        simplified_features.append(simplified_feature)

        # remove last coord if same as first (gmsh doesn't need it repeated)
        if exterior_coords[0] == exterior_coords[-1]:
            exterior_coords = exterior_coords[:-1]

        # skip if it's not at least a triangle
        n = len(exterior_coords)
        total_poly_points += n
        feature_poly_points.append(n)
        if n < 3:
            print(f"Simplified geometry for feature {properties.get('id', 'unknown')} too small -> skipping")
            continue

        simplified_polygons.append((simplified_polygon, height))

    print(f"Total number of polygon points: {total_poly_points}")
    print(f"Number of polygon points by feature: {feature_poly_points}")

    if geojson_path is not None:
        with open(geojson_path, "w") as f:
            geojson_data = {
                "type": "FeatureCollection",
                "features": simplified_features
            }
            json.dump(geojson_data, f, indent=2)

    return simplified_polygons

def convert_polygons_to_stl(
        polygons: List[Tuple[Polygon, float]],
        stl_path: str,
        bmd_specs: BlockMeshDict,
) -> None:
    gmsh.initialize()

    minx = miny = float('inf')
    maxx = maxy = float('-inf')
    maxz = minz = 0.0
    for polygon, height in polygons:
        # Get the coordinates of the exterior polygon and remove the last point if it duplicates the first
        exterior_coords = polygon.exterior.coords
        if exterior_coords[0] == exterior_coords[-1]:
            exterior_coords = exterior_coords[:-1]

        n = len(exterior_coords)

        # Get the points at ground level (z=0) and at the top of the building (z=height)
        points0 = []
        pointsH = []
        for coord in exterior_coords:
            x = coord[0] - bmd_specs.offset_x()
            y = coord[1] - bmd_specs.offset_y()

            minx = min(minx, x)
            miny = min(miny, y)
            maxx = max(maxx, x)
            maxy = max(maxy, y)
            maxz = max(maxz, height)

            points0.append(gmsh.model.geo.add_point(x, y, 0, 0))
            pointsH.append(gmsh.model.geo.add_point(x, y, height, 0))

        # Connect the points with lines
        lines0 = []
        linesH = []
        linesV = []
        for i in range(n):
            j = (i + 1) % n
            lines0.append(gmsh.model.geo.add_line(points0[i], points0[j]))
            linesH.append(gmsh.model.geo.add_line(pointsH[i], pointsH[j]))
            linesV.append(gmsh.model.geo.add_line(points0[i], pointsH[i]))

        # Create side surfaces
        for i in range(n):
            j = (i + 1) % n
            face = gmsh.model.geo.add_curve_loop([lines0[i], linesV[j], -linesH[i], -linesV[i]])
            gmsh.model.geo.add_plane_surface([face])

        # Create bottom and top surfaces
        face0 = gmsh.model.geo.add_curve_loop(lines0)
        faceH = gmsh.model.geo.add_curve_loop(linesH)
        gmsh.model.geo.add_plane_surface([face0])
        gmsh.model.geo.add_plane_surface([faceH])

    # check if the bounds do not violate the BMD specs
    if minx < bmd_specs.minx() or maxx > bmd_specs.maxx() or \
        miny < bmd_specs.miny() or maxy > bmd_specs.maxy() or \
        minz < bmd_specs.minz() or maxz > bmd_specs.maxz():
        raise RuntimeError("BMD specs violated")

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write(stl_path)
    gmsh.finalize()


class Job:
    def __init__(self, case_name: str, case_path: str) -> None:
        self._case_name = case_name
        self._case_path = case_path

        self.job_id = None
        self.status = None
        self.t_submit = None
        self.t_running = None
        self.t_queueing = None
        self.t_finished = None
        self.t_cancel = None

    def submit(self) -> str:
        self.t_submit = int(time.time())

        # check if the PBS file exists
        pbs_path = os.path.join(self._case_path, 'pbs.sh')
        if not os.path.isfile(pbs_path):
            raise RuntimeError(f"PBS file not found at {pbs_path}")

        # submit job
        logger.info(f"submitting job: pbs_path={pbs_path}")
        result = subprocess.run(['qsub', 'pbs.sh'], cwd=self._case_path, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"Job submission failed: {result}")

        # extract the pbs job id and start monitoring the job
        self.job_id = result.stdout.decode('utf-8')
        self.job_id = self.job_id.strip()
        logger.info(f"job submission successful: pbs_job_id={self.job_id}")

        return self.job_id

    def cancel(self) -> None:
        self.t_cancel = int(time.time())

        # run qdel
        result = subprocess.run(['qdel', self.job_id], cwd=self._case_path, capture_output=True)
        if result.returncode != 0:
            logger.warning(f"Running qdel on job {self.job_id} (status={self.status}) failed: {result}")

    def update_status(self) -> str:
        if self.job_id is None:
            raise RuntimeError(f"No job id found. Has the job been submitted? Has submission failed?")

        # if the job is already finished, then there is no need to check the status
        if self.status == 'F':
            return self.status

        # check the status by running qstat
        result = subprocess.run(['qstat', '-x', self.job_id], cwd=self._case_path, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"Running qstat failed: {result}")

        # extract the status and look for something like this:
        # 5985182.wlm01     restart-test     aydt                     0 Q medium
        for line in result.stdout.decode('utf-8').split('\n'):
            temp = line.split()
            if len(temp) == 6 and temp[0] == self.job_id:
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

                t_now = int(time.time())
                if status in ['Q', 'R', 'E', 'F']:
                    self.status = status

                    if status == 'Q' and self.t_queueing is None:
                        self.t_queueing = t_now

                    elif status in ['R', 'E'] and self.t_running is None:
                        self.t_running = t_now

                    elif status == 'F' and self.t_finished is None:
                        self.t_finished = t_now

                    return self.status

                else:
                    logger.warning(f"Unexpected status code '{status}' encountered: {temp}")

        # seems like we haven't found that job at all
        raise RuntimeError(f"No job with id {self.job_id} found.")


def update_progress(case_path: str, max_iterations: int, p_start: int, p_end: int) -> None:
    # do we already have the log?
    log_path = os.path.join(case_path, 'log.unsteadyThermalFoam')
    if not os.path.isfile(log_path):
        return

    # parse the log to get the latest timing
    latest_timing = None
    with open(log_path, 'r') as f:
        for line in f.readlines():
            # check for latest timing message. example:
            # Time = 416
            if line.startswith('Time = '):
                line = line.split(' = ')
                latest_timing = int(line[1])

    # update the progress
    if latest_timing is not None:
        p = int(p_start + (latest_timing / max_iterations)*(p_end - p_start))
        print_progress(p)


def run_simulation(case_path: str, case_name: str, end_time: int, p_start: int, p_end: int) -> None:
    # submit the simulation job
    logger.info(f"submit simulation job: case_name={case_name}")
    job = Job(case_name, case_path)
    job.submit()

    while True:
        # sleep for a while...
        time.sleep(30)

        # update progress
        update_progress(case_path, end_time, p_start, p_end)

        # update job status. possible results:
        # Q  Job is queued
        # F  Job is finished
        # R  Job is running
        s = job.update_status()

        if s == 'Q':
            logger.info(f"job is queueing...")

        elif s == 'R':
            logger.info(f"job is running...")

        elif s == 'F':
            logger.info(f"job finished!")
            break

        else:
            logger.warning(f"unexpected job status: {s}")


def convert_vtp_to_geotiff(
        vtp_path: str, geotiff_path: str, variable: str, bbox: dict, bmd_specs: BlockMeshDict,
        conversion = None
) -> np.ndarray:
    # read the mesh from the VTP file
    mesh = pv.read(vtp_path)

    # Extract coordinates and field
    points = mesh.points  # shape (n, 3)
    values = mesh[variable]  # temperature field, shape (n,)

    # Assume known bounds of original coordinates
    x_min, x_max = bmd_specs.minx(), bmd_specs.maxx()
    y_min, y_max = bmd_specs.miny(), bmd_specs.maxy()

    # Target geographic bounds
    lon_min, lon_max = bbox['west'], bbox['east']
    lat_min, lat_max = bbox['south'], bbox['north']

    # Linear interpolation from unitless to geographic coordinates
    x, y = points[:, 0], points[:, 1]
    lon = lon_min + (x - x_min) / (x_max - x_min) * (lon_max - lon_min)
    lat = lat_min + (y - y_min) / (y_max - y_min) * (lat_max - lat_min)

    n_cols = (lon_max - lon_min) / 0.00001
    n_rows = (lat_max - lat_min) / 0.00001
    n_cols = int(n_cols)
    n_rows = int(n_rows)

    grid_lon, grid_lat = np.meshgrid(
        np.linspace(lon_min, lon_max, n_cols),
        np.linspace(lat_min, lat_max, n_rows)
    )

    # Interpolate to grid
    grid_var = griddata((lon, lat), values, (grid_lon, grid_lat), method='linear')

    # convert if necessary
    if conversion is not None:
        grid_var = conversion(grid_var)

    # flip upside down (for GeoTIFF)
    grid_var = np.flipud(grid_var)

    # Define transform (top-left origin)
    res_lon = (lon_max - lon_min) / n_cols
    res_lat = (lat_max - lat_min) / n_rows
    transform = from_origin(lon_min, lat_max, res_lon, res_lat)

    # Save to GeoTIFF
    with rasterio.open(
            geotiff_path,
            "w",
            driver="GTiff",
            height=n_rows,
            width=n_cols,
            count=1,
            dtype=grid_var.dtype,
            crs="EPSG:4326",
            transform=transform,
    ) as dst:
        dst.write(grid_var, 1)

    # flip it back
    grid_var = np.flipud(grid_var)

    return grid_var


def kelvin_to_celsius(values: np.ndarray) -> np.ndarray:
    return values - 273.15


def function(wd_path: str):
    try:
        print_progress(0)

        # check if the required env variables are defined
        required = ['CASE_TEMPLATE_PATH']
        for name in required:
            if os.environ.get(name) is None:
                raise RuntimeError(f"Required environment variable {name} not defined.")

        # read parameters and determine bounding box
        parameters_path = os.path.join(wd_path, "parameters")
        with open(parameters_path, "r") as f:
            parameters = json.load(f)

        # make a copy of the case template directory
        case_src_path = os.environ['CASE_TEMPLATE_PATH']
        case_path = os.path.join(wd_path, 'case')
        shutil.copytree(case_src_path, case_path, dirs_exist_ok=True)
        print_progress(1)

        # define paths
        constant_path = os.path.join(case_path, 'constant')
        system_path = os.path.join(case_path, 'system')
        zero_path = os.path.join(case_path, '0')

        # load buildings and obtain BMD specs
        xy_buffer = 50.0
        z_buffer = 25.0
        buildings_geojson_path = os.path.join(wd_path, 'building-footprints')
        buildings, bmd_specs, to_wgs84 = load_buildings(buildings_geojson_path, xy_buffer, z_buffer)
        print_progress(2)

        # update vertices in system/blockMeshDict
        replace_in_file(os.path.join(system_path, 'blockMeshDict'), {
            '###X_MIN###': f"{bmd_specs.buffer_minx():.6f}",
            '###Y_MIN###': f"{bmd_specs.buffer_miny():.6f}",
            '###Z_MIN###': f"{bmd_specs.buffer_minz():.6f}",
            '###X_MAX###': f"{bmd_specs.buffer_maxx():.6f}",
            '###Y_MAX###': f"{bmd_specs.buffer_maxy():.6f}",
            '###Z_MAX###': f"{bmd_specs.buffer_maxz():.6f}"
        })
        print_progress(3)

        # update location_in_mesh: e.g., '(40101.500000 32101.500000 115.700000)'
        location_in_mesh = bmd_specs.location_in_mesh()
        replace_in_file(os.path.join(system_path, 'snappyHexMeshDict'), {'###LOCATION_IN_MESH###': location_in_mesh})
        print_progress(4)

        # simplify polygons
        simplified_buildings_geojson_path = os.path.join(wd_path, 'building-footprints-simplified')
        polygons = simplify_geometries(buildings, to_wgs84=to_wgs84, geojson_path=simplified_buildings_geojson_path)
        print_progress(5)

        # convert the building footprints from GeoJSON to STL
        stl_path = os.path.join(constant_path, 'triSurface', 'buildings.stl')
        convert_polygons_to_stl(polygons, stl_path, bmd_specs)
        print_progress(6)

        # update STL name in various files
        file_paths = [
            *[os.path.join(system_path, f) for f in ['snappyHexMeshDict', 'surfaceFeatureExtractDict', 'predefinedSampleDict', 'snappyToposetDict', 'createRemainingPatchDict']],
            *[os.path.join(constant_path, f) for f in ['boundaryRadiationProperties']],
            *[os.path.join(zero_path, f) for f in ['U', 'treeBlanking', 'sourceBlanking', 'T', 'qv', 'qrSolar', 'p_rgh', 'p', 'nut', 'IDefault', 'alphaT']]
        ]
        for path in file_paths:
            replace_in_file(path, {'###STL_NAME###': 'buildings'})
        print_progress(7)

        # update bbox in ABLDict
        abl_dict_path = os.path.join(constant_path, 'ABLDict')
        replace_in_file(abl_dict_path, {
            '###WEST###': f"{bmd_specs.minx():.6f}",
            '###EAST###': f"{bmd_specs.maxx():.6f}",
            '###SOUTH###': f"{bmd_specs.miny():.6f}",
            '###NORTH###': f"{bmd_specs.maxy():.6f}",
            '###LOWER###': f"{bmd_specs.minz():.6f}",
            '###UPPER###': f"{bmd_specs.maxz():.6f}",
        })
        print_progress(8)

        # update bbox in dampingFile.json
        damping_file_path = os.path.join(case_path, 'dampingFile.json')
        replace_in_file(damping_file_path, {
            '###WEST###': f"{bmd_specs.minx():.6f}",
            '###EAST###': f"{bmd_specs.maxx():.6f}",
            '###SOUTH###': f"{bmd_specs.miny():.6f}",
            '###NORTH###': f"{bmd_specs.maxy():.6f}",
            '###LOWER###': f"{bmd_specs.minz():.6f}",
            '###UPPER###': f"{bmd_specs.maxz():.6f}",
        })
        print_progress(9)

        # update predefinedSampleDict
        psd_file_path = os.path.join(system_path, 'predefinedSampleDict')
        replace_in_file(psd_file_path, {
            '###STL_NAME###': 'buildings',
            '###X_MIN###': f"{bmd_specs.buffer_minx():.6f}",
            '###Y_MIN###': f"{bmd_specs.buffer_miny():.6f}",
            '###X_MAX###': f"{bmd_specs.buffer_maxx():.6f}",
            '###Y_MAX###': f"{bmd_specs.buffer_maxy():.6f}",
        })
        print_progress(10)

        # update the PBS file
        pbs_source_path = os.path.join(case_path, 'pbs.sh.template')
        pbs_path = os.path.join(case_path, 'pbs.sh')
        shutil.copyfile(pbs_source_path, pbs_path)
        replace_in_file(pbs_path, {
            '###JOBNAME###': parameters['name']
        })
        print_progress(11)

        # update the end time
        control_dict_path = os.path.join(system_path, 'controlDict')
        end_time = parameters['end_time']
        replace_in_file(control_dict_path, {
            '###END_TIME###': str(end_time)
        })
        print_progress(12)

        # run the simulation
        run_simulation(case_path, parameters['name'], end_time, 12, 98)
        print_progress(98)

        # variable settings
        variables: List[dict] = [
            {
                'name': 'T',
                'unit': 'Celsius',
                'conversion': kelvin_to_celsius,
                'out_name': 'air_temperature'
            }
        ]

        # post-process the output data
        bounding_box = [
            parameters['area']['south'], parameters['area']['north'],
            parameters['area']['west'], parameters['area']['east']
        ]
        ppd_path = os.path.join(case_path, 'postProcessing', 'predefinedSampleDict')
        hdf5_out_path = os.path.join(wd_path, 'climatic-variables')
        step_time = 600
        with h5py.File(hdf5_out_path, "w") as f_out:
            geotiff_files: List[str] = []
            for variable in variables:
                data: List[np.ndarray] = []
                timestamps: List[int] = []
                for t in range(0, end_time + 1, step_time):
                    vtp_path = os.path.join(ppd_path, str(t), 'contour_xy_1.vtp')
                    if os.path.isfile(vtp_path):
                        geotiff_path = os.path.join(case_path, 'postProcessing', f"{variable['name']}_{t}.geotiff")
                        var_data = convert_vtp_to_geotiff(
                            vtp_path, geotiff_path, variable['name'], parameters['area'], bmd_specs,
                            conversion=variable['conversion']
                        )
                        geotiff_files.append(geotiff_path)
                        data.append(var_data)
                        timestamps.append(t)

                # create the output data set
                stacked = np.stack(data, axis=0)
                data_set = f_out.create_dataset(variable['out_name'], data=stacked, track_times=False)
                data_set.attrs['unit'] = variable['unit']
                data_set.attrs['shape'] = stacked.shape
                data_set.attrs['timestamps'] = timestamps
                data_set.attrs['bounding_box'] = bounding_box
        print_output('climatic-variables')
        print_progress(99)

        # prepare the visual validation output
        archive_path = os.path.join(wd_path, 'vv-package')
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(buildings_geojson_path, arcname=os.path.basename(buildings_geojson_path))
            tar.add(simplified_buildings_geojson_path, arcname=os.path.basename(simplified_buildings_geojson_path))
            for file_path in geotiff_files:
                tar.add(file_path, arcname=os.path.basename(file_path))
        print_output('vv-package')
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
    logger = Logging.get('ucm-scout-prep', level=logging.DEBUG, custom_log_path=_logfile_path)

    return_code = function(_working_directory)
    sys.exit(return_code)
