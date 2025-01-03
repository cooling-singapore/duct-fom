import json
import os
import pathlib
import shutil
import subprocess
import sys
import logging
import tempfile
import pyproj
from collections import defaultdict
from typing import Dict, List, DefaultDict
from zipfile import ZipFile, ZIP_DEFLATED

from csv_utils import ManageBuildingTypeCSV
from building_type_utils import NewBuildingStandardOperation

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio

CEA_REPO_URL = "https://github.com/cooling-singapore/CEA_for_DUCT"
CEA_DB_NAME = 'cea_databases'
DEFAULT_BUILDING_FLOOR_HEIGHT = 3.0
DEFAULT_BUILDING_YEAR = 2020


def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def print_progress(progress: int) -> None:
    print(f"trigger:progress:{progress}")
    sys.stdout.flush()


def print_output(output: str) -> None:
    print(f"trigger:output:{output}")
    sys.stdout.flush()


def create_zip(input_path, output_path):
    with ZipFile(output_path, 'w', ZIP_DEFLATED) as zip_file:
        _input = pathlib.Path(input_path)
        for file in _input.glob("**/*"):
            zip_file.write(file, file.relative_to(_input))


def create_dataframe_mapping_from_dict(building_mapping: Dict[str, List[str]],
                                       default_building_type: str) -> DefaultDict:
    """
    Flattens (values become the key) and set default for mapping
    """
    # Set default value
    out = defaultdict(lambda: default_building_type)
    for k, v in building_mapping.items():
        for lu in v:
            out[lu] = k
    return out


def clean_building_geometries(building_geometries: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    duplicated_names = building_geometries.duplicated(subset=['name'], keep=False)

    if duplicated_names.any():
        names = building_geometries.loc[duplicated_names, "name"]
        counts = names.value_counts()
        for name in names.unique():
            building_geometries.loc[building_geometries["name"] == name, "name"] = [f"{name}_{i}" for i in
                                                                                    range(counts[name])]
    return building_geometries


def create_cea_footprints(output_directory: str, building_footprints: gpd.GeoDataFrame) -> None:
    """
    Output footprints as shapefiles in CEA format
    The input should at least have `name` and `height` property
    """
    # Create new dataframe with the required columns
    zone_building_footprints = gpd.GeoDataFrame()
    zone_building_footprints["Name"] = building_footprints["name"].astype(str).str.replace(":", "-")
    zone_building_footprints["height_ag"] = building_footprints["height"]
    zone_building_footprints["floors_ag"] = (building_footprints["height"] / DEFAULT_BUILDING_FLOOR_HEIGHT).astype(int)
    # Assume no floors below ground
    zone_building_footprints[["height_bg", "floors_bg"]] = 0
    # Set geometries
    zone_building_footprints.set_geometry(building_footprints.geometry, crs=building_footprints.crs, inplace=True)

    os.makedirs(output_directory, exist_ok=True)
    zone_building_footprints.to_file(os.path.join(output_directory, "zone.shp"))

    # Create empty surrounding buildings
    surrounding_building_footprints = gpd.GeoDataFrame(columns=["Name", "height_ag", "floors_ag"],
                                                       geometry=[], crs=building_footprints.crs)
    surrounding_building_footprints.to_file(os.path.join(output_directory, "surroundings.shp"))


def create_cea_building_typology(output_directory: str, building_footprints: gpd.GeoDataFrame,
                                 building_type_mapping: DefaultDict[str, str],
                                 building_standard_mapping: DefaultDict[str, str]) -> None:
    """
    Map building properties from input to CEA building types
    The input should at least have `name` and `building_type` property
    """
    # Geometry not required in typology file but needed for geopandas to export to .dbf
    building_typology = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry")
    building_typology["Name"] = building_footprints["name"].astype(str).str.replace(":", "-")
    # Not important in this case, so just set a default year
    building_typology["YEAR"] = DEFAULT_BUILDING_YEAR

    # Map building types to CEA types
    building_types = building_footprints["building_type"].map(building_type_mapping)
    # Map building types to CEA types
    building_standards = building_footprints["building_type"].map(building_standard_mapping)

    # Set standard according to efficiency standard
    building_typology["STANDARD"] = building_standards

    # Assume single use, set everything else to zero-value
    building_typology["1ST_USE"] = building_types
    building_typology["1ST_USE_R"] = 1.0
    building_typology[["2ND_USE", "3RD_USE"]] = "NONE"
    building_typology[["2ND_USE_R", "3RD_USE_R"]] = 0.0

    # Copy only the .dbf file and discard the rest
    with tempfile.TemporaryDirectory() as tmpdir:
        building_typology.to_file(os.path.join(tmpdir, "typology.dbf"))
        os.makedirs(output_directory, exist_ok=True)
        shutil.copyfile(os.path.join(tmpdir, "typology.dbf"), os.path.join(output_directory, "typology.dbf"))


def get_cea_database_files(output_dir: str, commit_id: str) -> str:
    subprocess.run(["git", "clone", CEA_REPO_URL, output_dir])
    subprocess.run(["git", "checkout", commit_id], cwd=output_dir)

    return os.path.join(output_dir, "cea", "databases")


def create_cea_databases(output_path: str, cea_database_path: str, database_name: str) -> None:
    """
    Return database from CEA
    """
    database_folder_path = os.path.join(cea_database_path, database_name)
    create_zip(database_folder_path, output_path)


def create_terrain_raster(output_directory: str, area_boundary, terrain_height: float = 0.0) -> None:
    """
    Creates a flat terrain raster that covers the area with the given height
    """
    # TODO: Get realistic terrain data from data source instead of creating flat terrain
    arr = np.array([[terrain_height]])
    rows, cols = arr.shape

    os.makedirs(output_directory, exist_ok=True)
    with rasterio.open(
            os.path.join(output_directory, "terrain.tif"),
            'w',
            driver='GTiff',
            height=rows,
            width=cols,
            count=1,
            dtype=np.float64,
            crs='+proj=latlong',
            transform=rasterio.transform.from_bounds(*area_boundary, height=rows, width=cols),
    ) as dataset:
        dataset.write(arr, 1)


def create_weather_file(output_directory: str, cea_database_path: str, selection: str) -> None:
    """
    Create .epw weather file based on selection
    """
    # FIXME: Would be good to generate based on climate data and radiation
    weather_path = os.path.join(cea_database_path, "weather", selection)

    os.makedirs(output_directory, exist_ok=True)
    shutil.copyfile(weather_path, os.path.join(output_directory, "weather.epw"))



def inject_new_building_standard(working_path: str, cea_database_path: str, database_name: str):
    new_standard_list = ManageBuildingTypeCSV.detect_new_building_standards(working_path)
    building_type_util = NewBuildingStandardOperation(os.path.join(cea_database_path, database_name))
    for new_standard in new_standard_list:
        base_building_type = ManageBuildingTypeCSV.get_base_building_type(new_standard['base_building_type'])
        building_type_util.inject_new_standard_to_cea_db(base_building_type, new_standard)


def validate_building_footprints(gdf:gpd.GeoDataFrame) -> bool:
    # check negative height
    if 'height' in gdf.columns:
        negative_height_gdf = gdf[gdf['height'] <= 0]
        if not negative_height_gdf.empty:
            id_list = negative_height_gdf['id'].astype(str).tolist()
            raise ValueError('Feature IDs with negative height values: ' + ', '.join(id_list))
    else:
        raise KeyError('"height" is missing in the properties of building_footprints')

    # check duplicated geometry
    duplicated_geometries = gdf[gdf.duplicated(subset='geometry', keep=False)]
    duplicated_ids = duplicated_geometries['id'].astype('str').tolist()
    if duplicated_ids:
        raise ValueError('Feature IDs with duplicated geometries: ' + ', '.join(duplicated_ids))

    # check non-polygon geometry
    non_polygons = gdf[~gdf.geometry.geom_type.isin(['Polygon'])]
    non_polygons_ids = non_polygons['id'].astype('str').tolist()
    if non_polygons_ids:
        raise ValueError('Feature IDs with non-polygon geometries: ' + ', '.join(non_polygons_ids))

    return True


def get_mapping_type(type_dict: dict, building_type: str) -> str:
    for k, v in type_dict.items():
        if building_type in v:
            return k
    return ''

def validate_building_types(gdf:gpd.GeoDataFrame, working_dir:str) -> bool:
    # unzip cea databases
    cea_db_path = os.path.join(working_dir, CEA_DB_NAME)
    unzipped_path = os.path.join(working_dir, 'cea_db_unzipped')
    with ZipFile(cea_db_path, 'r') as db_ref:
        db_ref.extractall(unzipped_path)

    # get non-demand building type list from cea database
    building_types_csv_path = os.path.join(unzipped_path, 'archetypes', 'use_types')
    no_demand_type_list = []
    for type_csv in os.listdir(building_types_csv_path):
        if type_csv.endswith('csv'):
            building_type = type_csv.split('.')[0]
            df_type = pd.read_csv(os.path.join(building_types_csv_path, type_csv), skiprows=2)
            if df_type['COOLING'].eq('OFF').all():
                no_demand_type_list.append(building_type)

    # get building type dictionary from the file 'parameters'
    building_type_dict = {}
    param_path = os.path.join(working_dir, 'parameters')
    with open(param_path, 'r') as param_file:
        param_dict = json.load(param_file)
        building_type_dict = param_dict['building_type_mapping']

    # count non-demand features
    no_demand_item_num = 0
    for index, row in gdf.iterrows():
        building_type = row["building_type"]
        mapping_type = get_mapping_type(building_type_dict, building_type)
        if mapping_type and mapping_type in no_demand_type_list:
            no_demand_item_num += 1

    if no_demand_item_num == len(gdf):
        raise ValueError("There's no cooling demand in the building_footprints")
    else:
        shutil.rmtree(unzipped_path)
        return True


def coord_4326_to_32648(p_xy: (float, float)) -> (float, float):
    in_proj = pyproj.Proj('epsg:4326')
    out_proj = pyproj.Proj('epsg:32648')
    return pyproj.transform(in_proj, out_proj, x=p_xy[1], y=p_xy[0])


def function(working_directory: str) -> None:
    working_dir = os.path.realpath(os.path.expanduser(working_directory))

    # convert to EPSG:32648
    bf_path = os.path.join(working_dir, "building_footprints")
    with open(bf_path, 'r') as f:
        geojson = json.load(f)
        for feature in geojson['features']:
            coords0 = feature['geometry']['coordinates']
            coords1 = []
            for poly0 in coords0:
                poly1 = [coord_4326_to_32648(p_xy) for p_xy in poly0]
                coords1.append(poly1)
            feature['geometry']['coordinates'] = coords1
    with open(bf_path, 'w') as f:
        geojson['crs'] = {
            'type': 'name',
            'properties': {
                'name': 'urn:ogc:def:crs:EPSG::32648'
            }
        }
        json.dump(geojson, f, indent=2)

    building_footprints_gdf = gpd.read_file(bf_path)
    # check whether the building_footprints is valid for simulation
    validate_building_footprints(building_footprints_gdf)
    building_footprints_gdf = clean_building_geometries(building_footprints_gdf)

    with open(os.path.join(working_dir, "parameters")) as f:
        parameters = json.load(f)

    _building_type_mapping = parameters["building_type_mapping"]
    default_building_type = parameters["default_building_type"]
    _building_standard_mapping = parameters["building_standard_mapping"]
    default_building_standard = parameters["default_building_standard"]
    commit_id = parameters["commit_id"]
    database_name = parameters["database_name"]
    terrain_height = parameters["terrain_height"]
    weather = parameters["weather"]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Clone CEA git repo
        cea_repo = os.path.join(tmpdir, "cea_repo")
        cea_database_path = get_cea_database_files(cea_repo, commit_id)

        # Detect whether there's new standard
        inject_new_building_standard(working_dir, cea_database_path, database_name)
        print_progress(10)

        create_cea_databases(os.path.join(working_dir, "cea_databases"), cea_database_path, database_name)
        print_output("cea_databases")
        print_progress(20)

        # check whether there's cooling demand, if no, stop simulation
        # the validation can only be done after the cea_databases has been generated
        validate_building_types(building_footprints_gdf, working_dir)
        print_progress(30)

        cea_inputs = os.path.join(tmpdir, "cea_inputs")
        create_cea_footprints(os.path.join(cea_inputs, "building-geometry"), building_footprints_gdf)
        print_progress(50)

        # Create mappings
        building_type_mapping = create_dataframe_mapping_from_dict(_building_type_mapping,
                                                                   default_building_type)
        building_standard_mapping = create_dataframe_mapping_from_dict(_building_standard_mapping,
                                                                       default_building_standard)

        create_cea_building_typology(os.path.join(cea_inputs, "building-properties"), building_footprints_gdf,
                                     building_type_mapping, building_standard_mapping)

        create_terrain_raster(os.path.join(cea_inputs, "topography"), building_footprints_gdf.total_bounds,
                              terrain_height)
        create_weather_file(os.path.join(cea_inputs, "weather"), cea_database_path, weather)

        create_zip(cea_inputs, os.path.join(working_dir, "cea_run_package"))
        print_output("cea_run_package")
    print_progress(100)


if __name__ == '__main__':
    function(sys.argv[1])
