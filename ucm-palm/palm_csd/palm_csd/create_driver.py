#!/usr/bin/env python3
# ------------------------------------------------------------------------------ #
# This file is part of the PALM model system.
#
# PALM is free software: you can redistribute it and/or modify it under the terms
# of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# PALM is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# PALM. If not, see <http://www.gnu.org/licenses/>.
#
# Copyright 1997-2021  Leibniz Universitaet Hannover
# ------------------------------------------------------------------------------ #
#
# Description:
# ------------
# Processing tool for creating PIDS conform static drivers from rastered netCDF
# input
#
# @Author Bjoern Maronga (maronga@muk.uni-hannover.de)
#
# @TODO Make input files optional
# @TODO Allow for ASCII input of terrain height and building height
# @TODO Modularize reading config file
# @TODO Convert to object-oriented treatment (buidings, trees)
# @TODO Automatically shift child domains so that their origin lies intersects
#       a edge note of the parent grid
# ------------------------------------------------------------------------------ #

from palm_csd.netcdf_interface import *
from palm_csd.tools import *
from palm_csd.canopy_generator import *
import numpy as np
import numpy.ma as ma


def read_config_file(input_configuration_file):
    import configparser
    from math import floor
    import os

    config = configparser.RawConfigParser(allow_no_value=True)

    # Check if configuration files exists and quit otherwise
    if not os.path.isfile(input_configuration_file):
        print("Error. No configuration file " + input_configuration_file + " found.")
        raise FileNotFoundError

    config.read(input_configuration_file)

    # Definition of settings

    global settings_bridge_width
    global settings_lai_roof_intensive
    global settings_lai_roof_extensive
    global settings_lai_tree_lower_threshold
    global settings_season
    global settings_lai_low_default
    global settings_lai_high_default
    global settings_patch_height_default
    global settings_lai_alpha
    global settings_lai_beta
    global settings_veg_type_below_trees
    global ndomains

    # Definition of global configuration parameters
    global global_acronym
    global global_angle
    global global_author
    global global_campaign
    global global_comment
    global global_contact
    global global_data_content
    global global_dependencies
    global global_institution
    global global_keywords
    global global_location
    global global_palm_version
    global global_references
    global global_site
    global global_source
    global global_time

    global output_file
    global output_version

    # Definition of domain parameters
    global domain_names
    global domain_px
    global domain_x0
    global domain_y0
    global domain_x1
    global domain_y1
    global domain_nx
    global domain_ny
    global domain_dz
    global domain_3d
    global domain_high_vegetation
    global domain_ip
    global domain_za
    global domain_parent
    global domain_green_roofs
    global domain_street_trees
    global domain_canopy_patches
    global domain_overhanging_trees
    global domain_remove_low_lai_tree
    global domain_water_temperatures

    # Definition of input data parameters
    global input_names
    global input_px

    global input_file_x
    global input_file_y
    global input_file_x_UTM
    global input_file_y_UTM
    global input_file_lat
    global input_file_lon
    global input_file_zt
    global input_file_buildings_2d
    global input_file_bridges_2d
    global input_file_building_id
    global input_file_bridges_id
    global input_file_building_type
    global input_file_building_type
    global input_file_lai
    global input_file_vegetation_type
    global input_file_vegetation_height
    global input_file_pavement_type
    global input_file_water_type
    global input_file_water_temperature
    global input_file_street_type
    global input_file_street_crossings
    global input_file_soil_type
    global input_file_vegetation_on_roofs
    global input_file_tree_crown_diameter
    global input_file_tree_height
    global input_file_tree_trunk_diameter
    global input_file_tree_type
    global input_file_patch_height
    global input_file_patch_type

    global zt_all
    global zt_min

    # Default settings (except paths to input data). Every entry listed is optional in the
    # configuration file, every entry not listed has to given in the configuration file.
    # Global settings with the value None are not written to the output file.
    default_settings = {
        # settings
        "bridge_width": 3.0,
        "lai_roof_intensive": 2.5,
        "lai_roof_extensive": 0.8,
        "lai_high_vegetation_default": 6.0,
        "lai_low_vegetation_default": 1.0,
        "lai_tree_lower_threshold": 0.,
        "lai_alpha": 5.0,
        "lai_beta": 3.0,
        "patch_height_default": 10.0,
        "season": "summer",
        "vegetation_type_below_trees": 3,

        # global
        "acronym": None,
        "author": None,
        "campaign": None,
        "comment": None,
        "contact_person": None,
        "data_content": None,
        "dependencies": None,
        "institution": None,
        "keywords": None,
        "location": None,
        "origin_time": "",
        "palm_version": None,
        "references": None,
        "rotation_angle": 0.,
        "site": None,
        "source": None,

        "allow_high_vegetation": False,
        "buildings_3d": False,
        "domain_parent": None,
        "generate_vegetation_patches": True,
        "interpolate_terrain": False,
        "overhanging_trees": True,
        "remove_low_lai_tree": False,
        "street_trees": True,
        "use_palm_z_axis": False,
        "vegetation_on_roofs": True,
        "water_temperature_per_water_type": None,

        "path": None,

        "version": None
    }

    ndomains = 0

    domain_names = []
    domain_px = []
    domain_x0 = []
    domain_y0 = []
    domain_x1 = []
    domain_y1 = []
    domain_nx = []
    domain_ny = []
    domain_dz = []
    domain_3d = []
    domain_high_vegetation = []
    domain_ip = []
    domain_za = []
    domain_parent = []
    domain_green_roofs = []
    domain_street_trees = []
    domain_canopy_patches = []
    domain_overhanging_trees = []
    domain_remove_low_lai_tree = []
    domain_water_temperatures = []

    zt_min = 0.0
    zt_all = []

    input_names = []
    input_px = []

    input_file_x = []
    input_file_y = []
    input_file_x_UTM = []
    input_file_y_UTM = []
    input_file_lat = []
    input_file_lon = []

    input_file_zt = []
    input_file_buildings_2d = []
    input_file_bridges_2d = []
    input_file_building_id = []
    input_file_bridges_id = []
    input_file_building_type = []
    input_file_lai = []
    input_file_vegetation_type = []
    input_file_vegetation_height = []
    input_file_pavement_type = []
    input_file_water_type = []
    input_file_water_temperature = []
    input_file_street_type = []
    input_file_street_crossings = []
    input_file_soil_type = []
    input_file_vegetation_on_roofs = []
    input_file_tree_crown_diameter = []
    input_file_tree_height = []
    input_file_tree_trunk_diameter = []
    input_file_tree_type = []
    input_file_patch_height = []
    input_file_patch_type = []

    # Load all user parameters from config file
    for i in range(0, len(config.sections())):

        read_tmp = config.sections()[i]

        if read_tmp == 'global':
            global_acronym = get_setting(config, read_tmp, 'acronym', default_settings)
            global_angle = get_setting(config, read_tmp, 'rotation_angle', default_settings,
                                       variable_type=float)
            global_author = get_setting(config, read_tmp, 'author', default_settings)
            global_campaign = get_setting(config, read_tmp, 'campaign', default_settings)
            global_comment = get_setting(config, read_tmp, 'comment', default_settings)
            global_contact = get_setting(config, read_tmp, 'contact_person', default_settings)
            global_data_content = get_setting(config, read_tmp, 'data_content', default_settings)
            global_dependencies = get_setting(config, read_tmp, 'dependencies', default_settings)
            global_institution = get_setting(config, read_tmp, 'institution', default_settings)
            global_keywords = get_setting(config, read_tmp, 'keywords', default_settings)
            global_location = get_setting(config, read_tmp, 'location', default_settings)
            global_time = get_setting(config, read_tmp, 'origin_time', default_settings)
            global_palm_version = get_setting(config, read_tmp, 'palm_version', default_settings)
            global_references = get_setting(config, read_tmp, 'references', default_settings)
            global_site = get_setting(config, read_tmp, 'site', default_settings)
            global_source = get_setting(config, read_tmp, 'source', default_settings)

        if read_tmp == 'settings':
            settings_lai_roof_intensive = \
                get_setting(config, read_tmp, 'lai_roof_intensive', default_settings,
                            variable_type=float)
            settings_lai_roof_extensive = \
                get_setting(config, read_tmp, 'lai_roof_extensive', default_settings,
                            variable_type=float)
            settings_bridge_width = \
                get_setting(config, read_tmp, 'bridge_width', default_settings,
                            variable_type=float)
            settings_season = get_setting(config, read_tmp, 'season', default_settings)
            settings_lai_high_default = \
                get_setting(config, read_tmp, 'lai_high_vegetation_default', default_settings,
                            variable_type=float)
            settings_lai_low_default = \
                get_setting(config, read_tmp, 'lai_low_vegetation_default', default_settings,
                            variable_type=float)
            settings_lai_tree_lower_threshold = \
                get_setting(config, read_tmp, 'lai_tree_lower_threshold', default_settings,
                            variable_type=float)
            settings_patch_height_default = \
                get_setting(config, read_tmp, 'patch_height_default', default_settings,
                            variable_type=float)
            settings_lai_alpha = \
                get_setting(config, read_tmp, 'lai_alpha', default_settings,
                            variable_type=float)
            settings_lai_beta = \
                get_setting(config, read_tmp, 'lai_beta', default_settings,
                            variable_type=float)
            settings_veg_type_below_trees = \
                get_setting(config, read_tmp, 'vegetation_type_below_trees', default_settings,
                            variable_type=int)

        if read_tmp == 'output':
            path_out = get_setting(config, read_tmp, 'path', default_settings)
            output_file = get_filename(config, read_tmp, 'file_out', path_out)
            output_version = \
                get_setting(config, read_tmp, 'version', default_settings,
                            variable_type=float)

        if read_tmp.split("_")[0] == 'domain':
            ndomains = ndomains + 1
            domain_names.append(read_tmp.split("_")[1])

            # temporary values
            pixel_size = \
                get_setting(config, read_tmp, 'pixel_size', default_settings, variable_type=float)
            origin_x = \
                get_setting(config, read_tmp, 'origin_x', default_settings, variable_type=float)
            origin_y = \
                get_setting(config, read_tmp, 'origin_y', default_settings, variable_type=float)
            nx = \
                get_setting(config, read_tmp, 'nx', default_settings, variable_type=int)
            ny = \
                get_setting(config, read_tmp, 'ny', default_settings, variable_type=int)
            x0 = int(floor(
                origin_x / pixel_size
            ))
            y0 = int(floor(
                origin_y / pixel_size
            ))

            domain_px.append(pixel_size)
            domain_nx.append(nx)
            domain_ny.append(ny)
            domain_dz.append(
                get_setting(config, read_tmp, 'dz', default_settings,
                            variable_type=float)
            )
            domain_3d.append(
                get_setting(config, read_tmp, 'buildings_3d', default_settings,
                            variable_type=bool)
            )
            domain_high_vegetation.append(
                get_setting(config, read_tmp, 'allow_high_vegetation', default_settings,
                            variable_type=bool)
            )
            domain_canopy_patches.append(
                get_setting(config, read_tmp, 'generate_vegetation_patches', default_settings,
                            variable_type=bool)
            )
            domain_ip.append(
                get_setting(config, read_tmp, 'interpolate_terrain', default_settings,
                            variable_type=bool)
            )
            domain_za.append(
                get_setting(config, read_tmp, 'use_palm_z_axis', default_settings,
                            variable_type=bool)
            )
            if domain_ip[ndomains - 1] and not domain_za[ndomains - 1]:
                domain_za[ndomains - 1] = True
                print("+++ Overwrite user setting for use_palm_z_axis")

            domain_parent.append(
                get_setting(config, read_tmp, 'domain_parent', default_settings)
            )

            domain_x0.append(x0)
            domain_y0.append(y0)
            domain_x1.append(x0 + nx)
            domain_y1.append(y0 + ny)

            domain_green_roofs.append(
                get_setting(config, read_tmp, 'vegetation_on_roofs', default_settings,
                            variable_type=bool)
            )
            domain_street_trees.append(
                get_setting(config, read_tmp, 'street_trees', default_settings,
                            variable_type=bool)
            )
            domain_overhanging_trees.append(
                get_setting(config, read_tmp, 'overhanging_trees', default_settings,
                            variable_type=bool)
            )
            domain_remove_low_lai_tree.append(
                get_setting(config, read_tmp, 'remove_low_lai_tree', default_settings,
                            variable_type=bool)
            )
            domain_water_temperatures.append(
                get_setting(config, read_tmp, 'water_temperature_per_water_type', default_settings,
                            variable_type=str)
            )

        if read_tmp.split("_")[0] == 'input':
            input_names.append(read_tmp.split("_")[1])
            input_px.append(
                get_setting(config, read_tmp, 'pixel_size', default_settings,
                            variable_type=float)
            )
            path_input = get_setting(config, read_tmp, 'path', default_settings)

            input_file_x.append(
                get_filename(config, read_tmp, 'file_x', path_input)
            )
            input_file_y.append(
                get_filename(config, read_tmp, 'file_y', path_input)
            )
            input_file_lat.append(
                get_filename(config, read_tmp, 'file_lat', path_input)
            )
            input_file_lon.append(
                get_filename(config, read_tmp, 'file_lon', path_input)
            )
            input_file_x_UTM.append(
                get_filename(config, read_tmp, 'file_x_UTM', path_input)
            )
            input_file_y_UTM.append(
                get_filename(config, read_tmp, 'file_y_UTM', path_input)
            )
            input_file_zt.append(
                get_filename(config, read_tmp, 'file_zt', path_input)
            )

            input_file_buildings_2d.append(
                get_filename(config, read_tmp, 'file_buildings_2d', path_input)
            )
            input_file_bridges_2d.append(
                get_filename(config, read_tmp, 'file_bridges_2d', path_input)
            )
            input_file_building_id.append(
                get_filename(config, read_tmp, 'file_building_id', path_input)
            )
            input_file_bridges_id.append(
                get_filename(config, read_tmp, 'file_bridges_id', path_input)
            )
            input_file_building_type.append(
                get_filename(config, read_tmp, 'file_building_type', path_input)
            )

            input_file_lai.append(
                get_filename(config, read_tmp, 'file_lai', path_input, optional=True)
            )
            input_file_vegetation_type.append(
                get_filename(config, read_tmp, 'file_vegetation_type', path_input)
            )
            input_file_vegetation_height.append(
                get_filename(config, read_tmp, 'file_vegetation_height', path_input)
            )
            input_file_patch_height.append(
                get_filename(config, read_tmp, 'file_patch_height', path_input)
            )
            input_file_patch_type.append(
                get_filename(config, read_tmp, 'file_patch_type', path_input,
                             optional=True)
            )
            input_file_tree_crown_diameter.append(
                get_filename(config, read_tmp, 'file_tree_crown_diameter', path_input,
                             optional=True)
            )
            input_file_tree_height.append(
                get_filename(config, read_tmp, 'file_tree_height', path_input,
                             optional=True)
            )
            input_file_tree_trunk_diameter.append(
                get_filename(config, read_tmp, 'file_tree_trunk_diameter', path_input,
                             optional=True)
            )
            input_file_tree_type.append(
                get_filename(config, read_tmp, 'file_tree_type', path_input)
            )
            input_file_vegetation_on_roofs.append(
                get_filename(config, read_tmp, 'file_vegetation_on_roofs', path_input)
            )

            input_file_pavement_type.append(
                get_filename(config, read_tmp, 'file_pavement_type', path_input)
            )
            input_file_water_type.append(
                get_filename(config, read_tmp, 'file_water_type', path_input)
            )
            input_file_water_temperature.append(
                get_filename(config, read_tmp, 'file_water_temperature', path_input, optional=True)
            )
            input_file_street_type.append(
                get_filename(config, read_tmp, 'file_street_type', path_input)
            )
            input_file_street_crossings.append(
                get_filename(config, read_tmp, 'file_street_crossings', path_input)
            )

            # input_file_soil_type.append(config.get(read_tmp, 'path') + "/" +
            #     config.get(read_tmp, 'file_soil_type'))
    return 0


############################################################

def create_driver(input_configuration_file):
    # Definition of settings
    global settings_bridge_width
    global settings_lai_roof_intensive
    global settings_lai_roof_extensive
    global settings_lai_tree_lower_threshold
    global settings_season
    global settings_lai_low_default
    global settings_lai_high_default
    global settings_patch_height_default
    global settings_lai_alpha
    global settings_lai_beta
    global settings_veg_type_below_trees
    global ndomains

    # Definition of global configuration parameters
    global global_acronym
    global global_angle
    global global_author
    global global_campaign
    global global_comment
    global global_contact
    global global_data_content
    global global_dependencies
    global global_institution
    global global_keywords
    global global_location
    global global_palm_version
    global global_references
    global global_site
    global global_source
    global global_time

    global output_file
    global output_version

    # Definition of domain parameters
    global domain_names
    global domain_px
    global domain_x0
    global domain_y0
    global domain_x1
    global domain_y1
    global domain_nx
    global domain_ny
    global domain_dz
    global domain_3d
    global domain_high_vegetation
    global domain_ip
    global domain_za
    global domain_parent
    global domain_green_roofs
    global domain_street_trees
    global domain_canopy_patches
    global domain_overhanging_trees
    global domain_remove_low_lai_tree

    # Definition of input data parameters
    global input_names
    global input_px

    global input_file_x
    global input_file_y
    global input_file_x_UTM
    global input_file_y_UTM
    global input_file_lat
    global input_file_lon
    global input_file_zt
    global input_file_buildings_2d
    global input_file_bridges_2d
    global input_file_building_id
    global input_file_bridges_id
    global input_file_building_type
    global input_file_building_type
    global input_file_lai
    global input_file_vegetation_type
    global input_file_vegetation_height
    global input_file_pavement_type
    global input_file_water_type
    global input_file_water_temperature
    global input_file_street_type
    global input_file_street_crossings
    global input_file_soil_type
    global input_file_vegetation_on_roofs
    global input_file_tree_crown_diameter
    global input_file_tree_height
    global input_file_tree_trunk_diameter
    global input_file_tree_type
    global input_file_patch_height

    global zt_all
    global zt_min

    datatypes = {
        "x": "f4",
        "y": "f4",
        "z": "f4",
        "lat": "f4",
        "lon": "f4",
        "E_UTM": "f4",
        "N_UTM": "f4",
        "zt": "f4",
        "buildings_2d": "f4",
        "buildings_3d": "b",
        "bridges_2d": "f4",
        "building_id": "i",
        "bridges_id": "i",
        "building_type": "b",
        "nsurface_fraction": "i",
        "vegetation_type": "b",
        "vegetation_height": "f4",
        "pavement_type": "b",
        "water_type": "b",
        "street_type": "b",
        "street_crossings": "b",
        "soil_type": "b",
        "surface_fraction": "f4",
        "building_pars": "f4",
        "vegetation_pars": "f4",
        "tree_data": "f4",
        "tree_id": "i",
        "tree_type": "f4",
        "tree_types": "b",
        "nbuilding_pars": "i",
        "nvegetation_pars": "i",
        "zlad": "f4",
        "water_pars": "f4",
        "nwater_pars": "i",
    }

    fillvalues = {
        "lat": float(-9999.0),
        "lon": float(-9999.0),
        "E_UTM": float(-9999.0),
        "N_UTM": float(-9999.0),
        "zt": float(-9999.0),
        "buildings_2d": float(-9999.0),
        "buildings_3d": np.byte(-127),
        "bridges_2d": float(-9999.0),
        "building_id": int(-9999),
        "bridges_id": int(-9999),
        "building_type": np.byte(-127),
        "nsurface_fraction": int(-9999),
        "vegetation_type": np.byte(-127),
        "vegetation_height": float(-9999.0),
        "pavement_type": np.byte(-127),
        "water_type": np.byte(-127),
        "street_type": np.byte(-127),
        "street_crossings": np.byte(-127),
        "soil_type": np.byte(-127),
        "surface_fraction": float(-9999.0),
        "building_pars": float(-9999.0),
        "vegetation_pars": float(-9999.0),
        "tree_data": float(-9999.0),
        "tree_id": int(-9999),
        "tree_type": float(-9999.0),
        "tree_types": np.byte(-127),
        "water_pars": float(-9999.0),
    }

    defaultvalues = {
        "lat": float(-9999.0),
        "lon": float(-9999.0),
        "E_UTM": float(-9999.0),
        "N_UTM": float(-9999.0),
        "zt": float(0.0),
        "buildings_2d": float(0.0),
        "buildings_3d": 0,
        "bridges_2d": float(0.0),
        "building_id": int(0),
        "bridges_id": int(0),
        "building_type": 1,
        "nsurface_fraction": int(-9999),
        "vegetation_type": 3,
        "vegetation_height": float(-9999.0),
        "pavement_type": 1,
        "water_type": 1,
        "street_type": 1,
        "street_crossings": 0,
        "soil_type": 1,
        "surface_fraction": float(0.0),
        "buildings_pars": float(-9999.0),
        "tree_data": float(-9999.0),
        "tree_type": np.byte(-127),
        "patch_type": 0,
        "vegetation_pars": float(-9999.0),
        "water_pars": float(-9999.0),
        "water_temperature": 283.0,
    }

    #  vegetation_types that are considered high vegetation
    vt_high_vegetation = ma.array([4,     # evergreen needleleaf trees
                                   5,     # deciduous needleleaf trees
                                   6,     # evergreen broadleaf trees
                                   7,     # deciduous broadleaf trees
                                   17,    # mixed forest/woodland
                                   18])   # interrupted forest

    # Read configuration file and set parameters accordingly
    read_config_file(input_configuration_file)

    filename = []
    ii = []
    ii_parent = []
    # Define indices and filenames for all domains and create netCDF files
    for i in range(0, ndomains):

        # Calculate indices and input files
        ii.append(input_px.index(domain_px[i]))
        filename.append(output_file + "_" + domain_names[i])
        if domain_parent[i] is not None:
            ii_parent.append(input_px.index(domain_px[domain_names.index(domain_parent[i])]))
        else:
            ii_parent.append(None)

        x_UTM = nc_read_from_file_2d(input_file_x[ii[i]], "Band1", domain_x0[i], domain_x0[i] + 1,
                                     domain_y0[i], domain_y0[i] + 1)
        y_UTM = nc_read_from_file_2d(input_file_y[ii[i]], "Band1", domain_x0[i], domain_x0[i] + 1,
                                     domain_y0[i], domain_y0[i] + 1)
        lat = nc_read_from_file_2d(input_file_lat[ii[i]], "Band1", domain_x0[i], domain_x0[i] + 1,
                                   domain_y0[i], domain_y0[i] + 1)
        lon = nc_read_from_file_2d(input_file_lon[ii[i]], "Band1", domain_x0[i], domain_x0[i] + 1,
                                   domain_y0[i], domain_y0[i] + 1)

        # Calculate position of origin
        x_UTM_origin = float(x_UTM[0, 0]) - 0.5 * (float(x_UTM[0, 1]) - float(x_UTM[0, 0]))
        y_UTM_origin = float(y_UTM[0, 0]) - 0.5 * (float(y_UTM[1, 0]) - float(y_UTM[0, 0]))
        x_origin = float(lon[0, 0]) - 0.5 * (float(lon[0, 1]) - float(lon[0, 0]))
        y_origin = float(lat[0, 0]) - 0.5 * (float(lat[1, 0]) - float(lat[0, 0]))

        # Create netCDF output file and set global attributes
        nc_create_file(filename[i])
        nc_write_global_attributes(filename[i],
                                   x_UTM_origin, y_UTM_origin, y_origin, x_origin, global_time,
                                   global_acronym, global_angle, global_author, global_campaign,
                                   global_comment, global_contact, global_data_content,
                                   global_dependencies, global_institution, global_keywords,
                                   global_location, global_palm_version, global_references,
                                   global_site, global_source, output_version)

        del x_UTM, y_UTM, lat, lon

    # Process terrain height
    for i in range(0, ndomains):
        # Read and write terrain height (zt)
        zt = nc_read_from_file_2d(input_file_zt[ii[i]], 'Band1', domain_x0[i], domain_x1[i],
                                  domain_y0[i], domain_y1[i])

        # Final step: add zt array to the global array
        zt_all.append(zt)
        del zt

    # Calculate the global (all domains) minimum of the terrain height. This value will be
    # substracted for all data sets
    zt_min = min(zt_all[0].flatten())
    for i in range(0, ndomains):
        zt_min = min(zt_min, min(zt_all[i].flatten()))

    del zt_all[:]

    print("Shift terrain heights by -" + str(zt_min))
    for i in range(0, ndomains):

        # Read and write terrain height (zt)
        zt = nc_read_from_file_2d(input_file_zt[ii[i]], 'Band1', domain_x0[i], domain_x1[i],
                                  domain_y0[i], domain_y1[i])
        x = nc_read_from_file_1d(input_file_x[ii[i]], "x", domain_x0[i], domain_x1[i])
        y = nc_read_from_file_1d(input_file_y[ii[i]], "y", domain_y0[i], domain_y1[i])

        zt = zt - zt_min

        nc_write_global_attribute(filename[i], 'origin_z', float(zt_min))

        # If necessary, interpolate parent domain terrain height on child domain grid and blend
        # the two
        if domain_ip[i]:
            parent_id = domain_names.index(domain_parent[i])
            tmp_x0 = int(domain_x0[i] * domain_px[i] / domain_px[parent_id]) - 1
            tmp_y0 = int(domain_y0[i] * domain_px[i] / domain_px[parent_id]) - 1
            tmp_x1 = int(domain_x1[i] * domain_px[i] / domain_px[parent_id]) + 1
            tmp_y1 = int(domain_y1[i] * domain_px[i] / domain_px[parent_id]) + 1

            tmp_x = nc_read_from_file_1d(input_file_x[ii_parent[i]], "x", tmp_x0, tmp_x1)
            tmp_y = nc_read_from_file_1d(input_file_y[ii_parent[i]], "y", tmp_y0, tmp_y1)

            zt_parent = nc_read_from_file_2d(input_file_zt[ii_parent[i]], 'Band1', tmp_x0, tmp_x1,
                                             tmp_y0, tmp_y1)

            zt_parent = zt_parent - zt_min

            # Interpolate array and bring to PALM grid of child domain
            zt_ip = interpolate_2d(zt_parent, tmp_x, tmp_y, x, y)
            zt_ip = bring_to_palm_grid(zt_ip, x, y, domain_dz[parent_id])

            # Shift the child terrain height according to the parent mean terrain height
            print("Shifting terrain height of domain {}: -{} +{}".format(domain_names[i],
                                                                         str(np.mean(zt)),
                                                                         str(np.mean(zt_ip))))
            zt = zt - np.mean(zt) + np.mean(zt_ip)

            # Blend over the parent and child terrain height within a radius of 50 px (or less if
            # domain is smaller than 50 px)
            zt = blend_array_2d(zt, zt_ip, min(50, min(zt.shape) * 0.5))

            del(zt_ip)

        # Final step: add zt array to the global array
        zt_all.append(zt)
        del zt

    # Read and shift x and y coordinates, shift terrain height according to its minimum value and
    # write all data to file
    for i in range(0, ndomains):
        # Read horizontal grid variables from zt file and write them to output file
        x = nc_read_from_file_1d(input_file_x[ii[i]], "x", domain_x0[i], domain_x1[i])
        y = nc_read_from_file_1d(input_file_y[ii[i]], "y", domain_y0[i], domain_y1[i])
        x = x - min(x.flatten()) + domain_px[i] / 2.0
        y = y - min(y.flatten()) + domain_px[i] / 2.0
        nc_write_dimension(filename[i], 'x', x, datatypes["x"])
        nc_write_dimension(filename[i], 'y', y, datatypes["y"])
        nc_write_attribute(filename[i], 'x', 'long_name', 'x')
        nc_write_attribute(filename[i], 'x', 'standard_name', 'projection_x_coordinate')
        nc_write_attribute(filename[i], 'x', 'units', 'm')
        nc_write_attribute(filename[i], 'y', 'long_name', 'y')
        nc_write_attribute(filename[i], 'y', 'standard_name', 'projection_y_coordinate')
        nc_write_attribute(filename[i], 'y', 'units', 'm')

        lat = nc_read_from_file_2d(input_file_lat[ii[i]], "Band1", domain_x0[i], domain_x1[i],
                                   domain_y0[i], domain_y1[i])
        lon = nc_read_from_file_2d(input_file_lon[ii[i]], "Band1", domain_x0[i], domain_x1[i],
                                   domain_y0[i], domain_y1[i])

        nc_write_to_file_2d(filename[i], 'lat', lat, datatypes["lat"], 'y', 'x', fillvalues["lat"])
        nc_write_to_file_2d(filename[i], 'lon', lon, datatypes["lon"], 'y', 'x', fillvalues["lon"])

        nc_write_attribute(filename[i], 'lat', 'long_name', 'latitude')
        nc_write_attribute(filename[i], 'lat', 'standard_name', 'latitude')
        nc_write_attribute(filename[i], 'lat', 'units', 'degrees_north')

        nc_write_attribute(filename[i], 'lon', 'long_name', 'longitude')
        nc_write_attribute(filename[i], 'lon', 'standard_name', 'longitude')
        nc_write_attribute(filename[i], 'lon', 'units', 'degrees_east')

        x_UTM = nc_read_from_file_2d(input_file_x_UTM[ii[i]], "Band1", domain_x0[i], domain_x1[i],
                                     domain_y0[i], domain_y1[i])
        y_UTM = nc_read_from_file_2d(input_file_y_UTM[ii[i]], "Band1", domain_x0[i], domain_x1[i],
                                     domain_y0[i], domain_y1[i])

        nc_write_to_file_2d(filename[i], 'E_UTM', x_UTM, datatypes["E_UTM"], 'y', 'x',
                            fillvalues["E_UTM"])
        nc_write_to_file_2d(filename[i], 'N_UTM', y_UTM, datatypes["N_UTM"], 'y', 'x',
                            fillvalues["N_UTM"])

        nc_write_attribute(filename[i], 'E_UTM', 'long_name', 'easting')
        nc_write_attribute(filename[i], 'E_UTM', 'standard_name', 'projection_x_coorindate')
        nc_write_attribute(filename[i], 'E_UTM', 'units', 'm')

        nc_write_attribute(filename[i], 'N_UTM', 'long_name', 'northing')
        nc_write_attribute(filename[i], 'N_UTM', 'standard_name', 'projection_y_coorindate')
        nc_write_attribute(filename[i], 'N_UTM', 'units', 'm')

        nc_write_crs(filename[i], nc_read_from_file_crs(input_file_x_UTM[ii[i]], "Band1"))

        # If necessary, bring terrain height to PALM's vertical grid. This is either forced by
        # the user or implicitly by using interpolation for a child domain
        if domain_za[i]:
            zt_all[i] = bring_to_palm_grid(zt_all[i], x, y, domain_dz[i])

        nc_write_to_file_2d(filename[i], 'zt', zt_all[i], datatypes["zt"], 'y', 'x',
                            fillvalues["zt"])
        nc_write_attribute(filename[i], 'zt', 'long_name', 'orography')
        nc_write_attribute(filename[i], 'zt', 'units', 'm')
        nc_write_attribute(filename[i], 'zt', 'res_orig', domain_px[i])
        nc_write_attribute(filename[i], 'zt', 'coordinates', 'E_UTM N_UTM lon lat')
        nc_write_attribute(filename[i], 'zt', 'grid_mapping', 'crs')

    del zt_all

    # Process building height, id, and type
    for i in range(0, ndomains):
        buildings_2d = nc_read_from_file_2d(input_file_buildings_2d[ii[i]], 'Band1', domain_x0[i],
                                            domain_x1[i], domain_y0[i], domain_y1[i])

        building_id = nc_read_from_file_2d(input_file_building_id[ii[i]], 'Band1', domain_x0[i],
                                           domain_x1[i], domain_y0[i], domain_y1[i])

        building_type = nc_read_from_file_2d(input_file_building_type[ii[i]], 'Band1', domain_x0[i],
                                             domain_x1[i], domain_y0[i], domain_y1[i])
        ma.masked_greater_equal(building_type, 254, copy=False)
        building_type = ma.where(building_type < 1, defaultvalues["building_type"], building_type)

        check = check_arrays_2(buildings_2d, building_id)
        # make masks equal
        if not check:
            buildings_2d.mask = ma.mask_or(building_id.mask, buildings_2d.mask)
            # copy mask from building_2d to building_id
            building_id.mask = buildings_2d.mask.copy()
            print("Data check #1 " + str(
                check_arrays_2(buildings_2d, building_id)))

        check = check_arrays_2(buildings_2d, building_type)
        if not check:
            building_type.mask = ma.mask_or(buildings_2d.mask, building_type.mask)
            building_type = ma.where(building_type.mask & ~buildings_2d.mask,
                                     defaultvalues["building_type"], building_type)
            print("Data check #2 " + str(
                check_arrays_2(buildings_2d, building_type)))

        nc_write_to_file_2d(filename[i], 'buildings_2d', buildings_2d, datatypes["buildings_2d"],
                            'y', 'x', fillvalues["buildings_2d"])
        nc_write_attribute(filename[i], 'buildings_2d', 'long_name', 'buildings')
        nc_write_attribute(filename[i], 'buildings_2d', 'units', 'm')
        nc_write_attribute(filename[i], 'buildings_2d', 'res_orig', domain_px[i])
        nc_write_attribute(filename[i], 'buildings_2d', 'lod', 1)
        nc_write_attribute(filename[i], 'buildings_2d', 'coordinates', 'E_UTM N_UTM lon lat')
        nc_write_attribute(filename[i], 'buildings_2d', 'grid_mapping', 'crs')

        nc_write_to_file_2d(filename[i], 'building_id', building_id, datatypes["building_id"], 'y',
                            'x', fillvalues["building_id"])
        nc_write_attribute(filename[i], 'building_id', 'long_name', 'building id')
        nc_write_attribute(filename[i], 'building_id', 'units', '')
        nc_write_attribute(filename[i], 'building_id', 'res _orig', domain_px[i])
        nc_write_attribute(filename[i], 'building_id', 'coordinates', 'E_UTM N_UTM lon lat')
        nc_write_attribute(filename[i], 'building_id', 'grid_mapping', 'crs')

        nc_write_to_file_2d(filename[i], 'building_type', building_type, datatypes["building_type"],
                            'y', 'x', fillvalues["building_type"])
        nc_write_attribute(filename[i], 'building_type', 'long_name', 'building type')
        nc_write_attribute(filename[i], 'building_type', 'units', '')
        nc_write_attribute(filename[i], 'building_type', 'res_orig', domain_px[i])
        nc_write_attribute(filename[i], 'building_type', 'coordinates', 'E_UTM N_UTM lon lat')
        nc_write_attribute(filename[i], 'building_type', 'grid_mapping', 'crs')

    del buildings_2d
    del building_id
    del building_type

    # Create 3d buildings if necessary. In that course, read bridge objects and add them to
    # building layer
    for i in range(0, ndomains):

        if domain_3d[i]:
            x = nc_read_from_file_2d_all(filename[i], 'x')
            y = nc_read_from_file_2d_all(filename[i], 'y')
            buildings_2d = nc_read_from_file_2d_all(filename[i], 'buildings_2d')
            building_id = nc_read_from_file_2d_all(filename[i], 'building_id')

            bridges_2d = nc_read_from_file_2d(input_file_bridges_2d[ii[i]], 'Band1', domain_x0[i],
                                              domain_x1[i], domain_y0[i], domain_y1[i])
            bridges_id = nc_read_from_file_2d(input_file_bridges_id[ii[i]], 'Band1', domain_x0[i],
                                              domain_x1[i], domain_y0[i], domain_y1[i])

            ma.masked_equal(bridges_2d, 0.0, copy=False)
            building_id = ma.where(bridges_2d.mask, building_id, bridges_id)

            if ma.any(~buildings_2d.mask):
                buildings_3d, z = make_3d_from_2d(buildings_2d, x, y, domain_dz[i])
                if ma.any(~bridges_2d.mask):
                    buildings_3d = make_3d_from_bridges_2d(buildings_3d, bridges_2d, x, y,
                                                           domain_dz[i], settings_bridge_width)
                else:
                    print("Skipping creation of 3D bridges (no bridges in domain)")

                nc_write_dimension(filename[i], 'z', z, datatypes["z"])
                nc_write_attribute(filename[i], 'z', 'long_name', 'z')
                nc_write_attribute(filename[i], 'z', 'units', 'm')

                nc_overwrite_to_file_2d(filename[i], 'building_id', building_id)

                nc_write_to_file_3d(filename[i], 'buildings_3d', buildings_3d,
                                    datatypes["buildings_3d"], 'z', 'y', 'x',
                                    fillvalues["buildings_3d"])
                nc_write_attribute(filename[i], 'buildings_3d', 'long_name', 'buildings 3d')
                nc_write_attribute(filename[i], 'buildings_3d', 'units', '')
                nc_write_attribute(filename[i], 'buildings_3d', 'res_orig', domain_px[i])
                nc_write_attribute(filename[i], 'buildings_3d', 'lod', 2)

                del buildings_3d

            else:
                print("Skipping creation of 3D buildings (no buildings in domain)")

            del bridges_2d, bridges_id, building_id, buildings_2d

    # Read vegetation type, water_type, pavement_type, soil_type and make fields consistent
    for i in range(0, ndomains):

        building_type = nc_read_from_file_2d_all(filename[i], 'building_type')

        vegetation_type = nc_read_from_file_2d(input_file_vegetation_type[ii[i]], 'Band1',
                                               domain_x0[i], domain_x1[i], domain_y0[i],
                                               domain_y1[i])
        ma.masked_equal(vegetation_type, 255, copy=False)
        vegetation_type = ma.where(vegetation_type < 1,
                                   defaultvalues["vegetation_type"], vegetation_type)

        pavement_type = nc_read_from_file_2d(input_file_pavement_type[ii[i]], 'Band1', domain_x0[i],
                                             domain_x1[i], domain_y0[i], domain_y1[i])
        ma.masked_equal(pavement_type, 255, copy=False)
        pavement_type = ma.where(pavement_type < 1,
                                 defaultvalues["pavement_type"], pavement_type)

        water_type = nc_read_from_file_2d(input_file_water_type[ii[i]], 'Band1', domain_x0[i],
                                          domain_x1[i], domain_y0[i], domain_y1[i])
        ma.masked_equal(water_type, 255, copy=False)
        water_type = ma.where(water_type < 1, defaultvalues["water_type"], water_type)

        # TODO: replace by real soil input data
        soil_type = nc_read_from_file_2d(input_file_vegetation_type[ii[i]], 'Band1', domain_x0[i],
                                         domain_x1[i], domain_y0[i], domain_y1[i])
        ma.masked_equal(soil_type, 255, copy=False)
        soil_type = ma.where(soil_type < 1, defaultvalues["soil_type"], soil_type)

        # Make arrays consistent
        # #1 Set vegetation type to masked for pixel where a pavement type is set
        vegetation_type.mask = ma.mask_or(vegetation_type.mask, ~pavement_type.mask)

        # #2 Set vegetation type to masked for pixel where a building type is set
        vegetation_type.mask = ma.mask_or(vegetation_type.mask, ~building_type.mask)

        # #3 Set vegetation type to masked for pixel where a water type is set
        vegetation_type.mask = ma.mask_or(vegetation_type.mask, ~water_type.mask)

        # #4 Remove pavement for pixels with buildings
        pavement_type.mask = ma.mask_or(pavement_type.mask, ~building_type.mask)

        # #5 Remove pavement for pixels with water.
        pavement_type.mask = ma.mask_or(pavement_type.mask, ~water_type.mask)

        # #6 Remove water for pixels with buildings
        water_type.mask = ma.mask_or(water_type.mask, ~building_type.mask)

        # Correct vegetation_type when a vegetation height is available and is indicative of low
        # vegetation
        vegetation_height = nc_read_from_file_2d(input_file_vegetation_height[ii[i]], 'Band1',
                                                 domain_x0[i], domain_x1[i], domain_y0[i],
                                                 domain_y1[i])

        # correct vegetation_type depending on vegetation_height
        # ma.where gives ma.masked when its first argument is ma.masked. We don't want this for
        # vegetation_height here so do an extra check and use .data
        vegetation_type = ma.where((~vegetation_height.mask) & (vegetation_height.data == 0.0) &
                                   ma_isin(vegetation_type, vt_high_vegetation),
                                   3, vegetation_type)
        ma.masked_where((vegetation_height == 0.0) & ma_isin(vegetation_type, vt_high_vegetation),
                        vegetation_height, copy=False)

        # Check for consistency and fill empty fields with default vegetation type
        consistency_array, test = check_consistency_4(vegetation_type, building_type, pavement_type,
                                                      water_type)

        if test:
            vegetation_type = ma.where(consistency_array == 0, defaultvalues["vegetation_type"],
                                       vegetation_type)
            consistency_array, test = check_consistency_4(vegetation_type, building_type,
                                                          pavement_type, water_type)

        # #7 Todo: to be removed: set default soil type everywhere
        soil_type = ma.where(vegetation_type.mask & pavement_type.mask,
                             ma.masked,
                             defaultvalues["soil_type"])

        # Check for consistency and fill empty fields with default vegetation type
        consistency_array, test = check_consistency_3(vegetation_type, pavement_type, soil_type)

        # Create surface_fraction array
        x = nc_read_from_file_2d_all(filename[i], 'x')
        y = nc_read_from_file_2d_all(filename[i], 'y')
        nsurface_fraction = np.arange(0, 3)
        surface_fraction = ma.ones((len(nsurface_fraction), len(y), len(x)))

        surface_fraction[0, :, :] = ma.where(vegetation_type.mask, 0.0, 1.0)
        surface_fraction[1, :, :] = ma.where(pavement_type.mask, 0.0, 1.0)
        surface_fraction[2, :, :] = ma.where(water_type.mask, 0.0, 1.0)

        nc_write_dimension(filename[i], 'nsurface_fraction', nsurface_fraction,
                           datatypes["nsurface_fraction"])
        nc_write_to_file_3d(filename[i], 'surface_fraction', surface_fraction,
                            datatypes["surface_fraction"], 'nsurface_fraction', 'y', 'x',
                            fillvalues["surface_fraction"])
        nc_write_attribute(filename[i], 'surface_fraction', 'long_name', 'surface fraction')
        nc_write_attribute(filename[i], 'surface_fraction', 'units', '')
        nc_write_attribute(filename[i], 'surface_fraction', 'res_orig', domain_px[i])
        del surface_fraction

        nc_write_to_file_2d(filename[i], 'vegetation_type', vegetation_type,
                            datatypes["vegetation_type"], 'y', 'x', fillvalues["vegetation_type"])
        nc_write_attribute(filename[i], 'vegetation_type', 'long_name', 'vegetation type')
        nc_write_attribute(filename[i], 'vegetation_type', 'units', '')
        nc_write_attribute(filename[i], 'vegetation_type', 'res_orig', domain_px[i])
        nc_write_attribute(filename[i], 'vegetation_type', 'coordinates', 'E_UTM N_UTM lon lat')
        nc_write_attribute(filename[i], 'vegetation_type', 'grid_mapping', 'crs')
        del vegetation_type

        nc_write_to_file_2d(filename[i], 'pavement_type', pavement_type, datatypes["pavement_type"],
                            'y', 'x', fillvalues["pavement_type"])
        nc_write_attribute(filename[i], 'pavement_type', 'long_name', 'pavement type')
        nc_write_attribute(filename[i], 'pavement_type', 'units', '')
        nc_write_attribute(filename[i], 'pavement_type', 'res_orig', domain_px[i])
        nc_write_attribute(filename[i], 'pavement_type', 'coordinates', 'E_UTM N_UTM lon lat')
        nc_write_attribute(filename[i], 'pavement_type', 'grid_mapping', 'crs')
        del pavement_type

        nc_write_to_file_2d(filename[i], 'water_type', water_type, datatypes["water_type"], 'y',
                            'x', fillvalues["water_type"])
        nc_write_attribute(filename[i], 'water_type', 'long_name', 'water type')
        nc_write_attribute(filename[i], 'water_type', 'units', '')
        nc_write_attribute(filename[i], 'water_type', 'res_orig', domain_px[i])
        nc_write_attribute(filename[i], 'water_type', 'coordinates', 'E_UTM N_UTM lon lat')
        nc_write_attribute(filename[i], 'water_type', 'grid_mapping', 'crs')
        del water_type

        nc_write_to_file_2d(filename[i], 'soil_type', soil_type, datatypes["soil_type"], 'y', 'x',
                            fillvalues["soil_type"])
        nc_write_attribute(filename[i], 'soil_type', 'long_name', 'soil type')
        nc_write_attribute(filename[i], 'soil_type', 'units', '')
        nc_write_attribute(filename[i], 'soil_type', 'res_orig', domain_px[i])
        nc_write_attribute(filename[i], 'soil_type', 'coordinates', 'E_UTM N_UTM lon lat')
        nc_write_attribute(filename[i], 'soil_type', 'grid_mapping', 'crs')
        del soil_type

        del x
        del y

        # pixels with bridges get building_type = 7 = bridge. This does not change the _type
        # setting for the under-bridge area
        # NOTE: when bridges are present the consistency check will fail at the moment
        if domain_3d[i]:
            if np.any(~building_type.mask):
                bridges_2d = nc_read_from_file_2d(input_file_bridges_2d[ii[i]], 'Band1',
                                                  domain_x0[i], domain_x1[i], domain_y0[i],
                                                  domain_y1[i])
                ma.masked_equal(bridges_2d, 0.0, copy=False)
                building_type = ma.where(bridges_2d.mask, building_type, 7)
                nc_overwrite_to_file_2d(filename[i], 'building_type', building_type)

                del building_type
                del bridges_2d

    # Read/write street type and street crossings
    for i in range(0, ndomains):
        street_type = nc_read_from_file_2d(input_file_street_type[ii[i]], 'Band1', domain_x0[i],
                                           domain_x1[i], domain_y0[i], domain_y1[i])
        ma.masked_equal(street_type, 255, copy=False)
        street_type = ma.where(street_type < 1, defaultvalues["street_type"], street_type)

        pavement_type = nc_read_from_file_2d_all(filename[i], 'pavement_type')
        street_type.mask = ma.mask_or(pavement_type.mask, street_type.mask)

        nc_write_to_file_2d(filename[i], 'street_type', street_type, datatypes["street_type"], 'y',
                            'x', fillvalues["street_type"])
        nc_write_attribute(filename[i], 'street_type', 'long_name', 'street type')
        nc_write_attribute(filename[i], 'street_type', 'units', '')
        nc_write_attribute(filename[i], 'street_type', 'res_orig', domain_px[i])
        nc_write_attribute(filename[i], 'street_type', 'coordinates', 'E_UTM N_UTM lon lat')
        nc_write_attribute(filename[i], 'street_type', 'grid_mapping', 'crs')
        del street_type

        street_crossings = nc_read_from_file_2d(input_file_street_crossings[ii[i]], 'Band1',
                                                domain_x0[i], domain_x1[i], domain_y0[i],
                                                domain_y1[i])
        ma.masked_equal(street_crossings, 255, copy=False)
        street_crossings = ma.where(street_crossings < 1,
                                    defaultvalues["street_crossings"], street_crossings)

        nc_write_to_file_2d(filename[i], 'street_crossing', street_crossings,
                            datatypes["street_crossings"], 'y', 'x', fillvalues["street_crossings"])
        nc_write_attribute(filename[i], 'street_crossing', 'long_name', 'street crossings')
        nc_write_attribute(filename[i], 'street_crossing', 'units', '')
        nc_write_attribute(filename[i], 'street_crossing', 'res_orig', domain_px[i])
        nc_write_attribute(filename[i], 'street_crossing', 'coordinates', 'E_UTM N_UTM lon lat')
        nc_write_attribute(filename[i], 'street_crossing', 'grid_mapping', 'crs')
        del street_crossings

    # Read/write vegetation on roofs
    for i in range(0, ndomains):
        if domain_green_roofs[i]:
            green_roofs = nc_read_from_file_2d(input_file_vegetation_on_roofs[ii[i]], 'Band1',
                                               domain_x0[i], domain_x1[i], domain_y0[i],
                                               domain_y1[i])
            buildings_2d = nc_read_from_file_2d_all(filename[i], 'buildings_2d')

            x = nc_read_from_file_2d_all(filename[i], 'x')
            y = nc_read_from_file_2d_all(filename[i], 'y')
            nbuilding_pars = np.arange(0, 150)
            building_pars = ma.ones((len(nbuilding_pars), len(y), len(x)))
            building_pars[:, :, :] = ma.masked

            # assign green fraction on roofs
            building_pars[3, :, :] = ma.where(
                (~buildings_2d.mask) & (green_roofs != 0.0),
                1, ma.masked)

            # assign leaf area index for vegetation on roofs
            building_pars[4, :, :] = ma.where(
                (~(building_pars.mask[3, :, :])) & (green_roofs >= 0.5),
                settings_lai_roof_intensive, ma.masked)
            building_pars[4, :, :] = ma.where(
                (~(building_pars.mask[3, :, :])) & (green_roofs < 0.5),
                settings_lai_roof_extensive, building_pars[4, :, :])

            nc_write_dimension(filename[i], 'nbuilding_pars', nbuilding_pars,
                               datatypes["nbuilding_pars"])
            nc_write_to_file_3d(filename[i], 'building_pars', building_pars,
                                datatypes["building_pars"], 'nbuilding_pars', 'y', 'x',
                                fillvalues["building_pars"])
            nc_write_attribute(filename[i], 'building_pars', 'long_name', 'building_pars')
            nc_write_attribute(filename[i], 'building_pars', 'units', '')
            nc_write_attribute(filename[i], 'building_pars', 'res_orig', domain_px[i])
            nc_write_attribute(filename[i], 'building_pars', 'coordinates', 'E_UTM N_UTM lon lat')
            nc_write_attribute(filename[i], 'building_pars', 'grid_mapping', 'crs')

            del building_pars, buildings_2d, x, y

    # Read tree data and create LAD and BAD arrays using the canopy generator
    for i in range(0, ndomains):
        lai = nc_read_from_file_2d(input_file_lai[ii[i]], 'Band1', domain_x0[i], domain_x1[i],
                                   domain_y0[i], domain_y1[i])

        vegetation_type = nc_read_from_file_2d_all(filename[i], 'vegetation_type')

        lai.mask = ma.mask_or(vegetation_type.mask, lai.mask)

        x = nc_read_from_file_2d_all(filename[i], 'x')
        y = nc_read_from_file_2d_all(filename[i], 'y')
        nvegetation_pars = np.arange(0, 12)
        vegetation_pars = ma.ones((len(nvegetation_pars), len(y), len(x)))
        vegetation_pars[:, :, :] = ma.masked

        vegetation_pars[1, :, :] = lai

        # Write out first version of LAI. Will later be overwritten.
        nc_write_dimension(filename[i], 'nvegetation_pars', nvegetation_pars,
                           datatypes["nvegetation_pars"])
        nc_write_to_file_3d(filename[i], 'vegetation_pars', vegetation_pars,
                            datatypes["vegetation_pars"], 'nvegetation_pars', 'y', 'x',
                            fillvalues["vegetation_pars"])
        nc_write_attribute(filename[i], 'vegetation_pars', 'long_name', 'vegetation_pars')
        nc_write_attribute(filename[i], 'vegetation_pars', 'units', '')
        nc_write_attribute(filename[i], 'vegetation_pars', 'res_orig', domain_px[i])
        nc_write_attribute(filename[i], 'vegetation_pars', 'coordinates', 'E_UTM N_UTM lon lat')
        nc_write_attribute(filename[i], 'vegetation_pars', 'grid_mapping', 'crs')

        del lai, vegetation_pars, vegetation_type

    # Read tree data and create LAD and BAD arrays using the canopy generator
    for i in range(0, ndomains):
        if domain_street_trees[i]:

            vegetation_pars = nc_read_from_file_2d_all(filename[i], 'vegetation_pars')

            lai = nc_read_from_file_2d(input_file_lai[ii[i]], 'Band1', domain_x0[i], domain_x1[i],
                                       domain_y0[i], domain_y1[i])

            x = nc_read_from_file_2d_all(filename[i], 'x')
            y = nc_read_from_file_2d_all(filename[i], 'y')

            # Save lai data as default for low and high vegetation
            lai_low = lai
            lai_high = lai

            # Read all tree parameters from file
            tree_height = nc_read_from_file_2d(input_file_tree_height[ii[i]], 'Band1', domain_x0[i],
                                               domain_x1[i], domain_y0[i], domain_y1[i])
            ma.masked_where(tree_height <= 0.0, tree_height, copy=False)

            tree_crown_diameter = nc_read_from_file_2d(input_file_tree_crown_diameter[ii[i]],
                                                       'Band1', domain_x0[i], domain_x1[i],
                                                       domain_y0[i], domain_y1[i])
            ma.masked_where(tree_crown_diameter <= 0.0, tree_crown_diameter, copy=False)

            tree_trunk_diameter = nc_read_from_file_2d(input_file_tree_trunk_diameter[ii[i]],
                                                       'Band1', domain_x0[i], domain_x1[i],
                                                       domain_y0[i], domain_y1[i])
            ma.masked_where(tree_trunk_diameter <= 0.0, tree_trunk_diameter, copy=False)

            tree_type = nc_read_from_file_2d(input_file_tree_type[ii[i]], 'Band1', domain_x0[i],
                                             domain_x1[i], domain_y0[i], domain_y1[i])
            patch_height = nc_read_from_file_2d(input_file_patch_height[ii[i]], 'Band1',
                                                domain_x0[i], domain_x1[i], domain_y0[i],
                                                domain_y1[i])

            # Remove missing values from the data. Reasonable values will be set by the tree
            # generator
            ma.masked_where((tree_height == 0.0) | (tree_height == -1.0),
                            tree_height, copy=False)
            ma.masked_where(
                (tree_trunk_diameter == 0.0) | (tree_trunk_diameter == -1.0),
                tree_trunk_diameter, copy=False)
            ma.masked_where((tree_type == 0.0) | (tree_type == -1.0),
                            tree_type, copy=False)

            # For vegetation pixel with missing height information (-1), set default patch height
            patch_height = ma.where(patch_height == -1.0, settings_patch_height_default,
                                    patch_height)

            # Convert trunk diameter from cm to m
            tree_trunk_diameter = tree_trunk_diameter * 0.01

            # Todo: Do we need the following?
            # Temporarily change missing value for tree_type
            # tree_type = np.where((tree_type == fillvalues["tree_type"]), fillvalues["tree_data"],
            #                     tree_type)

            # Compare patch height array with vegetation type and correct accordingly
            vegetation_type = nc_read_from_file_2d_all(filename[i], 'vegetation_type')

            # For zero-height patches, set vegetation_type to short grass and remove these pixels
            # from the patch height array
            # ma.where gives ma.masked when its first argument is ma.masked. We don't want this for
            # patch_height here so do an extra check and use .data
            vegetation_type = ma.where((~patch_height.mask) & (patch_height.data == 0.0) &
                                       ma_isin(vegetation_type, vt_high_vegetation),
                                       3, vegetation_type)
            ma.masked_where((patch_height == 0.0) & ma_isin(vegetation_type, vt_high_vegetation),
                            patch_height, copy=False)

            max_tree_height = ma.max(tree_height)
            max_patch_height = ma.max(patch_height)

            # Call canopy generator for single trees only if there is any tree height available
            # in the domain. This does not guarantee that there are street trees that can be
            # processed. This is checked in the canopy generator.
            if (max_tree_height is not ma.masked) | (max_patch_height is not ma.masked):

                lad, bad, tree_ids, tree_types, zlad = \
                    generate_single_tree_lad(x, y, domain_dz[i],
                                             max_tree_height, max_patch_height,
                                             tree_type, tree_height,
                                             tree_crown_diameter, tree_trunk_diameter, lai,
                                             settings_season, settings_lai_tree_lower_threshold,
                                             domain_remove_low_lai_tree[i])

                # Remove LAD volumes that are inside buildings
                if not domain_overhanging_trees[i]:
                    buildings_2d = nc_read_from_file_2d_all(filename[i], 'buildings_2d')
                    for k in range(0, len(zlad)):
                        ma.masked_where(~buildings_2d.mask, lad[k, :, :], copy=False)
                        ma.masked_where(~buildings_2d.mask, bad[k, :, :], copy=False)
                        ma.masked_where(~buildings_2d.mask, tree_ids[k, :, :], copy=False)

                    del buildings_2d

                nc_write_dimension(filename[i], 'zlad', zlad, datatypes["tree_data"])

                nc_write_to_file_3d(filename[i], 'lad', lad, datatypes["tree_data"], 'zlad', 'y',
                                    'x', fillvalues["tree_data"])
                nc_write_attribute(filename[i], 'lad', 'long_name', 'leaf area density')
                nc_write_attribute(filename[i], 'lad', 'units', '')
                nc_write_attribute(filename[i], 'lad', 'res_orig', domain_px[i])
                nc_write_attribute(filename[i], 'lad', 'coordinates', 'E_UTM N_UTM lon lat')
                nc_write_attribute(filename[i], 'lad', 'grid_mapping', 'crs')

                nc_write_to_file_3d(filename[i], 'bad', bad, datatypes["tree_data"], 'zlad', 'y',
                                    'x', fillvalues["tree_data"])
                nc_write_attribute(filename[i], 'bad', 'long_name', 'basal area density')
                nc_write_attribute(filename[i], 'bad', 'units', '')
                nc_write_attribute(filename[i], 'bad', 'res_orig', domain_px[i])
                nc_write_attribute(filename[i], 'bad', 'coordinates', 'E_UTM N_UTM lon lat')
                nc_write_attribute(filename[i], 'bad', 'grid_mapping', 'crs')

                nc_write_to_file_3d(filename[i], 'tree_id', tree_ids, datatypes["tree_id"], 'zlad',
                                    'y', 'x', fillvalues["tree_id"])
                nc_write_attribute(filename[i], 'tree_id', 'long_name', 'tree id')
                nc_write_attribute(filename[i], 'tree_id', 'units', '')
                nc_write_attribute(filename[i], 'tree_id', 'res_orig', domain_px[i])
                nc_write_attribute(filename[i], 'tree_id', 'coordinates', 'E_UTM N_UTM lon lat')
                nc_write_attribute(filename[i], 'tree_id', 'grid_mapping', 'crs')

                nc_write_to_file_3d(filename[i], 'tree_type', tree_types, datatypes["tree_types"],
                                    'zlad', 'y', 'x', fillvalues["tree_types"])
                nc_write_attribute(filename[i], 'tree_type', 'long_name', 'tree type')
                nc_write_attribute(filename[i], 'tree_type', 'units', '')
                nc_write_attribute(filename[i], 'tree_type', 'res_orig', domain_px[i])
                nc_write_attribute(filename[i], 'tree_type', 'coordinates', 'E_UTM N_UTM lon lat')
                nc_write_attribute(filename[i], 'tree_type', 'grid_mapping', 'crs')

                del lai, lad, bad, tree_ids, tree_types, zlad

            else:
                print('No street trees generated in domain ' + str(i))

            del vegetation_pars, tree_height, tree_crown_diameter, tree_trunk_diameter, tree_type, \
                patch_height, x, y

    # Create vegetation patches for locations with high vegetation type
    for i in range(0, ndomains):
        if domain_canopy_patches[i]:

            patch_height = nc_read_from_file_2d(input_file_patch_height[ii[i]], 'Band1',
                                                domain_x0[i], domain_x1[i], domain_y0[i],
                                                domain_y1[i])

            # For vegetation pixel with missing height information (-1), set default patch height
            patch_height = ma.where(patch_height == -1.0, settings_patch_height_default,
                                    patch_height)

            max_patch_height = ma.max(patch_height)
            if max_patch_height is not ma.masked:
                # Call canopy generator for single trees only if there is any tree height
                # available in the domain. This does not guarantee that there are street trees
                # that can be processed. This is checked in the canopy generator.

                # Load vegetation_type and lad array (at level z = 0) for re-processing
                vegetation_type = nc_read_from_file_2d_all(filename[i], 'vegetation_type')
                lad = nc_read_from_file_3d_all(filename[i], 'lad')
                tree_id = nc_read_from_file_3d_all(filename[i], 'tree_id')
                tree_types = nc_read_from_file_3d_all(filename[i], 'tree_type')
                zlad = nc_read_from_file_1d_all(filename[i], 'zlad')
                vegetation_pars = nc_read_from_file_3d_all(filename[i], 'vegetation_pars')
                lai = nc_read_from_file_2d(input_file_lai[ii[i]], 'Band1', domain_x0[i],
                                           domain_x1[i], domain_y0[i], domain_y1[i])
                patch_type_2d = nc_read_from_file_2d(input_file_patch_type[ii[i]], 'Band1',
                                                     domain_x0[i], domain_x1[i], domain_y0[i],
                                                     domain_y1[i])

                # patch_type_2d: use high vegetation vegetation_type if patch_type is missing
                # Note: vegetation_type corrected above
                patch_type_2d = ma.where(patch_type_2d.mask &
                                         ma_isin(vegetation_type, vt_high_vegetation),
                                         -vegetation_type, patch_type_2d)
                # patch_type_2d: use default value for the rest of the pixels
                patch_type_2d = ma.where(patch_type_2d.mask,
                                         defaultvalues["patch_type"], patch_type_2d)

                # Determine all pixels that do not already have an LAD but which are high
                # vegetation to a dummy value of 1.0 and remove all other pixels
                lai_high = ma.where(
                    (lad.mask[0, :, :]) &
                    (
                            ma_isin(vegetation_type, vt_high_vegetation) &
                            (patch_height.mask | (patch_height.data >= domain_dz[i]))
                    ),
                    1.0, ma.masked)

                # Treat all pixels where short grass is defined, but where a patch_height >= dz
                # is found, as high vegetation (often the case in backyards)
                lai_high = ma.where(
                    lai_high.mask &
                    ~patch_height.mask & (patch_height.data >= domain_dz[i]) &
                    ~vegetation_type.mask & (vegetation_type.data == 3),
                    1.0, lai_high)

                # If overhanging trees are allowed, assign pixels with patch_height > dz that are
                # not included in vegetation_type to high
                if domain_overhanging_trees[i]:
                    lai_high = ma.where(
                        lai_high.mask &
                        ~patch_height.mask & (patch_height.data >= domain_dz[i]),
                        1.0, lai_high)

                # Now, assign either the default LAI for high vegetation or keep 1.0 from the
                # lai_high array.
                lai_high = ma.where(~lai_high.mask & lai.mask,
                                    settings_lai_high_default, lai_high)

                # If LAI values are available in the LAI array, write them on the lai_high array
                lai_high = ma.where(~lai_high.mask & ~lai.mask,
                                    lai, lai_high)

                # Define a patch height wherever it is missing, but where a high vegetation LAI
                # was set
                patch_height = ma.where(~lai_high.mask & patch_height.mask,
                                        settings_patch_height_default, patch_height)

                # Remove pixels where street trees were already set
                ma.masked_where(~lad.mask[0, :, :], patch_height, copy=False)

                # Remove patch heights that have no lai_high value
                ma.masked_where(lai_high.mask, patch_height, copy=False)

                # For missing LAI values, set either the high vegetation default or the low
                # vegetation default
                lai_high = ma.where(lai_high.mask & ~patch_height.mask & (patch_height.data > 2.0),
                                    settings_lai_high_default, lai_high)
                lai_high = ma.where(lai_high.mask & ~patch_height.mask & (patch_height.data <= 2.0),
                                    settings_lai_low_default, lai_high)

                if ma.max(patch_height) >= (2.0 * domain_dz[i]):
                    print("    start calculating LAD (this might take some time)")

                    lad_patch, patch_id, patch_types, patch_nz, status = \
                        process_patch(domain_dz[i],
                                      patch_height,
                                      patch_type_2d,
                                      vegetation_type,
                                      max(zlad),
                                      lai_high,
                                      settings_lai_alpha,
                                      settings_lai_beta)

                    # Set negative ids for patches
                    patch_id = ma.where(patch_id.mask, patch_id, -patch_id)

                    # 2D loop in order to avoid memory problems with large arrays
                    for iii in range(0, domain_nx[i] + 1):
                        for jj in range(0, domain_ny[i] + 1):
                            tree_id[0:patch_nz + 1, jj, iii] = ma.where(
                                lad.mask[0:patch_nz + 1, jj, iii],
                                patch_id[0:patch_nz + 1, jj, iii], tree_id[0:patch_nz + 1, jj, iii])
                            tree_types[0:patch_nz + 1, jj, iii] = ma.where(
                                lad.mask[0:patch_nz + 1, jj, iii],
                                patch_types[0:patch_nz + 1, jj, iii],
                                tree_types[0:patch_nz + 1, jj, iii])
                            lad[0:patch_nz + 1, jj, iii] = ma.where(
                                lad.mask[0:patch_nz + 1, jj, iii],
                                lad_patch[0:patch_nz + 1, jj, iii], lad[0:patch_nz + 1, jj, iii])

                # Remove high vegetation wherever it is replaced by a leaf area density. This
                # should effectively remove all high vegetation pixels
                vegetation_type = ma.where(~lad.mask[0, :, :] & ~vegetation_type.mask,
                                           settings_veg_type_below_trees, vegetation_type)

                # Set default low LAI for pixels with an LAD (short grass below trees)
                lai_low = ma.where(lad.mask[0, :, :], lai, settings_lai_low_default)

                # Fill low vegetation pixels without LAI set or with LAI = 0 with default value
                lai_low = ma.where((lai_low.mask | (lai_low == 0.0)) & ~vegetation_type.mask,
                                   settings_lai_low_default, lai_low)

                # Remove lai for pixels that have no vegetation_type
                lai_low = ma.where(~vegetation_type.mask & (vegetation_type != 1),
                                   lai_low, ma.masked)

                # Overwrite lai in vegetation_parameters
                vegetation_pars[1, :, :] = ma.copy(lai_low)
                nc_overwrite_to_file_3d(filename[i], 'vegetation_pars', vegetation_pars)

                # Overwrite lad and id arrays
                nc_overwrite_to_file_3d(filename[i], 'lad', lad)
                nc_overwrite_to_file_3d(filename[i], 'tree_id', tree_id)
                nc_overwrite_to_file_3d(filename[i], 'tree_type', tree_types)

                nc_overwrite_to_file_2d(filename[i], 'vegetation_type', vegetation_type)

                del vegetation_type, lad, lai, patch_height, vegetation_pars, zlad
            else:
                print('No tree patches found in domain ' + str(i))

    # Final adjustment of vegetation parameters: remove LAI where a bare soil was set
    for i in range(0, ndomains):
        vegetation_type = nc_read_from_file_2d_all(filename[i], 'vegetation_type')
        vegetation_pars = nc_read_from_file_3d_all(filename[i], 'vegetation_pars')
        lai = vegetation_pars[1, :, :]

        # Remove lai for pixels that have no vegetation_type
        ma.masked_where(vegetation_type.mask | (vegetation_type == 1), lai, copy=False)

        # Overwrite lai in vegetation_parameters
        vegetation_pars[1, :, :] = ma.copy(lai)
        nc_overwrite_to_file_3d(filename[i], 'vegetation_pars', vegetation_pars)

        del vegetation_type, lai, vegetation_pars

    # Read/write water temperature
    for i in range(0, ndomains):
        if domain_water_temperatures[i] is not None or \
           input_file_water_temperature[ii[i]] is not None:

            # Read water type from output file and create water_pars
            water_type = nc_read_from_file_2d_all(filename[i], 'water_type')

            x = nc_read_from_file_1d_all(filename[i], 'x')
            y = nc_read_from_file_1d_all(filename[i], 'y')
            nwater_pars = np.arange(0, 7)
            water_pars = ma.masked_all((len(nwater_pars), len(y), len(x)))

            # Assign water temperature
            # First, set default value for all water surfaces
            water_pars[0, :, :] = ma.where(~water_type.mask, defaultvalues['water_temperature'],
            ma.masked)

            # Set specific water temperature per type as assigned in config
            if domain_water_temperatures[i] is not None:
                # Convert string from config file into list per water type
                water_temperature_list = []
                for temperature in domain_water_temperatures[i].split(","):
                    try:
                        water_temperature_list.append(float(temperature))
                    except ValueError:
                        water_temperature_list.append(None)

                for water_type_index, water_temperature in enumerate(water_temperature_list,
                                                                     start=1):
                    if water_temperature is not None:
                        water_pars[0, :, :] = ma.where(water_type == water_type_index,
                                                       water_temperature,
                                                       water_pars[0, :, :])

            # Set water temperature based on input file
            if input_file_water_temperature[ii[i]] is not None:
                water_temperature_from_file = \
                        nc_read_from_file_2d(input_file_water_temperature[ii[i]], 'Band1',
                                             domain_x0[i], domain_x1[i], domain_y0[i], domain_y1[i])
                water_temperature_from_file.mask = ma.mask_or(water_temperature_from_file.mask,
                                                              water_type.mask)
                water_temperature_from_file = ma.where((water_temperature_from_file < 265.0) &
                                                       (water_temperature_from_file > 373.15),
                                                       ma.masked,
                                                       water_temperature_from_file)
                water_pars[0, :, :] = ma.where(~water_temperature_from_file.mask,
                                               water_temperature_from_file,
                                               water_pars[0, :, :])
                del water_temperature_from_file

            nc_write_dimension(filename[i], 'nwater_pars', nwater_pars, datatypes["nwater_pars"])
            nc_write_to_file_3d(filename[i], 'water_pars', water_pars, datatypes["water_pars"],
                                'nwater_pars', 'y', 'x', fillvalues["water_pars"])
            nc_write_attribute(filename[i], 'water_pars', 'long_name', 'water_pars')
            nc_write_attribute(filename[i], 'water_pars', 'units', '')
            nc_write_attribute(filename[i], 'water_pars', 'res_orig', domain_px[i])
            nc_write_attribute(filename[i], 'water_pars', 'coordinates', 'E_UTM N_UTM lon lat')
            nc_write_attribute(filename[i], 'water_pars', 'grid_mapping', 'crs')

            del water_pars, water_type, x, y

    # Final consistency check
    for i in range(0, ndomains):
        vegetation_type = nc_read_from_file_2d_all(filename[i], 'vegetation_type')
        pavement_type = nc_read_from_file_2d_all(filename[i], 'pavement_type')
        building_type = nc_read_from_file_2d_all(filename[i], 'building_type')
        water_type = nc_read_from_file_2d_all(filename[i], 'water_type')
        soil_type = nc_read_from_file_2d_all(filename[i], 'soil_type')

        # Check for consistency and fill empty fields with default vegetation type
        consistency_array, test = check_consistency_4(vegetation_type, building_type, pavement_type,
                                                      water_type)

        # Check for consistency and fill empty fields with default vegetation type
        consistency_array, test = check_consistency_3(vegetation_type, pavement_type, soil_type)

        surface_fraction = nc_read_from_file_3d_all(filename[i], 'surface_fraction')
        surface_fraction[0, :, :] = ma.where(vegetation_type.mask, 0.0, 1.0)
        surface_fraction[1, :, :] = ma.where(pavement_type.mask, 0.0, 1.0)
        surface_fraction[2, :, :] = ma.where(water_type.mask, 0.0, 1.0)
        nc_overwrite_to_file_3d(filename[i], 'surface_fraction', surface_fraction)

        del vegetation_type, pavement_type, building_type, water_type, soil_type
