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
# NetCDF interface routines for palm_csd
#
# @Author Bjoern Maronga (maronga@muk.uni-hannover.de)
# ------------------------------------------------------------------------------ #

from netCDF4 import Dataset
import numpy.ma as ma


def nc_read_from_file_1d_all(filename, varname):

    try:
        f = open(filename)
        f.close()
    except FileNotFoundError:
        print("Error: " + filename + ". No such file. Aborting...")
        raise

    nc_file = Dataset(filename, "r+", format="NETCDF4")
    tmp_array = nc_file.variables[varname][:]
    nc_file.close()
    return tmp_array


def nc_read_from_file_2d(filename, varname, x0, x1, y0, y1):

    if filename is not None:
        try:
            f = open(filename)
            f.close()
        except FileNotFoundError:
            print("Error: " + filename + ". No such file. Aborting...")
            raise

        nc_file = Dataset(filename, "r", format="NETCDF4")
        tmp_array = nc_file.variables[varname][y0:y1 + 1, x0:x1 + 1]
        nc_file.close()
    else:
        tmp_array = ma.masked_all((y1 - y0 + 1, x1 - x0 + 1))

    return tmp_array


def nc_read_from_file_2d_all(filename, varname):

    try:
        f = open(filename)
        f.close()
    except FileNotFoundError:
        print("Error: " + filename + ". No such file. Aborting...")
        raise

    nc_file = Dataset(filename, "r+", format="NETCDF4")
    tmp_array = nc_file.variables[varname][:][:]
    nc_file.close()
    return tmp_array


def nc_read_from_file_3d_all(filename, varname):

    try:
        f = open(filename)
        f.close()
    except FileNotFoundError:
        print("Error: " + filename + ". No such file. Aborting...")
        raise

    nc_file = Dataset(filename, "r+", format="NETCDF4")
    tmp_array = nc_file.variables[varname][:][:][:]
    nc_file.close()
    return tmp_array


def nc_read_from_file_1d(filename, varname, x0, x1):

    try:
        f = open(filename)
        f.close()
    except FileNotFoundError:
        print("Error: " + filename + ". No such file. Aborting...")
        raise

    nc_file = Dataset(filename, "r", format="NETCDF4")
    tmp_array = nc_file.variables[varname][x0:x1 + 1]
    nc_file.close()
    return tmp_array


def nc_read_from_file_crs(filename, varname):
    """Return coordinate reference system from file."""

    try:
        f = open(filename)
        f.close()
    except FileNotFoundError:
        print("Error: " + filename + ". No such file. Aborting...")
        raise

    nc_file = Dataset(filename, "r", format="NETCDF4")
    crs_from_file = nc_file.variables[nc_file.variables[varname].grid_mapping]

    # Get EPSG code from crs
    try:
        epsg_code = crs_from_file.epsg_code
    except AttributeError:
        epsg_code = "unknown"
        if (crs_from_file.spatial_ref.find("ETRS89", 0, 100) and
                crs_from_file.spatial_ref.find("UTM", 0, 100)):
            if crs_from_file.spatial_ref.find("28N", 0, 100) != -1:
                epsg_code = "EPSG:25828"
            elif crs_from_file.spatial_ref.find("29N", 0, 100) != -1:
                epsg_code = "EPSG:25829"
            elif crs_from_file.spatial_ref.find("30N", 0, 100) != -1:
                epsg_code = "EPSG:25830"
            elif crs_from_file.spatial_ref.find("31N", 0, 100) != -1:
                epsg_code = "EPSG:25831"
            elif crs_from_file.spatial_ref.find("32N", 0, 100) != -1:
                epsg_code = "EPSG:25832"
            elif crs_from_file.spatial_ref.find("33N", 0, 100) != -1:
                epsg_code = "EPSG:25833"
            elif crs_from_file.spatial_ref.find("34N", 0, 100) != -1:
                epsg_code = "EPSG:25834"
            elif crs_from_file.spatial_ref.find("35N", 0, 100) != -1:
                epsg_code = "EPSG:25835"
            elif crs_from_file.spatial_ref.find("36N", 0, 100) != -1:
                epsg_code = "EPSG:25836"
            elif crs_from_file.spatial_ref.find("37N", 0, 100) != -1:
                epsg_code = "EPSG:25837"

    crs_var = CoordinateReferenceSystem(
        long_name="coordinate reference system",
        grid_mapping_name=crs_from_file.grid_mapping_name,
        semi_major_axis=crs_from_file.semi_major_axis,
        inverse_flattening=crs_from_file.inverse_flattening,
        longitude_of_prime_meridian=crs_from_file.longitude_of_prime_meridian,
        longitude_of_central_meridian=crs_from_file.longitude_of_central_meridian,
        scale_factor_at_central_meridian=crs_from_file.scale_factor_at_central_meridian,
        latitude_of_projection_origin=crs_from_file.latitude_of_projection_origin,
        false_easting=crs_from_file.false_easting,
        false_northing=crs_from_file.false_northing,
        spatial_ref=crs_from_file.spatial_ref,
        units="m",
        epsg_code=epsg_code,
    )

    return crs_var


def nc_create_file(filename):

    try:
        f = Dataset(filename, "w", format="NETCDF4")
        f.close()
        print("Created: " + filename + ".")
    except FileNotFoundError:
        print("Error. Could not create file: " + filename + ". Aborting...")
        raise

    return 0


# Write global attributes to the netcdf filename. Function arguments that are None are not added.
def nc_write_global_attributes(filename, origin_x, origin_y, origin_lat, origin_lon, origin_time,
                               global_acronym, global_angle,
                               global_author, global_campaign, global_comment, global_contact,
                               global_data_content,
                               global_dependencies, global_institution, global_keywords,
                               global_location, global_palm_version,
                               global_references, global_site, global_source, global_version):

    print("Writing global attributes to file...")

    f = Dataset(filename, "a", format="NETCDF4")

    f.setncattr('Conventions', "CF-1.7")

    if origin_x is not None:
        f.origin_x = origin_x
    if origin_y is not None:
        f.origin_y = origin_y
    if origin_time is not None:
        f.origin_time = origin_time
    if origin_lat is not None:
        f.origin_lat = origin_lat
    if origin_lon is not None:
        f.origin_lon = origin_lon

    if global_acronym is not None:
        f.acronym = global_acronym
    if global_angle is not None:
        f.rotation_angle = global_angle
    if global_author is not None:
        f.author = global_author
    if global_campaign is not None:
        f.campaign = global_campaign
    if global_comment is not None:
        f.comment = global_comment
    if global_contact is not None:
        f.contact = global_contact
    if global_data_content is not None:
        f.data_content = global_data_content
    if global_dependencies is not None:
        f.dependencies = global_dependencies
    if global_institution is not None:
        f.institution = global_institution
    if global_keywords is not None:
        f.keywords = global_keywords
    if global_location is not None:
        f.location = global_location
    if global_palm_version is not None:
        f.palm_version = global_palm_version
    if global_references is not None:
        f.references = global_references
    if global_site is not None:
        f.site = global_site
    if global_source is not None:
        f.source = global_source
    if global_version is not None:
        f.version = global_version

    f.close()

    return 0


def nc_write_global_attribute(filename, attribute, value):

    print("Writing attribute " + attribute + " to file...")

    f = Dataset(filename, "a", format="NETCDF4")

    f.setncattr(attribute, value)

    f.close()


def nc_write_dimension(filename, varname, array, datatype):
    try:
        f = Dataset(filename, "a", format="NETCDF4")
    except FileNotFoundError:
        print("Error. Could not open file: " + filename + ". Aborting...")
        raise

    print("Writing dimension " + varname + " to file...")

    f.createDimension(varname, len(array))
    temp = f.createVariable(varname, datatype, varname)
    temp[:] = array

    f.close()

    return 0


def nc_write_to_file_2d(filename, varname, array, datatype, dimname1, dimname2, fillvalue):
    try:
        f = Dataset(filename, "a", format="NETCDF4")
    except FileNotFoundError:
        print("Error. Could not open file: " + filename + ". Aborting...")
        raise

    print("Writing array " + varname + " to file...")

    temp = f.createVariable(varname, datatype, (dimname1, dimname2), fill_value=fillvalue)
    temp[:] = array

    f.close()

    return 0


def nc_overwrite_to_file_2d(filename, varname, array):
    try:
        f = Dataset(filename, "a", format="NETCDF4")
    except FileNotFoundError:
        print("Error. Could not open file: " + filename + ". Aborting...")
        raise

    print("Writing array " + varname + " to file...")

    temp = f.variables[varname]
    temp[:, :] = array

    f.close()

    return 0


def nc_overwrite_to_file_3d(filename, varname, array):
    try:
        f = Dataset(filename, "a", format="NETCDF4")
    except FileNotFoundError:
        print("Error. Could not open file: " + filename + ". Aborting...")
        raise

    print("Writing array " + varname + " to file...")

    temp = f.variables[varname]
    temp[:, :, :] = array

    f.close()

    return 0


def nc_write_to_file_3d(filename, varname, array, datatype, dimname1, dimname2, dimname3,
                        fillvalue):
    try:
        f = Dataset(filename, "a", format="NETCDF4")
    except FileNotFoundError:
        print("Error. Could not open file: " + filename + ". Aborting...")
        raise

    print("Writing array " + varname + " to file...")

    temp = f.createVariable(varname, datatype, (dimname1, dimname2, dimname3), fill_value=fillvalue)
    temp[:, :, :] = array

    f.close()

    return 0


def nc_write_attribute(filename, variable, attribute, value):
    f = Dataset(filename, "a", format="NETCDF4")

    var = f.variables[variable]
    var.setncattr(attribute, value)

    f.close()

    return 0


def nc_write_crs(filename, crs_var):
    try:
        f = Dataset(filename, "a", format="NETCDF4")
    except FileNotFoundError:
        print("Error. Could not open file: " + filename + ". Aborting...")
        raise

    print("Writing crs to file...")

    temp = f.createVariable("crs", "i")

    temp.long_name = crs_var.long_name
    temp.grid_mapping_name = crs_var.grid_mapping_name
    temp.semi_major_axis = crs_var.semi_major_axis
    temp.inverse_flattening = crs_var.inverse_flattening
    temp.longitude_of_prime_meridian = crs_var.longitude_of_prime_meridian
    temp.longitude_of_central_meridian = crs_var.longitude_of_central_meridian
    temp.scale_factor_at_central_meridian = crs_var.scale_factor_at_central_meridian
    temp.latitude_of_projection_origin = crs_var.latitude_of_projection_origin
    temp.false_easting = crs_var.false_easting
    temp.false_northing = crs_var.false_northing
    temp.spatial_ref = crs_var.spatial_ref
    temp.units = crs_var.units
    temp.epsg_code = crs_var.epsg_code

    f.close()

    return 0


class CoordinateReferenceSystem:
    def __init__(self, long_name, grid_mapping_name, semi_major_axis, inverse_flattening,
                 longitude_of_prime_meridian, longitude_of_central_meridian,
                 scale_factor_at_central_meridian, latitude_of_projection_origin, false_easting,
                 false_northing, spatial_ref, units, epsg_code):
        self.long_name = long_name
        self.grid_mapping_name = grid_mapping_name
        self.semi_major_axis = semi_major_axis
        self.inverse_flattening = inverse_flattening
        self.longitude_of_prime_meridian = longitude_of_prime_meridian
        self.longitude_of_central_meridian = longitude_of_central_meridian
        self.scale_factor_at_central_meridian = scale_factor_at_central_meridian
        self.latitude_of_projection_origin = latitude_of_projection_origin
        self.false_easting = false_easting
        self.false_northing = false_northing
        self.spatial_ref = spatial_ref
        self.units = units
        self.epsg_code = epsg_code
