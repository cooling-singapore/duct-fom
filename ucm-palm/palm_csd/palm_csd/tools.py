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
# Support routines for palm_csd
#
# @Author Bjoern Maronga (maronga@muk.uni-hannover.de)
# ------------------------------------------------------------------------------ #
import numpy as np
import numpy.ma as ma
from scipy.interpolate import interp2d


def blend_array_2d(array1, array2, radius):
    # Blend over the parent and child terrain height within a given radius

    gradient_matrix = np.ones(array1.shape)
    radius = int(radius)

    for j in range(0, radius):
        gradient_matrix[:, j] = float(j) / float(radius)
        gradient_matrix[:, -j - 1] = float(j) / float(radius)
        gradient_matrix[j, :] = float(j) / float(radius)
        gradient_matrix[-j - 1, :] = float(j) / float(radius)

    for j in range(0, radius):
        for i in range(0, radius):
            gradient_matrix[j, i] = max(1 - np.sqrt((i - (0. + radius))**2
                                                    + (j - (0. + radius))**2) / radius, 0)
            gradient_matrix[-j - 1, i] = max(1 - np.sqrt((i - (0. + radius))**2
                                                         + (j - (0. + radius))**2) / radius, 0)
            gradient_matrix[j, -i - 1] = max(1 - np.sqrt((i - (0. + radius))**2
                                                         + (j - (0. + radius))**2) / radius, 0)
            gradient_matrix[-j - 1, -i - 1] = max(1 - np.sqrt((i - (0. + radius))**2
                                                              + (j - (0. + radius))**2) / radius, 0)

    array_blended = array1 * gradient_matrix + (1.0 - gradient_matrix) * array2

    return array_blended


def interpolate_2d(array, x1, y1, x2, y2):
    tmp_int2d = interp2d(x1, y1, array, kind='linear')
    array_ip = tmp_int2d(x2.astype(float), y2.astype(float))

    return array_ip


def bring_to_palm_grid(array, x, y, dz):
    # Bring the parent terrain height to the child grid
    k_tmp = np.arange(0, max(array.flatten()) + dz * 2, dz)
    k_tmp[1:] = k_tmp[1:] - dz * 0.5
    for n in range(0, len(x)):
        for m in range(0, len(y)):
            for k in range(1, len(k_tmp + 1)):
                if k_tmp[k] > array[m, n]:
                    array[m, n] = k_tmp[k - 1] + dz * 0.5
                    break

    return array


def make_3d_from_2d(array_2d, x, y, dz):
    k_tmp = np.arange(0, ma.max(array_2d) + dz * 2, dz)  # ma.max flattens argument by default

    k_tmp[1:] = k_tmp[1:] - dz * 0.5
    array_3d = ma.ones((len(k_tmp), len(y), len(x)), dtype=np.byte)

    for n in range(0, len(x)):
        for m in range(0, len(y)):
            if array_2d[m, n] is ma.masked:
                array_3d[:, m, n] = 0
            else:
                for k in range(0, len(k_tmp)):
                    if k_tmp[k] > array_2d.data[m, n]:
                        array_3d[k, m, n] = 0

    return array_3d, k_tmp


def make_3d_from_bridges_2d(array_3d, array_2d, x, y, dz, width):
    for n in range(0, len(x)):
        for m in range(0, len(y)):
            if ~array_2d.mask[m, n]:
                print(str(n) + "/" + str(m) + "/")
                k_min = max(int((array_2d[m, n] - width) / dz), 0)
                k_max = int(round(array_2d[m, n] / dz))
                array_3d[k_min:k_max + 1, m, n] = 1

    return array_3d.astype(np.byte)


def check_arrays_2(array1, array2):
    mask1 = array1.mask
    mask2 = array2.mask
    result = np.array_equal(mask1, mask2)

    return result


def check_consistency_3(array1, array2, array3):
    # Todo: is -1 for array3 correct?
    tmp_array = np.where(array1.mask, 0, 1) + np.where(array2.mask, 0, 1) + \
                np.where(array3.mask, 0, -1)

    test = np.any(tmp_array != 0)
    if test:
        print("soil_type array is not consistent!")
        print("max: " + str(max(tmp_array.flatten())) + ", min: " + str(min(tmp_array.flatten())))
    else:
        print("soil_type array is consistent!")
    return tmp_array, test


# Check if at every point only one of the arrays is not masked
def check_consistency_4(array1, array2, array3, array4):
    tmp_array = np.where(array1.mask, 0, 1) + np.where(array2.mask, 0, 1) + \
                np.where(array3.mask, 0, 1) + np.where(array4.mask, 0, 1)

    test = np.any(tmp_array != 1)
    if test:
        print("*_type arrays are not consistent!")
        print("max: " + str(max(tmp_array.flatten())) + ", min: " + str(min(tmp_array.flatten())))
    else:
        print("*_type arrays are consistent!")
    return tmp_array, test


# Get option entry in section of config. path is prepended if not None. If optional, return None if
# option is not found in the configuration.
def get_filename(config, section, option, path, optional=False):
    if optional:
        filename = config.get(section, option, fallback=None)
    else:
        filename = config.get(section, option)
    if (filename is not None) & (path is not None):
        filename = path + "/" + filename
    return filename


# Get option entry in section of config. If option is in default, its value is the fallback.
# The expected type of the variable can be defined via variable_type. The default is a string.
# Data type is handled here because getboolean is quite clever and complicated to handle later.
# The other cases could be casted later but e.g. float(None) also does not work.
def get_setting(config, section, option, default, variable_type=str):

    if variable_type is str:
        get_function = getattr(config, 'get')
    elif variable_type is bool:
        get_function = getattr(config, 'getboolean')
    elif variable_type is int:
        get_function = getattr(config, 'getint')
    elif variable_type is float:
        get_function = getattr(config, 'getfloat')
    else:
        raise TypeError

    if option in default:
        setting = get_function(section, option, fallback=default[option])
    else:
        setting = get_function(section, option)

    return setting


# check for each element of array if it is in comparison
# Todo: use just ma.isin() once the bug in https://github.com/numpy/numpy/issues/19877 or
#  https://stackoverflow.com/questions/69160969/ is fixed
def ma_isin(array, comparison):
    return ma.MaskedArray(
        data=np.isin(array, comparison),
        mask=array.mask.copy())
