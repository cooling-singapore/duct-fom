def Volumetric_averaged_building_height(Image):
    import numpy as np
    import math

    eps = 10 ** (-8)
    temp = np.multiply(Image, Image)

    num_h = np.nansum(temp)
    den_h = np.nansum(Image)+eps
    h = num_h / den_h
    if math.isnan(h):
        h = 0.001

    return h