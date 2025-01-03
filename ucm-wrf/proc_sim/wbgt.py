"""
Estimation of Wet-bulb Globe Temperature (WBGT) with WRF output.
This is a python version of the HeatMetrics R package by Spangler et al. (2019), which is based on the method for estimating WBGT by Liljegren, et al. 2008. The package is converted to python with minor changes for implementation in DUCT.

For the DUCT, we set these as constants:
- All grids are considered as 'Urban' (for stability class for wind speed estimation)
- Assume lapse rate of -6.5°C/km for wind speed estimation
- Set all wind speed height at 10m

Original C code:
Liljegren, J. C., Carhart, R. A., Lawday, P., Tschopp, S., & Sharp, R. (2008). Modeling the wet bulb globe temperature using standard meteorological measurements. Journal of Occupational and Environmental Hygiene, 5(10), 645–655. https://doi.org/10.1080/15459620802310770

R package ‘Heatmetrics’:
Spangler, K. R., Liang, S., & Wellenius, G. A. (2022). Wet‐bulb globe temperature, universal thermal climate index, and other heat metrics for US counties, 2000–2020. Scientific Data, 9(1), 326. https://doi.org/10.1038/s41597‐022‐01405‐3
"""

import numpy as np
from datetime import datetime, timedelta
import math


def calc_solarDA(jd, hour):
    ###
    # Derive solar declination angle
    # Parmeters:
    # jd = julian day of calender (1-366)
    # hour = 1:24 UTC
    ###
    # Calculate angular fraction of the year in radians
    g = (360 / 365.25) * (jd + (hour / 24))  # fractional year g in degrees
    g = g - 360 if g > 360 else g
    g_rad = g * (np.pi / 180)  # convert to radians

    # Calculate the solar declination angle, lowercase delta, in degrees:
    d = 0.396372 - 22.91327 * np.cos(g_rad) + 4.025430 * np.sin(g_rad) - 0.387205 * \
        np.cos(2 * g_rad) + 0.051967 * np.sin(2 * g_rad) - 0.154527 * \
        np.cos(3 * g_rad) + 0.084798 * np.sin(3 * g_rad)

    tc = (0.004297 + 0.107029 * np.cos(g_rad) - 1.837877 * np.sin(g_rad) -
          0.837378 * np.cos(2 * g_rad) - 2.340475 * np.sin(2 * g_rad))

    outputs = {"d": d, "tc": tc}

    return outputs


def calc_cza(lat, lon, y, mon, d, hr):
    ###
    # Derive the cosine of the solar zenith angle
    # Parmeters:
    #    lat = Degrees north latitude (-90 to 90)
    #    lon = Degrees east longitude (-180 to 180)
    #    y = Year (four digits, e.g., 2020)
    #    mon = Month (1-12)
    #    d = Day of month (whole number)
    #    hr = Hour (1-24 UTC)
    # Returns cosine of the solar zenith angle (cza)
    ###
    # Convert input date and time to datetime object
    input_date = datetime(int(y), int(mon), int(d)) + timedelta(hours=hr)

    # Calculate Julian Day
    jd = input_date.timetuple().tm_yday

    hr = 24 + hr if hr < 0 else hr
    # Calculate declination angle and time correction for solar angle
    d_tc = calc_solarDA(jd, hr)
    d = d_tc['d']
    tc = d_tc['tc']

    d_rad = d * (np.pi / 180)
    lat_rad = lat * (np.pi / 180)

    sindec_sinlat = np.sin(d_rad) * np.sin(lat_rad)
    cosdec_coslat = np.cos(d_rad) * np.cos(lat_rad)

    # Solar hour angle [h.deg]
    sha_rad = ((hr - 12) * 15 + lon + tc) * (np.pi / 180)
    csza = sindec_sinlat + cosdec_coslat * np.cos(sha_rad)

    csza = max(0, csza)
    return csza


def daynum(year, month, day):
    ###
    # Calculate day number of year, e.g. Jan. 1 = 01/01 = 001 Dec. 31 = 12/31 = 365 or 366.
    # Parmeters:
    #    year = 4-digit year
    #    month = Number of month (1-12)
    #    day = Day of month
    # Returns the day of year, i.e., "y-day" (1-366)
    ###
    if year < 1:
        return -1

    # Leap years (LY) are divisible by 4, except for centurial years not divisible by 400.
    # Examples of LY: 1996, 2000, 2004; 1896, *NOT* 1900, 1904
    # 1900 is NOT a leap year because it is a centurial year that is not divisible by 400, unlike 2000.
    leapyr = 0  # default is non-leap year
    if ((year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)):
        leapyr = 1

    begmonth = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

    dnum = begmonth[month - 1] + day
    if leapyr == 1 and month > 2:
        dnum += 1

    return dnum


# def solarposition(year, month, day, days_1900, latitude, longitude):
def solarposition(year, month, day, latitude, longitude):
    ###
    # Calculates the Sun's apparent right ascension, apparent declination, altitude, atmospheric refraction correction applicable to the altitude, azimuth,and distance from Earth using the Astronomical Almanac of 1990, which is applicable for years 1950 - 2050, with an accuracy of 0.1 arc minutes for refraction at altitudes of at least 15 degrees.
    # Parmeters:
    #    year Four-digit year (Gregorian calendar)
    #    month Month number (1-12)
    #    day Fractional day of month (0-32)
    #    days_1900 Days since 1 January 1900 at 00:00:00 UTC (will be zero at default since year-month-day is given
    #    latitude Degrees north latitude (-90 to 90)
    #    longitude Degrees east longitude (-180 to 180)
    # Returns a list containing apparent solar right ascension ("ap_ra"), apparent solar declination ("ap_dec"), solar #altitude ("altitude"), refraction correction ("refraction"), solar azimuth ("azimuth"), and distance of Sun from #Earth ("distance")
    # ** No need option to give days by days_1900 format.
    ###
    # Constants
    temp = 15  # Earth mean atmospheric temperature at sea level, deg C
    pressure = 1013.25  # Earth mean atmospheric pressure at sea level, hPa (equivalent to mb)
    PI = 3.1415926535897932
    TWOPI = 6.2831853071795864
    DEG_RAD = 0.017453292519943295
    RAD_DEG = 57.295779513082323

    # Default return value
    retVal = -1

    # Check latitude and longitude for proper range before calculating dates
    if latitude < -90 or latitude > 90 or longitude < -180 or longitude > 180:
        return retVal  # Default retVal is -1, indicating an error

    # If year is not zero then assume date is specified by year, month, day.
    # If year is zero then assume date is specified by days_1900.

    if year != 0:
        # Assume date given by {year, month, day} or {year, 0, daynumber}

        if year < 1950 or year > 2049:
            return retVal  # Year is out of bounds; retVal == -1

        if month != 0:
            if month < 1 or month > 12 or day < 0.0 or day > 33.0:
                return retVal  # retVal == -1
            daynumber = daynum(year, month, math.floor(day))
        else:
            if day < 0.0 or day > 368.0:
                return retVal  # retVal is still -1
            daynumber = math.floor(day)

        # Construct Julian centuries since J2000 at 0 hours UT of date,
        # days.fraction since J2000, and UT hours.

        delta_years = year - 2000

        # delta_days is days from 2000/01/00 (1900's are negative).
        delta_days = math.floor(delta_years * 365 + delta_years / 4 + daynumber)
        if year > 2000:
            delta_days += 1  # J2000 is 2000/01/01.5

        days_J2000 = delta_days - 1.5
        cent_J2000 = days_J2000 / 36525.0

        integral = math.floor(day)
        ut = day - integral

        days_J2000 += ut
        ut *= 24.0

    #   print(days_J2000)
    # else:  # Date given by days_1900, i.e., number of days since 1900
    #    # e.g., days_1900 is 18262 for 1950/01/00, and 54788 for 2049/12/32.
    #    # A.A. 1990, K2-K4.
    #    if days_1900 < 18262.0 or days_1900 > 54788.0:
    #        return retVal  # retVal is still -1, i.e., error

    # Construct days.fraction since J2000, UT hours, and
    # Julian centuries since J2000 at 0 hours UT of date.
    # days_1900 is 36524 for 2000/01/00. J2000 is 2000/01/01.5
    # days_J2000 = days_1900 - 36525.5

    # integral = math.floor(days_1900)
    # ut = (days_1900 - integral) * 24

    # cent_J2000 = (integral - 36525.5) / 36525.0

    # Compute solar position parameters.
    # A.A. (1990, C24)

    mean_anomaly = (357.528 + 0.9856003 * days_J2000)
    mean_longitude = (280.460 + 0.9856474 * days_J2000)

    # Put mean_anomaly and mean_longitude in the range 0 -> 2 pi (from degrees to radians)
    integral = math.floor(mean_anomaly / 360.0)
    mean_anomaly = (mean_anomaly / 360.0 - integral) * TWOPI
    integral = math.floor(mean_longitude / 360.0)
    mean_longitude = (mean_longitude / 360.0 - integral) * TWOPI

    mean_obliquity = (23.439 - 4.0e-7 * days_J2000) * DEG_RAD  # convert to radians
    ecliptic_long = ((1.915 * math.sin(mean_anomaly)) +
                     (0.020 * math.sin(2.0 * mean_anomaly))) * DEG_RAD + mean_longitude

    distance = 1.00014 - 0.01671 * math.cos(mean_anomaly) - 0.00014 * math.cos(2.0 * mean_anomaly)

    # Tangent of ecliptic_long separated into sine and cosine parts for ap_ra.
    ap_ra = math.atan2(math.cos(mean_obliquity) * math.sin(ecliptic_long), math.cos(ecliptic_long))

    # Change range of ap_ra from -pi -> pi to 0 -> 2 pi
    if ap_ra < 0.0:
        ap_ra += TWOPI

    # Put ap_ra in the range 0 -> 24 hours.
    integral = math.floor(ap_ra / TWOPI)
    ap_ra = (ap_ra / TWOPI - integral) * 24.0

    ap_dec = math.asin(math.sin(mean_obliquity) * math.sin(ecliptic_long))

    # Calculate local mean sidereal time.
    # A.A. 199

    # Horner's method of polynomial exponent expansion used for gmst0h.
    gmst0h = 24110.54841 + cent_J2000 * (8640184.812866 + cent_J2000 * (0.093104 - cent_J2000 * 6.2e-6))

    # Convert gmst0h from seconds to hours and put in the range 0 -> 24.
    # gmst0h = modf(gmst0h / 3600.0 / 24.0, &integral) * 24.0
    integral = math.floor(gmst0h / 3600 / 24)
    if integral < 0:
        integral += 1  # to match behavior of C mod
    gmst0h = (gmst0h / 3600 / 24 - integral) * 24

    if gmst0h < 0.0:
        gmst0h += 24.0

    # Ratio of lengths of mean solar day to mean sidereal day is 1.00273790934
    # in 1990. Change in sidereal day length is < 0.001 second over a century.
    # A. A. 1990, B6.

    lmst = gmst0h + (ut * 1.00273790934) + longitude / 15.0

    # Put lmst in the range 0 -> 24 hours.
    integral = math.floor(lmst / 24)
    if integral < 0:
        integral += 1  # to match behavior of C mod
    lmst = (lmst / 24 - integral) * 24

    if lmst < 0.0:
        lmst += 24.0

    # Calculate local hour angle, altitude, azimuth, and refraction correction.
    # A.A. 1990, B61-B62

    local_ha = lmst - ap_ra

    # Put hour angle in the range -12 to 12 hours.
    if local_ha < -12.0:
        local_ha += 24.0
    elif local_ha > 12.0:
        local_ha -= 24.0

    # Convert latitude and local_ha to radians
    latitude *= DEG_RAD
    local_ha = local_ha / 24.0 * TWOPI

    cos_apdec = math.cos(ap_dec)
    sin_apdec = math.sin(ap_dec)
    cos_lat = math.cos(latitude)
    sin_lat = math.sin(latitude)
    cos_lha = math.cos(local_ha)

    altitude = math.asin(sin_apdec * sin_lat + cos_apdec * cos_lha * cos_lat)
    cos_alt = math.cos(altitude)

    # Avoid tangent overflow at altitudes of +-90 degrees.
    # 1.57079615 radians is equal to 89.99999 degrees.

    if abs(altitude) < 1.57079615:
        tan_alt = math.tan(altitude)
    else:
        tan_alt = 6.0e6

    cos_az = (sin_apdec * cos_lat - cos_apdec * cos_lha * sin_lat) / cos_alt
    sin_az = -(cos_apdec * math.sin(local_ha) / cos_alt)
    azimuth = math.acos(cos_az)

    # Change range of azimuth from 0 -> pi to 0 -> 2 pi
    if math.atan2(sin_az, cos_az) < 0.0:
        azimuth = TWOPI - azimuth

    # Convert ap_dec, altitude, and azimuth to degrees
    ap_dec *= RAD_DEG
    altitude *= RAD_DEG
    azimuth *= RAD_DEG

    # Compute refraction correction to be added to altitude to obtain actual position.
    # * Refraction calculated for altitudes of -1 degree or more allows for a
    # * pressure of 1040 mb and temperature of -22 C. Lower pressure and higher
    # * temperature combinations yield less than 1 degree refraction.
    # * NOTE:
    # * The two equations listed in the A.A. have a crossover altitude of
    # * 19.225 degrees at standard temperature and pressure. This crossover point
    # * is used instead of 15 degrees altitude so that refraction is smooth over
    # * the entire range of altitudes. The maximum residual error introduced by
    # * this smoothing is 3.6 arc seconds at 15 degrees. Temperature or pressure
    # * other than standard will shift the crossover altitude and change the error.

    if altitude < -1.0 or tan_alt == 6.0e6:
        refraction = 0.0
    else:
        if altitude < 19.225:
            refraction = (0.1594 + (altitude) * (0.0196 + 0.00002 * (altitude))) * pressure
            refraction = refraction / (1.0 + (altitude) * (0.505 + 0.0845 * (altitude))) * (273.0 + 15.0)
        else:
            refraction = 0.00452 * (pressure / (273.0 + 15.0)) / tan_alt

    # To match Michalsky's sunae program, the following line was inserted
    # by JC Liljegren to add the refraction correction to the solar altitude

    altitude += refraction

    # If we made it here, then everything worked.
    retVal = 0

    outputs = {"retVal": retVal, "distance": distance, "azimuth": azimuth, "refraction": refraction,
               "altitude": altitude, "ap_dec": ap_dec, "ap_ra": ap_ra}
    return outputs


# def calc_fdir(year, month, day, lat, lon, solar, cza):
def calc_fdir(year, month, dday, lat, lon, solar, cza):
    # To Calculate the fraction of the solar irradiance due to the direct beam
    # Parmeters:
    # year Year (4 digits)
    # month Month (1-12)
    # dday Day-fraction of month based on UTC time. Day number must include fractional day based on time, e.g., 4.5 = noon UTC on the 4th of the month.
    # lat Degrees north latitude (-90 to 90)
    # lon Degrees east longitude (-180 to 180)
    # solar Total surface solar irradiance (W/m2)
    # cza Cosine solar zenith angle (0-1)
    # Returns the fraction of irradiance due to direct beam ("fdir").

    # DEFAULTS
    # days_1900 = 0.0
    solarRet = solar
    fdir = 0

    # CONSTANTS
    SOLAR_CONST = 1367.0
    DEG_RAD = 0.017453292519943295
    CZA_MIN = 0.00873
    NORMSOLAR_MAX = 0.85

    # Call solarposition function
    # solarposObj = solarposition(year, month, day, days_1900, lat, lon)
    solarposObj = solarposition(year, month, dday, lat, lon)
    elev = solarposObj['altitude']
    soldist = solarposObj['distance']

    toasolar = SOLAR_CONST * max(0, cza) / (soldist * soldist)

    # If the sun is not fully above the horizon, then
    # set the maximum (top of atmosphere [TOA]) solar <- 0
    if cza < CZA_MIN:
        toasolar = 0

    if toasolar > 0:
        # Account for any solar sensor calibration errors and
        # make the solar irradiance consistent with normsolar
        normsolar = min(solar / toasolar, NORMSOLAR_MAX)
        solarRet = normsolar * toasolar

        # calculate fraction of the solar irradiance due to the direct beam
        if normsolar > 0:
            fdir = math.exp(3 - 1.34 * normsolar - 1.65 / normsolar)
            fdir = max(min(fdir, 0.9), 0.0)
        else:
            fdir = 0
            cza = 0  # added "cza = 0"
    else:
        fdir = 0
        cza = 0  # added "cza = 0"

    return fdir


###################  TGLOBE  #####################

def esat(tk, phase, Pair):
    ###
    # Calculates the saturation vapor pressure (mb) over liquid water (phase = 0) or ice (phase = 1).
    # Parameters:
    #        tk: Air temperature in Kelvin (K).
    #        phase: Over liquid water (0) or ice (1).
    #        Pair: Barometric pressure in millibars (equivalent to hPa).
    # Returns the saturation vapor pressure in millibars (equivalent to hPa).
    ###
    if phase == 0:
        y = (tk - 273.15) / (tk - 32.18)
        es = 6.1121 * math.exp(17.502 * y)
        # Apply "enhancement factor" to correct estimate for moist air:
        es *= (1.0007 + (3.46e-6 * Pair))
    else:  # over ice
        y = (tk - 273.15) / (tk - 0.6)
        es = 6.1115 * math.exp(22.452 * y)
        es *= (1.0003 + (4.18e-6 * Pair))

    return es


def emis_atm(Tair, rh, pres):
    ###
    # Calculates the atmospheric emissivity, a necessary input to the calculation of globe temperature.
    # Parameters:
    #    Tair: Air temperature in Kelvin (K)
    #    rh: Relative humidity as a proportion between 0 and 1
    #    pres: Barometric pressure in millibars (equivalent to hPa)
    # Returns the atmospheric emissivity
    ###

    eee = rh * esat(Tair, 0, pres)
    return 0.575 * (eee ** 0.143)


def viscosity(Tair):
    ###
    # Calculates the viscosity of air in units of kg/(m⋅s).
    #    This is an input into the calculation of thermal conductivity, wet-bulb temperature,
    #    and the convective heat transfer coefficient.
    ###
    # CONSTANTS
    M_AIR = 28.97
    sigma = 3.617
    eps_kappa = 97.0

    Tr = Tair / eps_kappa
    omega = (Tr - 2.9) / 0.4 * (-0.034) + 1.048
    return 2.6693e-6 * (M_AIR * Tair) ** 0.5 / (sigma ** 2 * omega)


def thermal_cond(Tair):
    ###
    # Calculates the thermal conductivity of air in units of W/(m⋅K).This value is used as an input to the calculation of the convective heat transfer coefficient.
    # Parameters:
    #    Tair Air temperature (K)
    ###
    # CONSTANTS
    Cp = 1003.5
    R_GAS = 8314.34
    M_AIR = 28.97
    R_AIR = R_GAS / M_AIR

    return (Cp + 1.25 * R_AIR) * viscosity(Tair)


def h_sphere_in_air(diameter, Tair, Pair, speed):
    ####
    # Calculate the convective heat transfer coefficient in units of W/(m2⋅K) for flow around a sphere.
    #    diameter Sphere diameter (m)
    #    Tair Air temperature (K)
    #    Pair Barometric pressure in millibars (equivalent to hPa)
    #    speed Wind speed (m/s)
    # Returns the convective heat transfer coefficient in units of W/(m2⋅K)
    ###
    # CONSTANTS
    R_GAS = 8314.34
    M_AIR = 28.97
    R_AIR = R_GAS / M_AIR
    MIN_SPEED = 0.5  # originally was 0.13 m/s
    Cp = 1003.5
    Pr = Cp / (Cp + 1.25 * R_AIR)

    density = Pair * 100 / (R_AIR * Tair)

    # Calculate Reynolds Number (Re)
    Re = max(speed, MIN_SPEED) * density * diameter / viscosity(Tair)

    # Calculate Nusselt Number (Nu)
    Nu = 2.0 + 0.6 * (Re ** 0.5) * (Pr ** 0.3333)

    return Nu * thermal_cond(Tair) / diameter


def Tglobe(Tair, rh, Pair, speed, solar, fdir, cza):
    ###
    #   Calculates the globe temperature as an input to the wet-bulb globe temperature (WBGT).
    #   Tair Dry-bulb air temperature (Kelvin)
    #   Relative humidity as proportion (0-1)
    #   Pair Barometric pressure in millibars (equivalent to hPa)
    #   speed Wind speed (m/s)
    #   solar Solar irradiance (W/m2)
    #   fdir Fraction of solar irradiance due to direct beam (0-1)
    #   cza Cosine of solar zenith angle (0-1)
    # Returns the globe temperature in degrees Celsius
    ###
    # The equation for Tglobe_new has cza in the denominator, so it will result in
    # NaN for cza = 0. This should only be 0 at nighttime, in which case both fdir
    # and cza should both be zero. When fdir and cza are both 0, the value of cza
    # has no bearing on Tglobe, even when solar > 0. To avoid an
    # unnecessary NA value, replace cza with 0.01 when cza < 0.01

    if cza < 0.01: cza = 0.01

    # CONSTANTS
    EMIS_GLOBE = 0.95
    ALB_GLOBE = 0.05
    ALB_SFC = 0.45
    D_GLOBE = 0.0508
    EMIS_SFC = 0.999
    STEFANB = 5.6696e-8
    CONVERGENCE = 0.02
    MAX_ITER = 100  # Increased from 50; adds to processing time but reduces missingness

    # VARIABLES
    Tsfc = Tair
    Tglobe_prev = Tair  # first guess is the air temperature
    iter = 0

    while iter < MAX_ITER:
        iter += 1
        Tref = 0.5 * (Tglobe_prev + Tair)  # evaluate properties at the average temperature
        h = h_sphere_in_air(D_GLOBE, Tref, Pair, speed)

        Tglobe_new = ((0.5 * (emis_atm(Tair, rh, Pair) * (Tair ** 4) + EMIS_SFC * (Tsfc ** 4)) -
                       h / (STEFANB * EMIS_GLOBE) * (Tglobe_prev - Tair) +
                       solar / (2 * STEFANB * EMIS_GLOBE) * (1 - ALB_GLOBE) *
                       (fdir * (1 / (2 * cza) - 1) + 1 + ALB_SFC)) ** 0.25)

        if abs(Tglobe_new - Tglobe_prev) < CONVERGENCE:
            break

        Tglobe_prev = 0.9 * Tglobe_prev + 0.1 * Tglobe_new

    if iter < MAX_ITER:
        return Tglobe_new - 273.15
    else:
        return None


###################  T NATURAL WET BULB #####################

from math import log


def dew_point(e, phase, Pair):
    ###
    # Calculates the dew point or frost point temperaturein units of Kelvin (K) from barometric pressure and vapor pressure.
    # Parameters:
    #    e: Vapor pressure in millibars
    #    phase: Indicator - 0 for dew point or 1 for frost point
    #    Pair: Barometric pressure in millibars (equivalent to hPa)
    # Returns the dew-point temperature in units of Kelvin (K)

    if phase == 0:  # Dew point
        # Calculate same enhancement factor as in function for saturation vapor pressure
        EF = 1.0007 + (3.46e-6 * Pair)
        z = log(e / (6.1121 * EF))
        tdk = 273.15 + 240.97 * z / (17.502 - z)

    else:  # Frost point
        EF = 1.0003 + (4.18e-6 * Pair)
        z = math.log(e / (6.1115 * EF))
        tdk = 273.15 + 272.55 * z / (22.452 - z)

    return tdk


def h_cylinder_in_air(diameter, length, Tair, Pair, speed):
    ###
    # Calculates the convective heat transfer coefficient in units of W/(m2⋅K) for a long cylinder in cross flow.
    # Parameters:
    #    diameter Cylinder diameter (m)
    #    length Cylinder length (m)
    #    Tair Air temperature (K)
    #    Pair Barometric pressure in millibars (equivalent to hPa)
    #    speed Wind speed (m/s)
    # Returns the convective heat transfer coefficient in units of W/(m2⋅K)
    ###
    # CONSTANTS
    a = 0.56
    b = 0.281
    c = 0.4
    R_GAS = 8314.34
    M_AIR = 28.97
    R_AIR = R_GAS / M_AIR
    MIN_SPEED = 0.5  # Originally was 0.13 m/s
    Cp = 1003.5
    Pr = Cp / (Cp + 1.25 * R_AIR)

    density = Pair * 100 / (R_AIR * Tair)
    Re = max(speed, MIN_SPEED) * density * diameter / viscosity(Tair)
    Nu = b * (Re ** (1 - c)) * (Pr ** (1 - a))

    return Nu * thermal_cond(Tair) / diameter


def diffusivity(Tair, Pair):
    ###
    # Calculate the diffusivity of water vapor in air, m2/s.
    # Parameters:
    #   Tair= Air temperature in Kelvin (K)
    #   Pair= Barometric pressure in millibars (equivalent to hPa)
    # Returns the diffusivity in units of m2/s
    ###
    # CONSTANTS
    M_AIR = 28.97
    M_H2O = 18.015
    Pcrit_air = 36.4
    Pcrit_h2o = 218.0
    Tcrit_air = 132.0
    Tcrit_h2o = 647.3
    a = 3.640e-4
    b = 2.334

    Pcrit13 = (Pcrit_air * Pcrit_h2o) ** (1 / 3)
    Tcrit512 = (Tcrit_air * Tcrit_h2o) ** (5 / 12)
    Tcrit12 = (Tcrit_air * Tcrit_h2o) ** 0.5
    Mmix = (1 / M_AIR + 1 / M_H2O) ** 0.5
    Patm = Pair / 1013.25  # Convert pressure from mb (or hPa) to atmospheres (atm)

    return a * ((Tair / Tcrit12) ** b) * Pcrit13 * Tcrit512 * Mmix / Patm * 1e-4


def evap(Tair):
    ###
    # Calculates the heat of evaporation in units of J/kg.
    # Parameters:
    #    Tair= Air temperature in Kelvin (K)
    # Returns the heat of evaporation in J/kg.
    # Reference for algorithm for calculating heat of vaporization: Meyra et al. (2004), https://doi.org/10.1016/j.fluid.2003.12.011
    ###
    # CONSTANTS
    Zc = 0.292  # Universal critical ratio
    Tc = 647.3  # Critical temperature of H2O (K)
    Tt = 273.16  # Triple temperature of H2O (K)
    dH_tp = 2500900  # enthalpy of vaporization of H2O at its triple point (J/kg)

    H = dH_tp * ((Tc - Tair) / (Tc - Tt)) ** ((Zc * Zc) * ((Tair - Tt) / (Tc - Tt)) + Zc)
    return H


def Twb(Tair, rh, Pair, speed, solar, fdir, cza):
    ###
    # Calculates the natural wet-bulb temperature.
    # Parameters:
    #   Tair Air temperature (dry bulb) in Kelvin (K)
    #   rh Relative humidity as a proportion (0-1)
    #  Pair Barometric pressure in millibars	(equivalent to hPa)
    #   speed Wind speed (m/s)
    #   solar Solar irradiance (W/m2)
    #   fdir Fraction of solar irradiance due to direct beam
    #   cza Cosine of solar zenith angle
    ###
    rad = 1  # indicator for wet-bulb temperature; 0 for psychrometric wet-bulb temperature

    # CONSTANTS
    CONVERGENCE = 0.02
    MAX_ITER = 100
    D_WICK = 0.007
    L_WICK = 0.0254
    PI = math.pi
    Cp = 1003.5
    R_GAS = 8314.34
    M_AIR = 28.97
    M_H2O = 18.015
    R_AIR = R_GAS / M_AIR
    Pr = Cp / (Cp + 1.25 * R_AIR)
    STEFANB = 5.6696e-8
    EMIS_SFC = 0.999
    EMIS_WICK = 0.95
    ALB_WICK = 0.4
    ALB_SFC = 0.45
    RATIO = Cp * M_AIR / M_H2O
    a = 0.56  # from Bedingfield and Drew

    # VARIABLES
    Tsfc = Tair
    sza = math.acos(cza)  # solar zenith angle, radians
    eair = rh * esat(Tair, 0, Pair)
    Tdew = dew_point(eair, 0, Pair)  # needs Pair to calculate the enhancement factor
    Twb_prev = Tdew  # first guess is the dew-point temperature
    converged = False
    iter = 0

    while True:
        iter += 1
        Tref = 0.5 * (Twb_prev + Tair)  # evaluate properties at the average temperature

        # Calculate convective heat transfer coefficient (h)
        h = h_cylinder_in_air(D_WICK, L_WICK, Tref, Pair, speed)

        # Calculate radiative heating term
        Fatm = STEFANB * EMIS_WICK * (0.5 * (emis_atm(Tair, rh, Pair) * (Tair ** 4) + EMIS_SFC * (Tsfc ** 4)) -
                                      (Twb_prev ** 4)) + (1 - ALB_WICK) * solar * (
                       (1 - fdir) * (1 + 0.25 * D_WICK / L_WICK) +
                       fdir * ((math.tan(sza) / PI) + 0.25 * D_WICK / L_WICK) + ALB_SFC)

        ewick = esat(Twb_prev, 0, Pair)
        density = Pair * 100 / (R_AIR * Tref)

        # Calculate Schmidt number (Sc)
        Sc = viscosity(Tref) / (density * diffusivity(Tref, Pair))

        Twb_new = Tair - evap(Tref) / RATIO * (ewick - eair) / (Pair - ewick) * ((Pr / Sc) ** a) + (Fatm / h * rad)
        if abs(Twb_new - Twb_prev) < CONVERGENCE:
            converged = True

        Twb_prev = 0.9 * Twb_prev + 0.1 * Twb_new

        if converged or iter == MAX_ITER:
            break

    if converged:
        return Twb_new - 273.15
    else:
        return None


###################  WET BULB GLOBE TEMPERATURE #####################

def calc_solar_parameters(year, month, day, lat, lon, solar, cza=None, fdir=None):
    ###
    # To calculate the adjusted surface solar irradiance, cosine of the solar zenith angle, and fraction of the solar irradiance due to the direct beam.
    #   year Year (4 digits)
    #   month Month (1-12)
    #   day Day-fraction of month based on UTC time. Day number must include fractional day based on time, e.g., 4.5 = noon UTC on the 4th of the month.
    #  lat Degrees north latitude (-90 to 90)
    #   lon Degrees east longitude (-180 to 180)
    #   solar Total surface solar irradiance (W/m2)
    # Returns adjusted solar radiation ("solarRet"), cosine of the solar zenith angle ("cza", unchanged if user-supplied), #and the fraction of irradiance due to direct beam ("fdir", unchanged if user-supplied).
    ##

    # DEFAULTS
    # days_1900 = 0.0   ## no need this. just give everything by date.
    solarRet = solar

    # CONSTANTS
    SOLAR_CONST = 1367.0
    DEG_RAD = 0.017453292519943295
    CZA_MIN = 0.00873
    NORMSOLAR_MAX = 0.85

    # Assuming solarposition is defined elsewhere and returns necessary values
    # solarposObj = solarposition(year, month, day, days_1900, lat, lon)
    solarposObj = solarposition(year, month, day, lat, lon)
    elev = solarposObj['altitude']

    if cza is None:
        cza = max(0, cos((90 - elev) * DEG_RAD))

    toasolar = SOLAR_CONST * max(0, cza) / (solarposObj['distance'] ** 2)

    cza = max(0, cza)

    if cza < CZA_MIN:
        toasolar = 0

    if toasolar > 0:
        normsolar = min(solar / toasolar, NORMSOLAR_MAX)

        solarRet = normsolar * toasolar

        if normsolar > 0:
            if fdir is None:
                fdir = exp(3 - 1.34 * normsolar - 1.65 / normsolar)
            fdir = max(min(fdir, 0.9), 0.0)
        else:
            fdir = 0
            cza = 0

    else:
        fdir = 0
        cza = 0

    return {"solarRet": solarRet, "cza": cza, "fdir": fdir}


def stab_srdt(daytime, speed, solar, dT):
    # Create a matrix to store stability classes
    lsrdt = [
        [1, 1, 2, 4, 0, 5, 6, 0],
        [1, 2, 3, 4, 0, 4, 5, 0],
        [2, 2, 3, 4, 0, 4, 4, 0],
        [3, 3, 4, 4, 0, 0, 0, 0],
        [3, 4, 4, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]

    if daytime == 1:
        if solar >= 925.0:
            j = 1
        elif solar >= 675.0:
            j = 2
        elif solar >= 175.0:
            j = 3
        else:
            j = 4

        if speed >= 6.0:
            i = 5
        elif speed >= 5.0:
            i = 4
        elif speed >= 3.0:
            i = 3
        elif speed >= 2.0:
            i = 2
        else:
            i = 1
    else:  # NOT daytime
        if dT >= 0.0:
            j = 7
        else:
            j = 6

        if speed >= 2.5:
            i = 3
        elif speed >= 2.0:
            i = 2
        else:
            i = 1

    return lsrdt[i - 1][j - 1]


def est_wind_speed(speed, zspeed, stability_class, urban):
    # Define constants:
    MIN_SPEED = 0.5  # 0.13 m/s in the original code
    REF_HEIGHT = 2.0

    urban_exp = [0.15, 0.15, 0.20, 0.25, 0.30, 0.30]
    rural_exp = [0.07, 0.07, 0.10, 0.15, 0.35, 0.55]

    if urban == 1:
        exponent = urban_exp[stability_class - 1]
    else:
        exponent = rural_exp[stability_class - 1]

    est_speed = speed * ((REF_HEIGHT / zspeed) ** exponent)
    est_speed = max(est_speed, MIN_SPEED)
    return est_speed


def wbgt(year, month, day, hr, lat, lon, solar, cza, fdir, pres, Tair,
         relhum, speed):
    ###
    # Calculates the outdoor wet bulb-globe temperature (WBGT) according to:
    #    WBGT = (0.1 ⋅ Ta) + (0.7 ⋅ Tw) + (0.2 ⋅ Tg)

    # The program predicts Tw and Tg using meteorological input data, and then combines the results to produce WBGT.
    # Reference: Liljegren, et al. Modeling the Wet Bulb Globe Temperature Using
    # Standard Meteorological Measurements. J. Occup. Environ. Hyg. 5, 645-655 (2008).
    # https://doi.org/10.1080/15459620802310770

    #    year 4-digit integer, e.g., 2007
    #    month Month (1-12) or month = 0 if reporting day as day of year
    #    **day Day of month in UTC (changed from dday)
    #    **hr Hour of day (added to derive dday and for cza)
    #    lat Degrees north latitude (-90 to 90)
    #    lon Degrees east longitude (-180 to 180)
    #    solar Solar irradiance (W/m2)
    #    cza Cosine solar zenith angle (0-1); **using calc_solar_parameters()
    #    fdir Fraction of surface solar radiation that is direct (0-1)
    #    pres Barometric pressure in millibars (equivalent to hPa)
    #    Tair Dry-bulb air temperature (deg. C)
    #    relhum Relative humidity (%)
    #    speed Wind speed (m/s)
    #    ** removed zpeed, dT, urban
    # Returns the wet-bulb globe temperature in degrees C.
    ###
    # Get the dday for calcsolarparameters
    dday = day + (hr / 23)

    inputs = [year, month, dday, lat, lon, solar, cza, fdir, pres, Tair, relhum, speed]

    # Check for missing data and return NA
    if any(x is None or x == -999 for x in inputs):
        return None

    # cza and fdir are assumed to be known. If they are not, set them to NA here
    # and the calc_solar_parameters() function will calculate approximations of them
    solar = calc_solar_parameters(year, month, dday, lat, lon, solar, cza, fdir)[
        "solarRet"]  # adjusted solar irradiance if out of bounds

    # estimate the 2-meter wind speed, if necessary
    REF_HEIGHT = 2.0  # 2-meter reference height
    MINIMUM_SPEED = 0.5

    dT = -0.084  # dT=vertical temperature diff between windspeed heights). Assume a vertical lapse rate of -6.5°C/km

    zspeed = 10  # height of wind speed 'measurement', in WRF use 10m wind speed.
    urban = 1  # set all to urban in sg

    if zspeed != REF_HEIGHT:
        if cza > 0:
            daytime = True
        else:
            daytime = False

        stability_class = stab_srdt(daytime, speed, solar, dT)
        speed = est_wind_speed(speed, zspeed, stability_class, urban)
    else:
        speed = max(speed, MINIMUM_SPEED)

    # Unit Conversions
    tk = Tair + 273.15  # deg. C to kelvin
    rh = relhum / 100.0  # relative humidity % to fraction

    # Calculate the globe (Tg), natural wet bulb (Tnwb), psychrometric wet bulb (Tpsy), and
    # outdoor wet bulb globe temperatures (Twbg)
    Tg = Tglobe(tk, rh, pres, speed, solar, fdir, cza)
    Tnwb = Twb(tk, rh, pres, speed, solar, fdir, cza)
    # print(Tnwb)
    Twbg = (0.1 * Tair) + (0.2 * Tg) + (0.7 * Tnwb)

    return Twbg

# wbgt(year, mon, day, lat, lon, solar, cza, fdir, pres, Tair, relhum, speed)
# wbgt(year=2020, month=7, day=4.5, hr= 1,lat=1, lon=103, solar=0, cza=0.5, fdir=0.5, pres=1013, Tair=28, relhum=50, speed=5)


