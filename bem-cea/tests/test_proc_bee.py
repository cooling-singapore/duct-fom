import sys
sys.path.append('../proc_bee')

import os
import tempfile

from proc_bee.processor import (determine_ah_sample_dates, make_ah_emissions, make_annual_energy_demand,
                                make_pv_potential)


BEMCEA_BEE_PATH = os.path.join(os.environ['TEST_DATA_PATH'], 'bemcea_bee')


def test_determine_ah_sample_dates():
    weather_file_path = os.path.join(BEMCEA_BEE_PATH, 'duct-cea', 'inputs', 'weather', 'weather.epw')
    ah_sample_dates = determine_ah_sample_dates(weather_file_path)
    print(ah_sample_dates)
    assert ah_sample_dates == ['01-02-2030', '01-04-2030', '01-06-2030', '01-08-2030', '01-10-2030', '01-12-2030']


def test_make_ah_emissions():
    duct_cea_path = os.path.join(BEMCEA_BEE_PATH, '6fvbEVCy', 'job', 'duct-project', 'duct-cea')
    ah_emissions_path = os.path.join(duct_cea_path, 'outputs', 'data', 'emissions', 'AH')

    with tempfile.TemporaryDirectory() as tempdir:
        output_path = os.path.join(tempdir, 'ah_emissions.csv')
        make_ah_emissions(ah_emissions_path, output_path)
        assert os.path.exists(output_path)


def test_make_building_demand():
    duct_cea_path = os.path.join(BEMCEA_BEE_PATH, '6fvbEVCy', 'job', 'duct-project', 'duct-cea')
    building_demand_path = os.path.join(duct_cea_path, 'outputs', 'data', 'demand')
    building_schedules_path = os.path.join(duct_cea_path, 'inputs', 'building-properties', 'schedules')

    with tempfile.TemporaryDirectory() as tempdir:
        output_path = os.path.join(tempdir, 'building_demand.csv')
        make_annual_energy_demand(building_demand_path, building_schedules_path, output_path)
        assert os.path.exists(output_path)


def test_make_pv_potential():
    duct_cea_path = os.path.join(BEMCEA_BEE_PATH, '6fvbEVCy', 'job', 'duct-project', 'duct-cea')
    building_demand_path = os.path.join(duct_cea_path, 'outputs', 'data', 'demand')
    building_schedules_path = os.path.join(duct_cea_path, 'inputs', 'building-properties', 'schedules')
    solar_path = os.path.join(duct_cea_path, 'outputs', 'data', 'potentials', 'solar')

    with tempfile.TemporaryDirectory() as tempdir:
        output_path = os.path.join(tempdir, 'building_demand.csv')
        annual_energy_demand = make_annual_energy_demand(building_demand_path, building_schedules_path, output_path)

        output_path = os.path.join(tempdir, 'pv_potential.csv')
        make_pv_potential(solar_path, output_path, annual_energy_demand)
        assert os.path.exists(output_path)
