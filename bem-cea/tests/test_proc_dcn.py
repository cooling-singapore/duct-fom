import sys
sys.path.append('../proc_dcn')

from proc_dcn.processor import read_supply_system, make_supply_systems_output, make_ah_emissions

import os
import tempfile


DCNCEA_SIM_PATH = os.path.join(os.environ['TEST_DATA_PATH'], 'dcncea_sim')


def test_read_supply_system():
    supply_system_path = os.path.join(DCNCEA_SIM_PATH, 'MmRNDGqg', 'duct-project', 'duct-cea',
                                      'outputs', 'data', 'optimization', 'centralized', 'current_DES')
    result = read_supply_system(supply_system_path)
    assert '292420487' in result
    assert '292420567' in result
    assert '373711013' in result
    assert '373711014' in result
    print(result)


def test_make_supply_systems_output():
    working_dir = os.path.join(DCNCEA_SIM_PATH, 'MmRNDGqg')
    with tempfile.TemporaryDirectory() as tempdir:
        output_path = os.path.join(tempdir, 'supply_systems.json')
        make_supply_systems_output(working_dir, output_path)
        assert os.path.exists(output_path)


def test_make_ah_emissions():
    ah_emissions_path = os.path.join(DCNCEA_SIM_PATH, 'MmRNDGqg', 'duct-project', 'duct-cea',
                                     'outputs', 'data', 'emissions', 'ah')
    with tempfile.TemporaryDirectory() as tempdir:
        output_path = os.path.join(tempdir, 'ah_emissions.csv')
        make_ah_emissions(ah_emissions_path, output_path)
        assert os.path.exists(output_path)
