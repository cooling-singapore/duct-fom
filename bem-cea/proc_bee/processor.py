import csv
import json
import re
import shlex
import sys
import os.path
import subprocess
import logging
from typing import Union, List, Dict, Optional
from zipfile import ZipFile

import numpy as np
import pandas as pd

from cea.config import Configuration, DEFAULT_CONFIG
from cea.inputlocator import InputLocator


DUCT_CEA_PROJECT_NAME = 'duct-project'
DUCT_CEA_SCENARIO_NAME = 'duct-cea'
PV_TYPE_MAP = {
    "Generic monocrystalline": "PV1",
    "Generic polycrystalline": "PV2",
    "Generic amorphous silicon": "PV3"
}


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


def print_message(severity: str, message: str) -> None:
    print(f"trigger:message:{severity}:{message}")
    sys.stdout.flush()


def get_cea_config(working_dir: str) -> Configuration:
    config = Configuration()
    config.project = os.path.join(working_dir, DUCT_CEA_PROJECT_NAME)
    config.scenario_name = DUCT_CEA_SCENARIO_NAME

    return config


def run_cmd(cmd: Union[str, List[str]], file_output: str = None) -> bytes:
    """
    Run command that writes stdout to file and raise error if return is non-zero
    """
    _cmd = shlex.split(cmd) if type(cmd) == str else cmd
    output = []
    with subprocess.Popen(_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
        for c in iter(lambda: process.stdout.read(1), b''):
            sys.stdout.buffer.write(c)
            output.append(c)
        process.wait()

        _output = b"".join(output)

        if file_output is not None:
            with open(file_output, "w") as f:
                f.write(_output.decode("utf-8"))

        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, process.args, output=process.stdout,
                                                stderr=process.stderr)
    return _output


def cea_workflow(cea_config: Configuration, ah_sampling_dates: List[str], pv_parameters: Optional[Dict]) -> None:
    from cea.datamanagement.terrain_helper import main as terrain_helper
    from cea.datamanagement.archetypes_mapper import main as archetypes_mapper
    from cea.demand.schedule_maker.schedule_maker import main as schedule_maker
    from cea.resources.radiation.main import main as radiation
    from cea.demand.demand_main import main as demand
    from cea.technologies.solar.photovoltaic import main as photovoltaic

    from cea.api import groups_helper
    from cea.api import heat_rejection

    # Set config
    cea_config.demand.overheating_warning = False
    cea_config.demand.ah_sampling_dates = ','.join(ah_sampling_dates)
    if pv_parameters is not None:
        cea_config.solar.panel_on_roof = pv_parameters["roof"]
        cea_config.solar.panel_on_wall = pv_parameters["walls"]
        cea_config.solar.annual_radiation_threshold = pv_parameters["annual_radiation_threshold"]
        cea_config.solar.custom_tilt_angle = pv_parameters["custom_tilt_angle"]
        cea_config.solar.panel_tilt_angle = pv_parameters["tilt_angle"]
        cea_config.solar.max_roof_coverage = pv_parameters["max_roof_coverage"]
        cea_config.solar.type_pvpanel = PV_TYPE_MAP[pv_parameters["type_pv"]]

    # Generate required input files
    print_message("info", "Generating CEA inputs")
    terrain_helper(cea_config)
    archetypes_mapper(cea_config)
    schedule_maker(cea_config)
    print_progress(15)

    # Calculate radiation
    print_message("info", "Calculating solar radiation")
    radiation(cea_config)
    print_progress(30)

    # Calculate demand
    print_message("info", "Calculating building demand")
    demand(cea_config)
    print_progress(45)

    # FIXME: Create empty values if PV parameters do not exist
    # Calculate potentials
    print_message("info", "Calculating photovoltaic potential")
    photovoltaic(cea_config)


def prepare_scenario(working_dir: str, locator: InputLocator) -> None:
    """
    Prepare scenario by extracting input files and moving them to match CEA project folder structure in working directory
    """
    # unpack the CEA run package
    cea_run_package = os.path.join(working_dir, "cea_run_package")
    with ZipFile(cea_run_package, "r") as zip_file:
        zip_file.extractall(locator.get_input_folder())

    # unpack the CEA databases
    cea_databases = os.path.join(working_dir, "cea_databases")
    with ZipFile(cea_databases, "r") as zip_file:
        zip_file.extractall(locator.get_databases_folder())


def determine_ah_sample_dates(weather_file_path: str) -> List[str]:
    with open(weather_file_path, 'r') as f:
        dates: Dict[int, Dict[int, set]] = {}
        for line in f.readlines():
            try:
                line = line.strip().split(',')
                year = int(line[0])
                month = int(line[1])
                day = int(line[2])

                if year not in dates:
                    dates[year] = {}
                if month not in dates[year]:
                    dates[year][month] = set()

                dates[year][month].add(day)
            except ValueError:
                continue
                # determine 6 dates (first day of every even month)
        sample_dates = []
        y = list(dates.keys())[0]
        for m in [2, 4, 6, 8, 10, 12]:
            d = min(dates[y][m])
            sample_dates.append(f"{d:02d}-{m:02d}-{y:04d}")

        return sample_dates


def make_annual_energy_demand(building_demand_path: str, building_schedules_path: str, output_path: str) -> dict:
    # read the total demand data
    output = {}
    total_demand_path = os.path.join(building_demand_path, 'Total_demand.csv')
    with open(total_demand_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            building_name = row['Name']

            # read the building schedule
            schedule = pd.read_csv(os.path.join(building_schedules_path, f'{building_name}.csv'), skiprows=2)
            occupancy = schedule[['DAY', 'OCCUPANCY']].groupby('DAY').sum(numeric_only=True)['OCCUPANCY']
            operating_hours = occupancy['WEEKDAY'] * 5 + occupancy['SATURDAY'] + occupancy['SUNDAY']

            # determine demand, GFA, EUI, EEI
            GRID_MWhyr = float(row['GRID_MWhyr'])
            GFA_m2 = float(row['GFA_m2'])
            EUI_kWhyrm2 = (GRID_MWhyr / GFA_m2) * 1000 if GFA_m2 > 0 else 0.0
            EEI_kWhyrm2 = (EUI_kWhyrm2 * 55 / operating_hours) if operating_hours > 0 else 0.0

            output[building_name] = {
                'GRID_MWhyr': GRID_MWhyr,
                'GFA_m2': GFA_m2,
                'EUI_kWhyrm2': EUI_kWhyrm2,
                'EEI_kWhyrm2': EEI_kWhyrm2,
                'OH_h': operating_hours
            }

    # write output as JSON file
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    return output


def make_pv_potential(solar_path: str, output_path: str, annual_energy_demand: dict) -> dict:
    # read the PV data for each building
    output = {}
    for filename in [f for f in os.listdir(solar_path) if f.endswith('_PV.csv')]:
        # extract the building name
        building_name = filename.split('_')[0]

        # read the data for all columns of interest
        columns = ['E_PV_gen_kWh', 'PV_roofs_top_E_kWh', 'Area_PV_m2', 'PV_roofs_top_m2', 'radiation_kWh']
        data = {c: [] for c in columns}
        with open(os.path.join(solar_path, filename), mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # add data of all columns of interest
                for c in columns:
                    data[c].append(float(row[c]))

        building_generation_total = np.sum(data['E_PV_gen_kWh'])
        building_generation_roof = np.sum(data['PV_roofs_top_E_kWh'])
        building_generation_walls = building_generation_total - building_generation_roof

        total_area = data['Area_PV_m2'][0]
        roof_area = data['PV_roofs_top_m2'][0]
        walls_area = total_area - roof_area

        total_radiation = np.sum(data['radiation_kWh'])

        building_EUI_kWh = annual_energy_demand[building_name]['EUI_kWhyrm2']
        GFA_m2 = annual_energy_demand[building_name]['GFA_m2']
        building_EGI_kWh = building_generation_total / GFA_m2 if GFA_m2 > 0 else 0.0

        output[building_name] = {
            'E_gen_roof_kWhyr': building_generation_roof,
            'E_gen_walls_kWhyr': building_generation_walls,
            'E_gen_total_kWhyr': building_generation_total,
            'PV_roof_area_m2': roof_area,
            'PV_walls_area_m2': walls_area,
            'PV_total_area_m2': total_area,
            'total_radiation_kWhyr': total_radiation,
            'GFA_m2': GFA_m2,
            'EGI_kWhyrm2': building_EGI_kWh,
            'EUI_kWhyrm2': building_EUI_kWh,
            'EGI_EUI_ratio': building_EGI_kWh / building_EUI_kWh if building_EUI_kWh > 0 else 0
        }

    # write output as JSON file
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    return output


def make_ah_emissions(ah_emissions_path: str, output_path: str) -> None:
    headers = [
        'folder_name', 'datetime', 'entity_name',
        'AH_0:KW', 'AH_1:KW', 'AH_2:KW', 'AH_3:KW', 'AH_4:KW', 'AH_5:KW',
        'AH_6:KW', 'AH_7:KW', 'AH_8:KW', 'AH_9:KW', 'AH_10:KW', 'AH_11:KW',
        'AH_12:KW', 'AH_13:KW', 'AH_14:KW', 'AH_15:KW', 'AH_16:KW', 'AH_17:KW',
        'AH_18:KW', 'AH_19:KW', 'AH_20:KW', 'AH_21:KW', 'AH_22:KW', 'AH_23:KW',
    ]

    conversion = {
        'W': 0.001,
        'KW': 1,
        'MW': 1000,
        'GW': 1000000
    }

    with open(output_path, 'w', newline='') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(headers)

        for root, dirs, files in os.walk(ah_emissions_path):
            for file in files:
                if file.endswith('.geojson'):
                    # read the geojson content
                    with open(os.path.join(root, file), 'r') as f_geojson:
                        content = json.load(f_geojson)

                    # process every feature
                    for feature in content['features']:
                        building_name = feature['properties'].get('building_name')
                        network_name = feature['properties'].get('network_name')
                        name = building_name if building_name else network_name

                        # initialise row with folder name and datetime string
                        row = [os.path.basename(root), re.search(r'(\d{8})', file).group(1), name]

                        # investigate properties
                        emissions = {h: 0 for h in range(24)}
                        for key in feature['properties']:
                            if key.startswith('AH_'):
                                # determine hour, unit and value
                                temp = key.split(':')
                                hour = int(temp[0].split('_')[1])
                                unit = temp[1].upper()
                                value = feature['properties'][key]

                                # convert value into KW
                                value = value * conversion[unit]
                                emissions[hour] = value

                        # extend the row
                        for hour in range(24):
                            row.append(emissions[hour])

                        writer.writerow(row)


def function(working_directory: str) -> None:
    working_dir = os.path.realpath(os.path.expanduser(working_directory))

    parameters_path = os.path.join(working_dir, "parameters")
    with open(parameters_path) as f:
        parameters = json.load(f)
    pv_parameters = parameters.get("pv")

    # Load CEA default config
    config = Configuration(DEFAULT_CONFIG)
    config.project = os.path.join(working_dir, DUCT_CEA_PROJECT_NAME)
    config.scenario_name = DUCT_CEA_SCENARIO_NAME

    # Load heat rejection plugin
    config.plugins = ["cea_heat_rejection_plugin.heat_rejection.HeatRejectionPlugin"]

    try:
        locator = InputLocator(config.scenario)

        prepare_scenario(working_dir, locator)
        print_progress(15)

        weather_file_path = os.path.join(locator.get_input_folder(), 'weather', 'weather.epw')
        ah_sampling_dates = determine_ah_sample_dates(weather_file_path)
        print_progress(30)

        cea_workflow(config, ah_sampling_dates, pv_parameters)
        print_progress(80)

        # post-process building demand data
        building_demand_path = os.path.join(working_dir, 'annual_energy_demand')
        annual_energy_demand = make_annual_energy_demand(locator.get_demand_results_folder(),
                                                         locator.get_building_weekly_schedules_folder(),
                                                         building_demand_path)
        print_output("annual_energy_demand")
        print_progress(90)

        # post-process PV potential
        solar_path = os.path.join(locator.scenario, "outputs", "data", "potentials", "solar")
        pv_potential_path = os.path.join(working_dir, 'pv_potential')
        make_pv_potential(solar_path, pv_potential_path, annual_energy_demand)
        print_output("pv_potential")
        print_progress(95)

        # post-process AH emissions
        ah_emissions_path = os.path.join(working_dir, 'ah_emissions')
        make_ah_emissions(locator.get_ah_emission_results_folder(), ah_emissions_path)
        print_output("ah_emissions")
        print_progress(100)

    except Exception as e:
        print_message("error", str(e))
        raise


if __name__ == '__main__':
    function(sys.argv[1])
