import csv
import json
import re
import sys
import os.path
import logging
from zipfile import ZipFile

from cea.config import Configuration, DEFAULT_CONFIG
from cea.inputlocator import InputLocator

DUCT_CEA_PROJECT_NAME = 'duct-project'
DUCT_CEA_SCENARIO_NAME = 'duct-cea'


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


def cea_workflow(cea_config: Configuration) -> None:
    # Import all CEA scripts directly
    from cea.datamanagement.terrain_helper import main as terrain_helper
    from cea.datamanagement.streets_helper import main as streets_helper
    from cea.datamanagement.archetypes_mapper import main as archetypes_mapper
    from cea.demand.schedule_maker.schedule_maker import main as schedule_maker
    from cea.resources.radiation.main import main as radiation
    from cea.demand.demand_main import main as demand
    from cea.resources.geothermal import main as shallow_geothermal_potential
    from cea.technologies.solar.photovoltaic import main as photovoltaic
    from cea.technologies.solar.photovoltaic_thermal import main as photovoltaic_thermal
    from cea.optimization_new.domain import main as optimization_new

    # Set config
    cea_config.demand.overheating_warning = False
    cea_config.optimization_new.network_type = "DC"
    cea_config.optimization_new.objective_functions = ["cost", "system_energy_demand", "anthropogenic_heat"]
    cea_config.optimization_new.generate_detailed_outputs = True
    cea_config.optimization_new.maximum_number_of_networks = 1
    cea_config.optimization_new.retain_run_results = False
    cea_config.optimization_new.networks_mutation_method = "ShuffleIndexes"
    cea_config.optimization_new.networks_crossover_method = "Uniform"

    # Generate required input files
    print_message("info", "Generating CEA inputs")
    terrain_helper(cea_config)
    streets_helper(cea_config)
    archetypes_mapper(cea_config)
    schedule_maker(cea_config)
    print_progress(15)

    # Calculate demand
    print_message("info", "Calculating building demand")
    radiation(cea_config)
    demand(cea_config)
    print_progress(30)

    # Calculate potentials
    # print_message("info", "Calculating energy potentials")
    # shallow_geothermal_potential(cea_config)
    # photovoltaic(cea_config)
    # photovoltaic_thermal(cea_config)
    # print_progress(45)

    # Run optimization
    print_message("info", "Running optimization")
    optimization_new(cea_config)


def read_supply_system(system_path: str) -> dict:
    result = {}

    # read supply system summary
    with open(os.path.join(system_path, 'Supply_systems', 'Supply_systems_summary.csv'), mode='r') as f0:
        for row0 in csv.DictReader(f0.readlines()[:-1]):
            name = str(row0['Supply_System'])

            # store the summary
            result[name] = {
                'summary': {
                    'Heat_Emissions_kWh': float(row0['Heat_Emissions_kWh']),
                    'System_Energy_Demand_kWh': float(row0['System_Energy_Demand_kWh']),
                    'GHG_Emissions_kgCO2': float(row0['GHG_Emissions_kgCO2']),
                    'Cost_USD': float(row0['Cost_USD'])
                }
            }

            # read supply system structures
            result[name]['structure'] = []
            structure_path = os.path.join(system_path, 'Supply_systems', f'{name}_supply_system_structure.csv')
            with open(structure_path, mode='r') as f1:
                for row1 in csv.DictReader(f1):
                    result[name]['structure'].append({
                        'Component': str(row1['Component']),
                        'Component_type': str(row1['Component_type']),
                        'Component_code': str(row1['Component_code']),
                        'Category': str(row1['Category']),
                        'Capacity_kW': float(row1['Capacity_kW']),
                        'Main_side': str(row1['Main_side']),
                        'Main_energy_carrier': str(row1['Main_energy_carrier']),
                        'Main_energy_carrier_code': str(row1['Main_energy_carrier_code']),
                        'Other_inputs': str(row1['Other_inputs']),
                        'Other_outputs': str(row1['Other_outputs'])
                    })

            # is there a network layout?
            network_path = os.path.join(system_path, 'networks', f'{name}_layout.geojson')
            if os.path.isfile(network_path):
                with open(network_path, mode='r') as f2:
                    result[name]['network'] = json.load(f2)
            else:
                result[name]['network'] = None

    return result


def make_supply_systems_output(working_dir: str, output_path: str) -> dict:
    result = {
        'DES': {},
        'DCS': {}
    }

    # analyse the results folder
    results_path = os.path.join(working_dir, DUCT_CEA_PROJECT_NAME, DUCT_CEA_SCENARIO_NAME,
                                "outputs", "data", "optimization", "centralized")

    # do we have the default solution? ie., without district cooling?
    current_des_path = os.path.join(results_path, 'current_DES')
    if not os.path.isdir(current_des_path):
        raise RuntimeError(f"No solution found for current DES at {current_des_path}")

    # read the current DES summary
    result['DES'] = read_supply_system(current_des_path)

    # read the DCS summaries
    for item in os.listdir(results_path):
        if item.startswith("DCS"):
            result['DCS'][item] = read_supply_system(os.path.join(results_path, item))

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result


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

    # Load CEA default config
    config = Configuration(DEFAULT_CONFIG)
    config.project = os.path.join(working_dir, DUCT_CEA_PROJECT_NAME)
    config.scenario_name = DUCT_CEA_SCENARIO_NAME

    try:
        locator = InputLocator(config.scenario)

        # unpack CEA run package
        cea_run_package = os.path.join(working_dir, "cea_run_package")
        with ZipFile(cea_run_package, "r") as zip_file:
            zip_file.extractall(locator.get_input_folder())
        print_progress(5)

        # unpack CEA databases
        cea_databases = os.path.join(working_dir, "cea_databases")
        with ZipFile(cea_databases, "r") as zip_file:
            zip_file.extractall(locator.get_databases_folder())
        print_progress(10)

        # execute CEA workflow
        cea_workflow(config)
        print_progress(90)

        # prepare output: supply_systems
        supply_systems_path = os.path.join(working_dir, 'supply_systems')
        make_supply_systems_output(working_dir, supply_systems_path)
        print_output("supply_systems")
        print_progress(95)

        # prepare output: ah_emissions
        ah_emissions_path = os.path.join(working_dir, 'ah_emissions')
        make_ah_emissions(locator.get_ah_emission_results_folder(), ah_emissions_path)
        print_output("ah_emissions")
        print_progress(100)

    except Exception as e:
        print_message("error", str(e))
        raise


if __name__ == '__main__':
    function(sys.argv[1])
