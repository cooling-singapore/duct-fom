import sys

from proc_gen.building_type_utils import NewBuildingStandardOperation

sys.path.append('../proc_gen')

from proc_gen.csv_utils import ManageBuildingTypeCSV
from proc_gen.processor import validate_building_footprints, function


import os
import geopandas as gpd
import pandas as pd
import tempfile
import shutil
import json

BEMCEA_GEN_PATH = os.path.join(os.environ['TEST_DATA_PATH'], 'bemcea_gen')


def test_read_building_csv():
    hotel_csv_path = os.path.join(BEMCEA_GEN_PATH, 'building_type_csv', 'SimplifiedUseTypeDescription_Hotel_HOTEL.csv')
    csv_reader = ManageBuildingTypeCSV(hotel_csv_path)
    hotel_dict = csv_reader.read_building_type_info()
    target_dict = {
        'csv_file_name': 'SimplifiedUseTypeDescription_Hotel_HOTEL.csv',
        'weekday_schedule': {
            '0': '3.91',
            '1': '3.91',
            '2': '3.91',
            '3': '3.91',
            '4': '3.91',
            '5': '3.91',
            '6': '3.04',
            '7': '1.74',
            '8': '1.74',
            '9': '0.87',
            '10': '0.87',
            '11': '0.87',
            '12': '0.87',
            '13': '0.87',
            '14': '0.87',
            '15': '1.30',
            '16': '2.17',
            '17': '2.17',
            '18': '2.17',
            '19': '3.04',
            '20': '3.04',
            '21': '3.48',
            '22': '3.91',
            '23': '3.91'
        },
        'weekend_schedule': {
            '0': '3.91',
            '1': '3.91',
            '2': '3.91',
            '3': '3.91',
            '4': '3.91',
            '5': '3.91',
            '6': '3.04',
            '7': '1.74',
            '8': '1.74',
            '9': '0.87',
            '10': '0.87',
            '11': '0.87',
            '12': '0.87',
            '13': '0.87',
            '14': '0.87',
            '15': '1.30',
            '16': '2.17',
            '17': '2.17',
            '18': '2.17',
            '19': '3.04',
            '20': '3.04',
            '21': '3.48',
            '22': '3.91',
            '23': '3.91'
        },
        'window_to_wall_ratio': '29',
        'u_roof': '0.6',
        'u_wall': '0.8',
        'u_win': '5.4',
        'g_win': '85',
        'efficiency': '2.75',
        'tcs_set_c': '24',
        'ea_wm2': '4.3',
        'el_wm2': '3.1',
        'vw_ldp': '40',
        'es': '90',
        'hs_ag': '25',
        'ns': '100'
    }
    assert hotel_dict == target_dict

    office_csv_path = os.path.join(BEMCEA_GEN_PATH, 'building_type_csv', 'SimplifiedUseTypeDescription_Office_OFFICE.csv')
    csv_reader = ManageBuildingTypeCSV(office_csv_path)
    office_dict = csv_reader.read_building_type_info()
    target_dict = {
        'csv_file_name': 'SimplifiedUseTypeDescription_Office_OFFICE.csv',
        'weekday_schedule': {
            '0': '0.00',
            '1': '0.00',
            '2': '0.00',
            '3': '0.00',
            '4': '0.00',
            '5': '0.00',
            '6': '1.00',
            '7': '2.00',
            '8': '9.50',
            '9': '9.50',
            '10': '9.50',
            '11': '9.50',
            '12': '5.00',
            '13': '9.50',
            '14': '9.50',
            '15': '9.50',
            '16': '9.50',
            '17': '3.00',
            '18': '1.00',
            '19': '1.00',
            '20': '1.00',
            '21': '1.00',
            '22': '0.50',
            '23': '0.50'
        },
        'weekend_schedule': {
            '0': '0.00',
            '1': '0.00',
            '2': '0.00',
            '3': '0.00',
            '4': '0.00',
            '5': '0.00',
            '6': '0.75',
            '7': '0.75',
            '8': '1.75',
            '9': '1.75',
            '10': '1.75',
            '11': '1.75',
            '12': '0.75',
            '13': '0.75',
            '14': '0.75',
            '15': '0.75',
            '16': '0.75',
            '17': '0.50',
            '18': '0.25',
            '19': '0.00',
            '20': '0.00',
            '21': '0.00',
            '22': '0.00',
            '23': '0.00'
        },
        'window_to_wall_ratio': '29',
        'u_roof': '0.6',
        'u_wall': '0.8',
        'u_win': '5.4',
        'g_win': '85',
        'efficiency': '2.75',
        'tcs_set_c': '24',
        'ea_wm2': '11',
        'el_wm2': '10',
        'vw_ldp': '0',
        'es': '90',
        'hs_ag': '25',
        'ns': '100'
    }
    assert office_dict == target_dict


def test_validate_csv():
    hotel_csv_path = os.path.join(BEMCEA_GEN_PATH, 'building_type_csv', 'SimplifiedUseTypeDescription_Hotel_HOTEL.csv')
    csv_reader = ManageBuildingTypeCSV(hotel_csv_path)
    val_result, val_info = csv_reader._validate_csv()
    assert val_result

    hotel_csv_path = os.path.join(BEMCEA_GEN_PATH, 'building_type_csv', 'RES_CONDO_GREEN.csv')
    csv_reader = ManageBuildingTypeCSV(hotel_csv_path)
    val_result, val_info = csv_reader._validate_csv()
    assert val_result

    hotel_csv_path = os.path.join(BEMCEA_GEN_PATH, 'building_type_csv', 'error_1.csv')
    csv_reader = ManageBuildingTypeCSV(hotel_csv_path)
    val_result, val_info = csv_reader._validate_csv()
    assert not val_result
    assert val_info == "Data type mismatch for 'Window to wall ratio' at row 3. Expected int, got other."

    hotel_csv_path = os.path.join(BEMCEA_GEN_PATH, 'building_type_csv', 'error_2.csv')
    csv_reader = ManageBuildingTypeCSV(hotel_csv_path)
    val_result, val_info = csv_reader._validate_csv()
    assert not val_result
    assert val_info == "Hour value mismatch for '24' at row 47. Expected 0 to 23, got 24."

    csv_path = os.path.join(BEMCEA_GEN_PATH, 'building_type_csv', 'error_3.csv')
    csv_reader = ManageBuildingTypeCSV(csv_path)
    val_result, val_info = csv_reader._validate_csv()
    assert not val_result
    assert val_info == "The first line of the csv must indicate the building type office,residential,hotel,retail of the standard"


def test_negative_height():
    gdf = gpd.read_file(os.path.join(BEMCEA_GEN_PATH, 'negative_height', 'negative_val'))
    try:
        validate_building_footprints(gdf)
    except ValueError as e:
        assert str(e) == 'Feature IDs with negative height values: 539766023'


def test_non_negative_height():
    gdf = gpd.read_file(os.path.join(BEMCEA_GEN_PATH, 'negative_height', 'non_negative_val'))
    assert validate_building_footprints(gdf)


def test_duplicated_geometry():
    gdf = gpd.read_file(os.path.join(BEMCEA_GEN_PATH, 'duplicate_geometries', 'duplicated'))
    try:
        validate_building_footprints(gdf)
    except ValueError as e:
        assert str(e) == 'Feature IDs with duplicated geometries: 152598430, 172023449, 172023457, 152598430, 172023449, 172023457'


def test_non_duplicated_geometry():
    gdf = gpd.read_file(os.path.join(BEMCEA_GEN_PATH, 'duplicate_geometries', 'non_duplicated'))
    assert validate_building_footprints(gdf)


def test_non_polygon_geometry():
    gdf = gpd.read_file(os.path.join(BEMCEA_GEN_PATH, 'non_polygon_geometries', 'building_footprints'))
    try:
        validate_building_footprints(gdf)
    except ValueError as e:
        assert str(e) == 'Feature IDs with non-polygon geometries: 451950402'


def test_no_demand_geojson():
    with tempfile.TemporaryDirectory() as temp_dir:
        shutil.copyfile(
            os.path.join(BEMCEA_GEN_PATH, 'no_demand', "building_footprints"),
            os.path.join(temp_dir, "building_footprints")
        )
        shutil.copyfile(
            os.path.join(BEMCEA_GEN_PATH, 'no_demand', "parameters"),
            os.path.join(temp_dir, "parameters")
        )

        try:
            function(temp_dir)
        except ValueError as e:
            assert str(e) == "There's no cooling demand in the building_footprints"


def test_get_base_building_type():
    base_type = ManageBuildingTypeCSV.get_base_building_type('residential')
    assert base_type == 'RES_CONDO_BASELINE'


def test_read_xlsx():
    building_obj = NewBuildingStandardOperation(os.path.join(BEMCEA_GEN_PATH, 'cea_files'))
    xlsx_path = os.path.join(BEMCEA_GEN_PATH, 'cea_files', 'archetypes', 'CONSTRUCTION_STANDARD.xlsx')
    xlsx_obj = building_obj._read_xlsx(xlsx_path, 'STANDARD_DEFINITION')
    assert isinstance(xlsx_obj, pd.DataFrame)


def test_copy_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        building_obj = NewBuildingStandardOperation(os.path.join(BEMCEA_GEN_PATH, 'cea_files'))
        source_path = os.path.join(BEMCEA_GEN_PATH, 'cea_files', 'archetypes', 'CONSTRUCTION_STANDARD.xlsx')
        target_path = os.path.join(temp_dir, 'CONSTRUCTION_STANDARD.xlsx')
        building_obj._copy_file(source_path, target_path)
        assert os.path.exists(target_path)


def test_copy_xlsx_line():
    with tempfile.TemporaryDirectory() as temp_dir:
        wd_path = os.path.join(temp_dir, 'cea_files')
        shutil.copytree(os.path.join(BEMCEA_GEN_PATH, 'cea_files'), wd_path)

        building_obj = NewBuildingStandardOperation(wd_path)
        xlsx_path = os.path.join(wd_path, 'archetypes', 'CONSTRUCTION_STANDARD.xlsx')
        xlsx_obj = building_obj._read_xlsx(xlsx_path, 'STANDARD_DEFINITION')

        building_obj._copy_xlsx_line(xlsx_obj, 'STANDARD', 'STANDARD1', 'NEW_STANDARD')
        located_row = xlsx_obj[xlsx_obj['STANDARD'] == 'NEW_STANDARD']
        assert not located_row.empty
        # located_row.iloc[0]
        assert located_row.iloc[0]['STANDARD'] == 'NEW_STANDARD'
        assert located_row.iloc[0]['YEAR_START'] == 1000
        assert located_row.iloc[0]['YEAR_END'] == 2040


def test_update_row_by_key():
    with tempfile.TemporaryDirectory() as temp_dir:
        wd_path = os.path.join(temp_dir, 'cea_files')
        shutil.copytree(os.path.join(BEMCEA_GEN_PATH, 'cea_files'), wd_path)

        building_obj = NewBuildingStandardOperation(wd_path)

        xlsx_path = os.path.join(wd_path, 'archetypes', 'CONSTRUCTION_STANDARD.xlsx')
        xlsx_obj = building_obj._read_xlsx(xlsx_path, 'STANDARD_DEFINITION')
        building_obj._update_row_by_key(xlsx_obj, 'STANDARD', 'STANDARD1', {
            'YEAR_START': 1001,
            'YEAR_END': 2041
        })

        updated_year_start = building_obj._read_column_value_by_key(
            xlsx_obj,
            'STANDARD',
            'STANDARD1',
            'YEAR_START'
        )
        updated_year_end = building_obj._read_column_value_by_key(
            xlsx_obj,
            'STANDARD',
            'STANDARD1',
            'YEAR_END'
        )

        assert updated_year_start == 1001
        assert updated_year_end == 2041


def test_update_use_types():
    with tempfile.TemporaryDirectory() as temp_dir:
        wd_path = os.path.join(temp_dir, 'cea_files')
        shutil.copytree(os.path.join(BEMCEA_GEN_PATH, 'cea_files'), wd_path)

        building_obj = NewBuildingStandardOperation(wd_path)
        new_standard_dict = {
            'csv_file_name': 'RES_CONDO_GREEN.csv',
            'ea_wm2': 20,
            'efficiency': 3.88,
            'el_wm2': 30,
            'es': 95,
            'g_win': 75,
            'hs_ag': 75,
            'ns': 100,
            'tcs_set_c': 25,
            'u_roof': 0.6,
            'u_wall': 0.75,
            'u_win': 3.1,
            'vw_ldp': 41,
        }
        building_obj.update_use_types(
            'RES_CONDO_BASELINE', 'RES_CONDO_GREEN', new_standard_dict)

        # read the USE_TYPE_PROPERTIES.xlsx to check updated values
        use_type_properties_xlsx_path = os.path.join(
            building_obj.cea_db_archetypes_path, 'use_types', 'USE_TYPE_PROPERTIES.xlsx')
        use_type_properties_internal_xlsx = building_obj._read_xlsx(use_type_properties_xlsx_path, 'INTERNAL_LOADS')
        use_type_properties_indoor_xlsx = building_obj._read_xlsx(use_type_properties_xlsx_path, 'INDOOR_COMFORT')

        ea_wm2 = building_obj._read_column_value_by_key(
            use_type_properties_internal_xlsx, 'code', 'RES_CONDO_GREEN', 'Ea_Wm2')
        el_wm2 = building_obj._read_column_value_by_key(
            use_type_properties_internal_xlsx, 'code', 'RES_CONDO_GREEN', 'El_Wm2')
        vw_ldp = building_obj._read_column_value_by_key(
            use_type_properties_internal_xlsx, 'code', 'RES_CONDO_GREEN', 'Vw_ldp')
        assert ea_wm2 == new_standard_dict['ea_wm2']
        assert el_wm2 == new_standard_dict['el_wm2']
        assert vw_ldp == new_standard_dict['vw_ldp']

        tcs_set_c = building_obj._read_column_value_by_key(
            use_type_properties_indoor_xlsx, 'code', 'RES_CONDO_GREEN', 'Tcs_set_C')
        assert tcs_set_c == new_standard_dict['tcs_set_c']


def test_update_schedule():
    with tempfile.TemporaryDirectory() as temp_dir:
        wd_path = os.path.join(temp_dir, 'cea_files')
        shutil.copytree(os.path.join(BEMCEA_GEN_PATH, 'cea_files'), wd_path)

        csv_path = os.path.join(temp_dir, 'building_type_csv')
        shutil.copytree(os.path.join(BEMCEA_GEN_PATH, 'building_type_csv'), csv_path)

        green_condo_csv_path = os.path.join(csv_path, 'RES_CONDO_GREEN.csv')
        csv_reader = ManageBuildingTypeCSV(green_condo_csv_path)
        res_green_dict = csv_reader.read_building_type_info()

        building_obj = NewBuildingStandardOperation(wd_path)
        building_obj.update_schedule('RES_CONDO_BASELINE', 'RES_CONDO_GREEN', res_green_dict)

        new_building_csv_path = os.path.join(wd_path, 'archetypes', 'use_types', 'RES_CONDO_GREEN.csv')
        assert os.path.exists(new_building_csv_path)
        os.remove(new_building_csv_path)


def test_update_standard():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, 'cea_files')
        shutil.copytree(os.path.join(BEMCEA_GEN_PATH, 'cea_files'), db_path)

        building_obj = NewBuildingStandardOperation(db_path)
        new_standard_dict = {
            'csv_file_name': 'RES_CONDO_GREEN.csv',
            'window_to_wall_ratio': 29,
            'ea_wm2': 20,
            'efficiency': 3.88,
            'el_wm2': 30,
            'es': 95,
            'g_win': 75,
            'hs_ag': 75,
            'ns': 100,
            'tcs_set_c': 25,
            'u_roof': 0.6,
            'u_wall': 0.75,
            'u_win': 3.1,
            'vw_ldp': 41,
        }
        building_obj.update_standard('RES_CONDO_BASELINE', 'RES_CONDO_GREEN', new_standard_dict)

        standard_xlsx_path = os.path.join(
            building_obj.cea_db_archetypes_path, 'CONSTRUCTION_STANDARD.xlsx')
        standard_definition_xlsx = building_obj._read_xlsx(standard_xlsx_path, 'STANDARD_DEFINITION')
        standard_envelope_xlsx = building_obj._read_xlsx(standard_xlsx_path, 'ENVELOPE_ASSEMBLIES')
        standard_hvac_xlsx = building_obj._read_xlsx(standard_xlsx_path, 'HVAC_ASSEMBLIES')
        standard_supply_xlsx = building_obj._read_xlsx(standard_xlsx_path, 'SUPPLY_ASSEMBLIES')

        es = building_obj._read_column_value_by_key(
            standard_envelope_xlsx, 'STANDARD', 'RES_CONDO_GREEN', 'Es')
        hs_ag = building_obj._read_column_value_by_key(
            standard_envelope_xlsx, 'STANDARD', 'RES_CONDO_GREEN', 'Hs_ag')
        ns = building_obj._read_column_value_by_key(
            standard_envelope_xlsx, 'STANDARD', 'RES_CONDO_GREEN', 'Ns')
        wwr_north = building_obj._read_column_value_by_key(
            standard_envelope_xlsx, 'STANDARD', 'RES_CONDO_GREEN', 'wwr_north')
        wwr_south = building_obj._read_column_value_by_key(
            standard_envelope_xlsx, 'STANDARD', 'RES_CONDO_GREEN', 'wwr_south')
        wwr_east = building_obj._read_column_value_by_key(
            standard_envelope_xlsx, 'STANDARD', 'RES_CONDO_GREEN', 'wwr_east')
        wwr_west = building_obj._read_column_value_by_key(
            standard_envelope_xlsx, 'STANDARD', 'RES_CONDO_GREEN', 'wwr_west')
        type_win = building_obj._read_column_value_by_key(
            standard_envelope_xlsx, 'STANDARD', 'RES_CONDO_GREEN', 'type_win')
        type_roof = building_obj._read_column_value_by_key(
            standard_envelope_xlsx, 'STANDARD', 'RES_CONDO_GREEN', 'type_roof')
        type_part = building_obj._read_column_value_by_key(
            standard_envelope_xlsx, 'STANDARD', 'RES_CONDO_GREEN', 'type_part')
        type_wall = building_obj._read_column_value_by_key(
            standard_envelope_xlsx, 'STANDARD', 'RES_CONDO_GREEN', 'type_wall')
        assert es == new_standard_dict['es']
        assert hs_ag == new_standard_dict['hs_ag']
        assert ns == new_standard_dict['ns']
        assert wwr_north == new_standard_dict['window_to_wall_ratio']
        assert wwr_east == new_standard_dict['window_to_wall_ratio']
        assert wwr_south == new_standard_dict['window_to_wall_ratio']
        assert wwr_west == new_standard_dict['window_to_wall_ratio']
        assert type_win == 'RES_CONDO_GREEN'
        assert type_roof == 'RES_CONDO_GREEN'
        assert type_part == 'RES_CONDO_GREEN'
        assert type_wall == 'RES_CONDO_GREEN'

        type_cs = building_obj._read_column_value_by_key(
            standard_supply_xlsx, 'STANDARD', 'RES_CONDO_GREEN', 'type_cs')
        assert type_cs == 'RES_CONDO_GREEN'

        start_year = building_obj._read_column_value_by_key(
            standard_definition_xlsx, 'STANDARD', 'RES_CONDO_GREEN', 'YEAR_START')
        new_standard_name = building_obj._read_column_value_by_key(
            standard_hvac_xlsx, 'STANDARD', 'RES_CONDO_GREEN', 'STANDARD')
        assert start_year == 1000
        assert new_standard_name == 'RES_CONDO_GREEN'


def test_update_envelop():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, 'cea_files')
        shutil.copytree(os.path.join(BEMCEA_GEN_PATH, 'cea_files'), db_path)

        building_obj = NewBuildingStandardOperation(db_path)
        new_standard_dict = {
            'csv_file_name': 'RES_CONDO_GREEN.csv',
            'window_to_wall_ratio': 29,
            'ea_wm2': 20,
            'efficiency': 3.88,
            'el_wm2': 30,
            'es': 95,
            'g_win': 75,
            'hs_ag': 75,
            'ns': 100,
            'tcs_set_c': 25,
            'u_roof': 0.6,
            'u_wall': 0.75,
            'u_win': 3.1,
            'vw_ldp': 41,
        }
        building_obj.update_envelop('RES_CONDO_BASELINE', 'RES_CONDO_GREEN', new_standard_dict)

        envelop_assemblies_xlsx_path = os.path.join(building_obj.cea_db_assemblies_path, 'ENVELOPE.xlsx')
        envelop_window_xlsx = building_obj._read_xlsx(envelop_assemblies_xlsx_path, 'WINDOW')
        envelop_roof_xlsx = building_obj._read_xlsx(envelop_assemblies_xlsx_path, 'ROOF')
        envelop_wall_xlsx = building_obj._read_xlsx(envelop_assemblies_xlsx_path, 'WALL')

        u_win = building_obj._read_column_value_by_key(
            envelop_window_xlsx, 'code', 'RES_CONDO_GREEN', 'U_win')
        g_win = building_obj._read_column_value_by_key(
            envelop_window_xlsx, 'code', 'RES_CONDO_GREEN', 'G_win')
        u_roof = building_obj._read_column_value_by_key(
            envelop_roof_xlsx, 'code', 'RES_CONDO_GREEN', 'U_roof')
        u_wall = building_obj._read_column_value_by_key(
            envelop_wall_xlsx, 'code', 'RES_CONDO_GREEN', 'U_wall')

        assert u_win == new_standard_dict['u_win']
        assert g_win == new_standard_dict['g_win']
        assert u_roof == new_standard_dict['u_roof']
        assert u_wall == new_standard_dict['u_wall']


def test_update_supply():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, 'cea_files')
        shutil.copytree(os.path.join(BEMCEA_GEN_PATH, 'cea_files'), db_path)

        building_obj = NewBuildingStandardOperation(db_path)
        new_standard_dict = {
            'csv_file_name': 'RES_CONDO_GREEN.csv',
            'window_to_wall_ratio': 29,
            'ea_wm2': 20,
            'efficiency': 3.88,
            'el_wm2': 30,
            'es': 95,
            'g_win': 75,
            'hs_ag': 75,
            'ns': 100,
            'tcs_set_c': 25,
            'u_roof': 0.6,
            'u_wall': 0.75,
            'u_win': 3.1,
            'vw_ldp': 41,
        }
        building_obj.update_supply('RES_CONDO_BASELINE', 'RES_CONDO_GREEN', new_standard_dict)
        supply_assemblies_xlsx_path = os.path.join(building_obj.cea_db_assemblies_path, 'SUPPLY.xlsx')
        cooling_window_xlsx = building_obj._read_xlsx(supply_assemblies_xlsx_path, 'COOLING')
        efficiency = building_obj._read_column_value_by_key(
            cooling_window_xlsx, 'code', 'RES_CONDO_GREEN', 'efficiency')
        assert efficiency == new_standard_dict['efficiency']


DEFAULT_BUILDING_TYPE_MAP = {
    "COOLROOM": [],
    "FOODSTORE": [],
    "GYM": [],
    "HOSPITAL": ["commercial:9"],
    "HOTEL": ["commercial:2"],
    "INDUSTRIAL": ["industrial:1"],
    "LAB": [],
    "LIBRARY": [],
    "MULTI_RES": ["residential:1"],
    "MUSEUM": [],
    "OFFICE": ["commercial:1"],
    "PARKING": [],
    "RESTAURANT": [],
    "RETAIL": ["commercial:3"],
    "SCHOOL": ["commercial:7"],
    "SERVERROOM": [],
    "SINGLE_RES": [],
    "SWIMMING": [],
    "UNIVERSITY": [],
}

BASELINE_MAPPING = {
    "OFFICE_BASELINE": [
        "commercial:1"
    ],
    "RES_CONDO_BASELINE": [
        "residential:1"
    ],
    "HOTEL_BASELINE": [
        "commercial:2"
    ],
    "RETAIL_BASELINE": [
        "commercial:3"
    ]
}

SLE_MAPPING = {
    "OFFICE_SLE": [
        "commercial:1"
    ],
    "RES_CONDO_SLE": [
        "residential:1"
    ],
    "HOTEL_SLE": [
        "commercial:2"
    ],
    "RETAIL_SLE": [
        "commercial:3"
    ]
}

PARAMETERS = {
    "building_type_mapping": DEFAULT_BUILDING_TYPE_MAP,
    "default_building_type": "MULTI_RES",
    "building_standard_mapping": {},
    "default_building_standard": "STANDARD1",
    "commit_id": "f0ae196",
    "database_name": "SG",
    "terrain_height": 0,
    "weather": "Singapore-Changi_2030_AB1_TMY.epw"
}


def test_proc_baseline():
    with tempfile.TemporaryDirectory() as temp_dir:
        shutil.copyfile(
            os.path.join(BEMCEA_GEN_PATH, 'proc', "building_footprints"),
            os.path.join(temp_dir, "building_footprints")
        )

        params = PARAMETERS.copy()
        params["building_standard_mapping"] = BASELINE_MAPPING
        with open(os.path.join(temp_dir, "parameters"), "w") as f:
            json.dump(params, f)

        try:
            function(temp_dir)
            assert True

        except Exception:
            assert False


def test_proc_SLE():
    with tempfile.TemporaryDirectory() as temp_dir:
        shutil.copyfile(
            os.path.join(BEMCEA_GEN_PATH, 'proc', "building_footprints"),
            os.path.join(temp_dir, "building_footprints")
        )

        params = PARAMETERS.copy()
        params["building_standard_mapping"] = SLE_MAPPING
        with open(os.path.join(temp_dir, "parameters"), "w") as f:
            json.dump(params, f)

        try:
            function(temp_dir)
            assert True

        except Exception:
            assert False
