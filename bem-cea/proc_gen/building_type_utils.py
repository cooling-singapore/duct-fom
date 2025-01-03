import os
import shutil
from typing import Union, Dict, Optional
import pandas as pd
from pandas import DataFrame


class NewBuildingStandardOperation:
    def __init__(self, cea_databases_path) -> None:
        self.cea_db_path = cea_databases_path
        self.cea_db_archetypes_path = os.path.join(cea_databases_path, 'archetypes')
        self.cea_db_assemblies_path = os.path.join(cea_databases_path, 'assemblies')

    def _read_xlsx(self, file_path: str, sheet_name: str) -> DataFrame:
        return pd.read_excel(file_path, sheet_name=sheet_name)

    def _read_csv(self, file_path: str, skip_rows: int = 0) -> DataFrame:
        return pd.read_csv(file_path, skiprows=skip_rows)

    def _copy_file(self, source_file_path: str, target_file_path: str):
        shutil.copyfile(source_file_path, target_file_path)

    def _save_xlsx_file(self, xlsx_path: str, dfs: Dict[str, DataFrame]):
        with pd.ExcelWriter(xlsx_path) as writer:
            for sheet_name, df in dfs.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    def _copy_xlsx_line(self, target_xlsx: DataFrame, key: str, source_val: str, target_val: str) -> Optional[DataFrame]:
        idx = target_xlsx[target_xlsx[key] == source_val].index

        if idx.empty:
            return None

        row_to_copy = target_xlsx.loc[idx]
        # Copy the row to a new row with the target key
        target_xlsx.loc[len(target_xlsx)] = row_to_copy.values.flatten()
        target_xlsx.at[len(target_xlsx) - 1, key] = target_val
        target_xlsx.reset_index(drop=True, inplace=True)

        return target_xlsx

    def _update_row_by_key(self, dataframe: DataFrame, key_column: str, key_value: str, update_values: Dict[str, Union[str, int, float]]) -> DataFrame:
        # Locate the row with the specified key value
        row_index = dataframe[dataframe[key_column] == key_value].index

        # Check if the key value exists in the DataFrame
        if not row_index.empty:
            # Update the values of indicated columns in that row
            for column, value in update_values.items():
                dataframe.at[row_index[0], column] = value
        else:
            print(f"No row with {key_column} value {key_value} found.")

        return dataframe

    def _delete_row_by_key(self, dataframe: DataFrame, key_column: str, key_value: str) -> DataFrame:
        dataframe = dataframe[dataframe[key_column] != key_value]
        return dataframe

    def _read_column_value_by_key(self, dataframe: DataFrame, key_column: str, key_value: str, column_to_read: str) -> Optional[Union[str, int, float]]:
        # Locate the row with the specified key value
        row = dataframe[dataframe[key_column] == key_value]

        # Check if the key value exists in the DataFrame
        if not row.empty:
            # Get the value of the indicated column in the located row
            column_value = row[column_to_read].iloc[0]
            return column_value
        else:
            print(f"No row with {key_column} value {key_value} found.")
            return None

    def _cal_occupancy(self, row: pd.Series, occ_m2p: float, new_standard_dict: dict) -> float:
        hour_key = str(int(row['HOUR']) - 1)
        if row['DAY'] == 'WEEKDAY':
            return new_standard_dict['weekday_schedule'][hour_key] * occ_m2p / 100
        elif row['DAY'] in ('SATURDAY', 'SUNDAY'):
            return new_standard_dict['weekend_schedule'][hour_key] * occ_m2p / 100
        else:
            raise KeyError('Unexpected key:' + row['DAY'])

    def _get_new_standard_name(self, new_standard_dict):
        base_building_type = new_standard_dict['base_building_type'].upper()
        object_id = new_standard_dict['csv_file_name'].split('.')[0]
        #sample: RESIDENTIAL_CUSTOM_34830656234bb6ed027e6bc312dbe4729da1b1cb63fe0c800ee78378f873db54
        return f'{base_building_type}_CUSTOM_{object_id}'

    def inject_new_standard_to_cea_db(self, base_building_type: str, new_standard_dict: dict):
        new_standard = self._get_new_standard_name(new_standard_dict)

        self.update_schedule(base_building_type, new_standard, new_standard_dict)
        self.update_use_types(base_building_type, new_standard, new_standard_dict)
        self.update_standard(base_building_type, new_standard, new_standard_dict)
        self.update_envelop(base_building_type, new_standard, new_standard_dict)
        self.update_supply(base_building_type, new_standard, new_standard_dict)

    def update_schedule(self, base_building_type: str, new_standard: str, new_standard_dict: dict):
        # copy schedule from the base type
        base_building_type_schedule_path = os.path.join(self.cea_db_archetypes_path, 'use_types',
                                                        base_building_type + '.csv')
        new_standard_schedule_path = os.path.join(self.cea_db_archetypes_path, 'use_types', new_standard + '.csv')
        self._copy_file(base_building_type_schedule_path, new_standard_schedule_path)
        schedule_csv = self._read_csv(new_standard_schedule_path, skip_rows=2)

        # read Occ_m2p value from USE_TYPE_PROPERTIES.xlsx
        use_type_properties_path = os.path.join(self.cea_db_archetypes_path, 'use_types', 'USE_TYPE_PROPERTIES.xlsx')
        internal_xlsx = self._read_xlsx(use_type_properties_path, 'INTERNAL_LOADS')
        occ_value = self._read_column_value_by_key(
            internal_xlsx, 'code', base_building_type, 'Occ_m2p')

        # update schedule csv
        schedule_csv['Occupancy'] = schedule_csv.apply(
            lambda row: self._cal_occupancy(row, occ_value, new_standard_dict), axis=1)

        # update the new schedule csv file
        with open(new_standard_schedule_path, "r") as file:
            lines = file.readlines()
        # Make modifications to the lines starting from the second line
        for i in range(3, len(lines)):
            line = lines[i].strip().split(',')
            line[2] = str(schedule_csv.loc[i-3, 'OCCUPANCY'])
            lines[i] = ','.join(line) + '\n'

        # Open the original CSV file for writing and overwrite its content
        with open(new_standard_schedule_path, "w") as file:
            file.writelines(lines)

    def update_standard(self, base_building_standard: str, new_standard: str, new_standard_dict: dict):
        standard_xlsx_path = os.path.join(self.cea_db_archetypes_path, 'CONSTRUCTION_STANDARD.xlsx')
        standard_definition_xlsx = self._read_xlsx(standard_xlsx_path, 'STANDARD_DEFINITION')
        standard_envelope_xlsx = self._read_xlsx(standard_xlsx_path, 'ENVELOPE_ASSEMBLIES')
        standard_hvac_xlsx = self._read_xlsx(standard_xlsx_path, 'HVAC_ASSEMBLIES')
        standard_supply_xlsx = self._read_xlsx(standard_xlsx_path, 'SUPPLY_ASSEMBLIES')
        self._copy_xlsx_line(
            standard_definition_xlsx,
            key='STANDARD', source_val=base_building_standard, target_val=new_standard
        )
        self._copy_xlsx_line(
            standard_envelope_xlsx,
            key='STANDARD', source_val=base_building_standard, target_val=new_standard
        )
        self._copy_xlsx_line(
            standard_hvac_xlsx,
            key='STANDARD', source_val=base_building_standard, target_val=new_standard
        )
        self._copy_xlsx_line(
            standard_supply_xlsx,
            key='STANDARD', source_val=base_building_standard, target_val=new_standard
        )
        # update envelop standard
        self._update_row_by_key(standard_envelope_xlsx, 'STANDARD', new_standard, {
            'Es': new_standard_dict['es'],
            'Hs_ag': new_standard_dict['hs_ag'],
            'Ns': new_standard_dict['ns'],
            'wwr_north': new_standard_dict['window_to_wall_ratio'],
            'wwr_south': new_standard_dict['window_to_wall_ratio'],
            'wwr_east': new_standard_dict['window_to_wall_ratio'],
            'wwr_west': new_standard_dict['window_to_wall_ratio'],
            'type_win': new_standard,
            'type_roof': new_standard,
            'type_part': new_standard,
            'type_wall': new_standard
        })
        # update supply cooling standard
        self._update_row_by_key(standard_supply_xlsx, 'STANDARD', new_standard, {
            'type_cs': new_standard
        })
        # save new standard
        self._save_xlsx_file(standard_xlsx_path, {
            'STANDARD_DEFINITION': standard_definition_xlsx,
            'ENVELOPE_ASSEMBLIES': standard_envelope_xlsx,
            'HVAC_ASSEMBLIES': standard_hvac_xlsx,
            'SUPPLY_ASSEMBLIES': standard_supply_xlsx
        })

    def update_supply(self, base_building_standard: str, new_standard: str, new_standard_dict: dict):
        standard_xlsx_path = os.path.join(self.cea_db_archetypes_path, 'CONSTRUCTION_STANDARD.xlsx')
        standard_supply_xlsx = self._read_xlsx(standard_xlsx_path, 'SUPPLY_ASSEMBLIES')

        source_supply_cooling_code = self._read_column_value_by_key(
            standard_supply_xlsx, 'STANDARD', base_building_standard, 'type_cs'
        )
        supply_assemblies_xlsx_path = os.path.join(self.cea_db_assemblies_path, 'SUPPLY.xlsx')
        supply_heating_xlsx = self._read_xlsx(supply_assemblies_xlsx_path, 'HEATING')
        supply_hot_water_xlsx = self._read_xlsx(supply_assemblies_xlsx_path, 'HOT_WATER')
        supply_cooling_xlsx = self._read_xlsx(supply_assemblies_xlsx_path, 'COOLING')
        supply_electricity_xlsx = self._read_xlsx(supply_assemblies_xlsx_path, 'ELECTRICITY')

        # copy base building type, and update it to new standard
        self._copy_xlsx_line(
            supply_cooling_xlsx,
            key='code', source_val=source_supply_cooling_code, target_val=new_standard
        )
        self._update_row_by_key(supply_cooling_xlsx, 'code', new_standard, {
            'efficiency': new_standard_dict['efficiency']
        })

        # save the update
        self._save_xlsx_file(supply_assemblies_xlsx_path, {
            'HEATING': supply_heating_xlsx,
            'HOT_WATER': supply_hot_water_xlsx,
            'COOLING': supply_cooling_xlsx,
            'ELECTRICITY': supply_electricity_xlsx
        })

    def update_envelop(self, base_building_standard: str, new_standard: str, new_standard_dict: dict):
        standard_xlsx_path = os.path.join(self.cea_db_archetypes_path, 'CONSTRUCTION_STANDARD.xlsx')
        standard_envelope_xlsx = self._read_xlsx(standard_xlsx_path, 'ENVELOPE_ASSEMBLIES')

        # read the win, roof, and wall standard from source standard
        source_type_win_code = self._read_column_value_by_key(

            standard_envelope_xlsx, 'STANDARD', base_building_standard, 'type_win'
        )
        source_type_roof_code = self._read_column_value_by_key(
            standard_envelope_xlsx, 'STANDARD', base_building_standard, 'type_roof'
        )
        source_type_wall_code = self._read_column_value_by_key(
            standard_envelope_xlsx, 'STANDARD', base_building_standard, 'type_wall'
        )

        # copy values from source standard
        envelop_assemblies_xlsx_path = os.path.join(self.cea_db_assemblies_path, 'ENVELOPE.xlsx')
        envelop_window_xlsx = self._read_xlsx(envelop_assemblies_xlsx_path, 'WINDOW')
        envelop_roof_xlsx = self._read_xlsx(envelop_assemblies_xlsx_path, 'ROOF')
        envelop_wall_xlsx = self._read_xlsx(envelop_assemblies_xlsx_path, 'WALL')
        envelop_construction_xlsx = self._read_xlsx(envelop_assemblies_xlsx_path, 'CONSTRUCTION')
        envelop_tightness_xlsx = self._read_xlsx(envelop_assemblies_xlsx_path, 'TIGHTNESS')
        envelop_floor_xlsx = self._read_xlsx(envelop_assemblies_xlsx_path, 'FLOOR')
        envelop_shading_xlsx = self._read_xlsx(envelop_assemblies_xlsx_path, 'SHADING')

        self._copy_xlsx_line(
            envelop_window_xlsx,
            key='code', source_val=source_type_win_code, target_val=new_standard
        )
        self._copy_xlsx_line(
            envelop_roof_xlsx,
            key='code', source_val=source_type_roof_code, target_val=new_standard
        )
        self._copy_xlsx_line(
            envelop_wall_xlsx,
            key='code', source_val=source_type_wall_code, target_val=new_standard
        )
        # update values of the window, roof, and wall in the new standard
        self._update_row_by_key(envelop_window_xlsx, 'code', new_standard, {
            'U_win': new_standard_dict['u_win'],
            'G_win': new_standard_dict['g_win']
        })
        self._update_row_by_key(envelop_roof_xlsx, 'code', new_standard, {
            'U_roof': new_standard_dict['u_roof']
        })
        self._update_row_by_key(envelop_wall_xlsx, 'code', new_standard, {
            'U_wall': new_standard_dict['u_wall']
        })
        # save envelop updates
        self._save_xlsx_file(envelop_assemblies_xlsx_path, {
            'CONSTRUCTION': envelop_construction_xlsx,
            'TIGHTNESS': envelop_tightness_xlsx,
            'WINDOW': envelop_window_xlsx,
            'ROOF': envelop_roof_xlsx,
            'WALL': envelop_wall_xlsx,
            'FLOOR': envelop_floor_xlsx,
            'SHADING': envelop_shading_xlsx
        })

    def update_use_types(self, base_building_type: str, new_standard_name: str, new_standard_dict: dict):
        # Open USE_TYPE_PROPERTIES.xlsx located in archetypes\use_types
        # Copy records of the base type and rename them with new standard type
        # Update values in the records of the new building type
        use_type_properties_path = os.path.join(self.cea_db_archetypes_path, 'use_types', 'USE_TYPE_PROPERTIES.xlsx')
        # Copy and update INTERNAL_LOADS
        use_type_properties_internal_xlsx = self._read_xlsx(use_type_properties_path, 'INTERNAL_LOADS')
        self._copy_xlsx_line(
            use_type_properties_internal_xlsx,
            key='code', source_val=base_building_type, target_val=new_standard_name
        )
        self._update_row_by_key(use_type_properties_internal_xlsx, 'code', new_standard_name, {
            'Ea_Wm2': new_standard_dict['ea_wm2'],
            'El_Wm2': new_standard_dict['el_wm2'],
            'Vw_ldp': new_standard_dict['vw_ldp']
        })
        # Copy and update INDOOR_COMFORT
        use_type_properties_indoor_xlsx = self._read_xlsx(use_type_properties_path, 'INDOOR_COMFORT')
        self._copy_xlsx_line(
            use_type_properties_indoor_xlsx,
            key='code', source_val=base_building_type, target_val=new_standard_name
        )
        self._update_row_by_key(use_type_properties_indoor_xlsx, 'code', new_standard_name, {
            'Tcs_set_C': new_standard_dict['tcs_set_c']
        })
        # save updates to USE_TYPE_PROPERTIES.xlsx
        self._save_xlsx_file(use_type_properties_path, {
            'INTERNAL_LOADS': use_type_properties_internal_xlsx,
            'INDOOR_COMFORT': use_type_properties_indoor_xlsx
        })
