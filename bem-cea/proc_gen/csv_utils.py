from zipfile import ZipFile, is_zipfile
import pandas as pd
import os
from typing import List, Dict, Union, Tuple


class ManageBuildingTypeCSV:
    supporting_building_type = ('office', 'residential', 'hotel', 'retail')
    attribute_map = {
        'Window to wall ratio': ['window_to_wall_ratio', 'int'],
        'Roof Thermal Transmittance (U-value)': ['u_roof', 'float'],
        'Wall Thermal Transmittance (U-value)': ['u_wall', 'float'],
        'Window Thermal Transmittance (U-value)': ['u_win', 'float'],
        'Window Solar Transmittance (g-value)': ['g_win', 'int'],
        'HVAC Efficiency': ['efficiency', 'float'],
        'Temperature setpoint': ['tcs_set_c', 'int'],
        'Peak appliance load': ['ea_wm2', 'int'],
        'Peak lighting load': ['el_wm2', 'int'],
        'Peak hot water demand': ['vw_ldp', 'int'],
        'Electrified space': ['es', 'int'],
        'Conditioned space': ['hs_ag', 'int'],
        'Net GFA': ['ns', 'int'],
        'Weekday': 'weekday_schedule',
        'Weekend': 'weekend_schedule'
    }

    percentage_list = ('window_to_wall_ratio', 'g_win', 'es', 'hs_ag', 'ns')

    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path

    def _set_percentage(self, key: str, value: int) -> Union[float, int]:
        if key in self.percentage_list:
            return value / 100
        else:
            return value

    def _set_attribute_val(self, return_dict: dict, row_val: list) -> bool:
        key, unit, val = row_val[0], row_val[1], row_val[2]
        if key not in self.attribute_map:
            return False
        else:
            if key in ('Weekday', 'Weekend'):
                mapped_key = self.attribute_map[key]
                return_dict[mapped_key][unit] = float(val)
            else:
                mapped_key, data_type = self.attribute_map[key]
                if data_type == 'float':
                    return_dict[mapped_key] = float(val)
                elif data_type == 'int':
                    val = int(val)
                    if mapped_key in self.percentage_list:
                        val = self._set_percentage(mapped_key, val)
                        return_dict[mapped_key] = val
                    else:
                        return_dict[mapped_key] = val
        return True

    def read_building_type_info(self) -> dict:
        return_dict = {
            'csv_file_name': os.path.basename(self.csv_path),
            'base_building_type': '',
            'weekday_schedule': {},
            'weekend_schedule': {}
        }

        df = pd.read_csv(self.csv_path)
        validate_result, validate_info = self._validate_csv()
        if validate_result is False:
            raise TypeError(validate_info)

        for index, row in df.iterrows():
            if index == 0:
                return_dict['base_building_type'] = row.index[1]
            else:
                self._set_attribute_val(return_dict, list(row))

        return return_dict

    def _validate_csv(self) -> Tuple[bool, str]:
        try:
            val_df = pd.read_csv(self.csv_path)
            # Validate each row against the attribute map
            for index, row in val_df.iterrows():
                attribute = row.iloc[0]
                unit = row.iloc[1]
                value = row.iloc[2]

                # the first row shall indicate the building type of the new standard
                if index == 0:
                    building_type = row.index[1]
                    if building_type not in self.supporting_building_type:
                        return (False,
                            f"The first line of the csv must indicate the building type {','.join(self.supporting_building_type)} of the standard")

                if attribute in self.attribute_map and len(self.attribute_map[attribute]) == 2:
                    expected_dtype = self.attribute_map[attribute][1]
                    actual_dtype = None
                    if expected_dtype == 'int':
                        try:
                            int(value)
                            actual_dtype = 'int'
                        except ValueError:
                            actual_dtype = 'other'
                    elif expected_dtype == 'float':
                        try:
                            float(value)
                            actual_dtype = 'float'
                        except ValueError:
                            actual_dtype = 'other'

                    if expected_dtype != actual_dtype:
                        return (False,
                                f"Data type mismatch for '{attribute}' at row {index + 1}. Expected {expected_dtype}, got {actual_dtype}.")
                elif attribute in ('Weekday', 'Weekend'):
                    expected_hour_values = ('0','1','2','3','4','5','6','7','8','9','10','11','12',
                                    '13','14','15','16','17','18','19','20','21','22','23')
                    if unit not in expected_hour_values:
                        return (False,
                                f"Hour value mismatch for '{unit}' at row {index + 1}. Expected 0 to 23, got {unit}.")
            # If all data types match
            return True, "Data types in the CSV file conform to the attribute map."
        except Exception as e:
            return False, f"Error processing the file: {e}"

    @staticmethod
    def get_base_building_type(base_building_type: str) -> str:
        base_type_map = {
            'residential': 'RES_CONDO_BASELINE',
            'office': 'OFFICE_BASELINE',
            'hotel': 'HOTEL_BASELINE',
            'retail': 'RETAIL_BASELINE'
        }
        if base_building_type not in base_type_map:
            raise KeyError(f'{base_building_type} is not a valid CEA building type.')
        else:
            return base_type_map[base_building_type]

    @staticmethod
    def detect_new_building_standards(working_path: str) -> List[Dict[str, str]]:
        new_standard_list = []
        for file in os.listdir(working_path):
            file_path = os.path.join(working_path, file)
            if file.startswith('bld_eff_standards') and is_zipfile(file_path):
                with ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(working_path)

        for file in os.listdir(working_path):
            if file.upper().endswith('CSV'):
                csv_path = os.path.join(working_path, file)
                csv_obj = ManageBuildingTypeCSV(csv_path)
                new_standard_list.append(csv_obj.read_building_type_info())
        return new_standard_list
