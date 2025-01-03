import os
import tempfile
import shutil
import igraph

from proc_uvp.processor import function as function_uvp
from proc_uwc.processor import function as function_uwc

test_data_path = os.environ['TEST_DATA_PATH']


def copy_input_files(wd_path: str, mapping: dict) -> None:
    for filename, source_path in mapping.items():
        destination_path = os.path.join(wd_path, filename)
        shutil.copyfile(source_path, destination_path)


def copy_output_files(wd_path: str, mapping: dict) -> None:
    for filename, destination_path in mapping.items():
        source_path = os.path.join(wd_path, filename)
        assert os.path.isfile(source_path)
        shutil.copyfile(source_path, destination_path)


def test_proc_uvp():
    with tempfile.TemporaryDirectory() as wd_path:
        # copy input files
        copy_input_files(wd_path, {
            'parameters': os.path.join(test_data_path, 'ucmmva_uvp', 'input', 'parameters'),
            'building-footprints': os.path.join(test_data_path, 'ucmmva_uvp', 'input', 'building-footprints'),
            'land-mask': os.path.join(test_data_path, 'ucmmva_uvp', 'input', 'city-admin-areas')
        })

        function_uvp(wd_path)
        assert True

        # copy output files
        copy_output_files(wd_path, {
            'hotspots': os.path.join(test_data_path, 'ucmmva_uvp', 'output', 'hotspots'),
        })


def test_proc_uwc_graph():
    edges = [
        (11, 21), (11, 12), (11, 22),
        (21, 11), (21, 31), (21, 22),
         (31, 21), (31, 32),
         (12, 11), (12, 22), (12, 13),
         (22, 21), (22, 12), (22, 32), (22, 23), (22, 33),
         (32, 31), (32, 22), (32, 33),
         (13, 12), (13, 23),
         (23, 13), (23, 22), (23, 33),
         (33, 32), (33, 23)
    ]
    weights = [
        1, 1, 1,
        1, 1, 1,
        1, 1,
        1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1,
        1, 1,
        1, 1, 1,
        1, 1
    ]

    g = igraph.Graph(edges, directed=True)
    print(g)

    g.es['weight'] = weights

    paths = g.get_shortest_paths(11, 33, weights='weight')
    assert len(paths) == 1
    print(paths[0])
    assert paths[0] == [11, 22, 33]

    paths = g.get_shortest_paths(12, 33, weights='weight')
    assert len(paths) == 1
    print(paths[0])
    assert paths[0] == [12, 22, 33]


def test_proc_uwc():
    with tempfile.TemporaryDirectory() as wd_path:
        # copy input files
        copy_input_files(wd_path, {
            'parameters': os.path.join(test_data_path, 'ucmmva_uwc', 'input', 'parameters'),
            'building-footprints': os.path.join(test_data_path, 'ucmmva_uwc', 'input', 'building-footprints'),
            'land-mask': os.path.join(test_data_path, 'ucmmva_uwc', 'input', 'city-admin-areas')
        })

        function_uwc(wd_path)
        assert True

        # copy output files
        copy_output_files(wd_path, {
            'wind-corridors-ns': os.path.join(test_data_path, 'ucmmva_uwc', 'output', 'wind-corridors-ns'),
            'wind-corridors-ew': os.path.join(test_data_path, 'ucmmva_uwc', 'output', 'wind-corridors-ew'),
            'wind-corridors-nwse': os.path.join(test_data_path, 'ucmmva_uwc', 'output', 'wind-corridors-nwse'),
            'wind-corridors-nesw': os.path.join(test_data_path, 'ucmmva_uwc', 'output', 'wind-corridors-nesw'),
            'building-footprints': os.path.join(test_data_path, 'ucmmva_uwc', 'output', 'building-footprints'),
            'land-mask': os.path.join(test_data_path, 'ucmmva_uwc', 'output', 'land-mask')
        })
