from postprocess_simulation_results import *
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help = 'output directory for simulation analysis', type = pathlib.Path, default = 'output', required = True)
args = parser.parse_args()

maps_nums = [31, 93, 126, 244, 414, 671, 701, 703, 711, 984]

maps_nums = [i for i in range(1000)]

if __name__ == '__main__':
    maps_csv_path = args.path.parent / 'maps.csv'
    if maps_csv_path.exists():
        maps_csv_path.unlink()
    for map_seed in maps_nums:
        map_specific_path = args.path / f'map_{str(map_seed).zfill(3)}'
        generate_maps_csv(map_seed, map_specific_path, save = False)