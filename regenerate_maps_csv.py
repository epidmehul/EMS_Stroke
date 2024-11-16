from postprocess_simulation_results import *
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help = 'output directory for simulation analysis', type = pathlib.Path, default = 'output', required = True)
args = parser.parse_args()

maps_nums = [31, 93, 126, 244, 414, 671, 701, 703, 711, 984]

if __name__ == '__main__':
    for map_seed in maps_nums:
        map_specific_path = args.path / 'map_{str(map_number).zfill(3)}'
        generate_maps_csv(map_seed, map_specific_path)