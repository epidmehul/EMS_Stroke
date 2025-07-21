from stroke_simulation import map_to_config
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--map', help = 'map seed to generate config file for', type = int)
parser.add_argument('-o', '--output', help = 'output directory for config file created from map', type = pathlib.Path, default = '../input_data')
args = parser.parse_args()

map_to_config(args.map, args.output / f'map_{args.map}.csv', export = True)