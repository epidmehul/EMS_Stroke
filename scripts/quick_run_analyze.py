from sim_code.stroke_simulation import *
from sim_code.postprocess_simulation_results import *
import argparse
import pathlib
# import multiprocessing as mp

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seeds', help = 'number of random seeds', type = int, default = 100)
parser.add_argument('-p', '--patients', help = 'number of patients', type = int, default = 1000)
parser.add_argument('-c', '--config', help = 'config file with simulation parameters', type = pathlib.Path, default = None)
parser.add_argument('-d', '--data', help = 'data file containing patient information on hexes and LKW times', type = pathlib.Path, default = None)
parser.add_argument('-t', '--times', help = 'data file containing travel times from hexes and hospitals to hospitals', type = pathlib.Path, default = None)
parser.add_argument('-m', '--map_seed', help = 'map number to save results under', type = int, default = 0)
parser.add_argument('-o', '--output', help = 'output directory for simulation results', type = pathlib.Path, default = pathlib.Path(__file__).parent / 'output')
args = parser.parse_args()

if __name__ == '__main__':
    config = read_config(args.config, args.data, args.times, None)
    df = run_map_simulations(map_seeds = [args.map_seed], 
                             num_patients = args.patients, 
                             num_patient_seeds = args.seeds, 
                             output_dir = args.output / 'sim_output', 
                             config = config,
                             save_format = 'parquet',
                             additional_file_name = None,
                             )
    process_data(df = df, 
                 output_dir = args.output / 'results',
                 map_number = args.map_seed)