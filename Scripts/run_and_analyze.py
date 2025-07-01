from stroke_simulation import *
from postprocess_simulation_results import *
import multiprocessing as mp
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seeds', help = 'number of random seeds', type = int, default = 40)
parser.add_argument('-p', '--patients', help = 'number of patients', type = int, default = 1000)
parser.add_argument('-c', '--config', help = 'config file with simulation parameters', type = pathlib.Path, default = None)
parser.add_argument('-d', '--data', help = 'data file containing patient information on hexes and LKW times', type = pathlib.Path, default = None)
parser.add_argument('-t', '--times', help = 'data file containing travel times from hexes and hospitals to hospitals', type = pathlib.Path, default = None)
parser.add_argument('-n', '--n_cores', help = 'number of cores for mp.Pool', type = int, default = 10)
args = parser.parse_args()

map_seeds = [0]

num_cores = args.n_cores

map_seeds = [i for i in range(1000)]
patient_seeds = [i for i in range(args.seeds)]
output_dir = '/work/users/p/w/pwlin/output2/parquet_files'

try:
    config_dict = read_config(args.config, args.data, args.times)
    if config_dict['hexes'] is not None:
        map_seeds = [0]
except:
    config_dict = None

def run_single_map(map_seed):
    run_map_simulations([map_seed], num_patients = args.patients, num_patient_seeds = args.seeds, save_format = 'parquet', output_dir = output_dir, config = config_dict)

def analyze_parquet(map_num):
    file_name = f'map_{str(map_num).zfill(3)}.parquet'
    df = read_output(pathlib.Path(output_dir) / file_name, save_format = 'parquet')
    return single_map_analysis_output(df, map_number = map_num, heatmap_diff = True, save = True, output_dir_str = '/work/users/p/w/pwlin/output2/results', line_errorbars = True, generated_map = False) 

if __name__ == '__main__':
    output_dir_path = pathlib.Path(output_dir)
    data_calcs_csv_path = pathlib.Path(output_dir_path.parent / 'all_results.csv')
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents = True)
    if not data_calcs_csv_path.parent.exists():
        data_calcs_csv_path.parent.mkdir(parents = True)
    with mp.Pool(num_cores) as pool:
        pool.map(run_single_map, map_seeds)
        results = pool.map(analyze_parquet, map_seeds)
    pd.concat(results, axis = 0).to_csv(data_calcs_csv_path, header = True, index = False, mode = 'w')