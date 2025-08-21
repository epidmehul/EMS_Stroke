from stroke_simulation import *
from postprocess_simulation_results import *
import multiprocessing as mp
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seeds', help = 'number of random seeds', type = int, default = 100)
parser.add_argument('-p', '--patients', help = 'number of patients', type = int, default = 1000)
parser.add_argument('-c', '--config', help = 'config file with simulation parameters', type = pathlib.Path, default = None)
parser.add_argument('-d', '--data', help = 'data file containing patient information on hexes and LKW times', type = pathlib.Path, default = None)
parser.add_argument('-t', '--times', help = 'data file containing travel times from hexes and hospitals to hospitals', type = pathlib.Path, default = None)
parser.add_argument('-w', '--width', help = 'confidence interval width as a proportion', type = float, default = 0.95)
parser.add_argument('-m', '--map_seed', help = 'map number to save results under', type = int, default = 0)
# parser.add_argument('-n', '--n_cores', help = 'number of cores for mp.Pool', type = int, default = 10)
parser.add_argument('-o', '--output', help = 'output directory for simulation and analysis', type = pathlib.Path, default = '/work/users/p/w/pwlin/new_output')

args = parser.parse_args()

map_seeds = [args.map_seed]
# map_seeds = [i for i in range(1000)]
patient_seeds = [i for i in range(args.seeds)]
# output_dir = args.output
output_dir = pathlib.Path('/work/users/p/w/pwlin/sampson_sensitivity')

config_dict = read_config(args.config, None, args.times)

hex_info = pd.read_csv(args.data)
config_dict['hexes'] = dict(hex_info.groupby('Hex').agg(lambda x: x)['n'])

def run_analyze(map_seed):
    df = run_map_simulations([map_seed], num_patients = args.patients, num_patient_seeds = args.seeds, save_format = 'parquet', output_dir = output_dir / 'parquet_files', config = config_dict)

    high_df = run_map_simulations([map_seed], num_patients = args.patients, num_patient_seeds = args.seeds, save_format = 'parquet', output_dir = output_dir / 'parquet_files', config = config_dict | {'patients_lvo_ischemic': 0.341})

    low_df = run_map_simulations([map_seed], num_patients = args.patients, num_patient_seeds = args.seeds, save_format = 'parquet', output_dir = output_dir / 'parquet_files', config = config_dict {'patients_lvo_ischemic': 0.141})
    
    cohort_avgs, intervals = process_data(df = df, output_dir = output_dir / 'results', map_number = map_seed, save_format = 'csv', psc_only = False, config = config_dict, errorbars = True)
    
    high_cohort_avgs, high_intervals = process_data(df = high_df, output_dir = output_dir / 'results', map_number = map_seed, save_format = 'csv', psc_only = False, config = config_dict, errorbars = True, additional_file_name = 'high')

    low_cohort_avgs, low_intervals = process_data(df = low_df, output_dir = output_dir / 'results', map_number = map_seed, save_format = 'csv', psc_only = False, config = config_dict, errorbars = True, additional_file_name = 'low')

if __name__ == '__main__':
    if not output_dir.exists():
        output_dir.mkdir(parents = True)
    run_analyze(args.map_seed)
    # with mp.Pool(args.n_cores) as pool:
        # pool.map(run_single_map, map_seeds)
        # results = pool.map(analyze_parquet, map_seeds)
    # pd.concat(results, axis = 0).to_csv(data_calcs_csv_path, header = True, index = False, mode = 'w')