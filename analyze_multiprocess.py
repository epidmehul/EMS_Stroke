from postprocess_simulation_results import *
import os
import multiprocessing as mp
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seeds', help = 'number of random seeds', type = int, default = 40)
parser.add_argument('-p', '--patients', help = 'number of patients', type = int, default = 1000)
args = parser.parse_args()

# os.chdir('/proj/patellab/Sheps/output')
# os.chdir('/work/users/p/w/pwlin')

# sim_results = read_output('run_0_100.csv')

current_dir = pathlib.Path('/work/users/p/w/pwlin/output/parquet_files')

# file_names = [
#     'run_0_100.csv',
#     'run_100_200.csv',
#     'run_200_300.csv',
#     'run_300_400.csv',
#     'run_400_500.csv',
#     'run_500_600.csv',
#     'run_600_700.csv',
#     'run_700_800.csv',
#     'run_800_900.csv',
#     'run_900_1000.csv'
# ]

map_seeds = [i for i in range(1000)]

# def analyze_output_file(filepath):
#     i = int(filepath.split('_')[1])
#     new_filepath = pathlib.Path(f'/proj/patellab/Sheps/output/{filepath}')
#     for chunk in read_csv_with_header(new_filepath, chunksize = 7 * 3 * args.seeds * args.patients):
#         chunk = process_chunks(chunk)
#         single_map_analysis_output(chunk, 
#                             map_number = i,
#                             heatmap_diff = True,
#                             save = True,
#                             output_dir_str = 'results')
#         i += 1

def analyze_parquet(map_num):
    file_name = f'map_{str(map_num).zfill(3)}.parquet'
    df = read_output(pathlib.Path(current_dir) / file_name, save_format = 'parquet')
    return single_map_analysis_output(df, map_number = map_num, heatmap_diff = True, save = True, output_dir_str = '/work/users/p/w/pwlin/output/results', line_errorbars = True) 

def psc_analyze_parquet(map_num):
    file_name = f'map_{str(map_num).zfill(3)}.parquet'
    df = read_output(pathlib.Path(current_dir) / file_name, save_format = 'parquet')
    df = df.loc[df['closest_destination'] != 'CSC', :].copy()
    return single_map_analysis_output(df, map_number = map_num, heatmap_diff = True, save = True, output_dir_str = '/work/users/p/w/pwlin/output/results', line_errorbars = True, additional_file_name='psc') 


if __name__ == '__main__':
    # pool = mp.Pool(10)
    data_calcs_csv_path = pathlib.Path('/work/users/p/w/pwlin/output/all_results.csv')
    psc_calcs_csv_path = pathlib.Path('/work/users/p/w/pwlin/output/psc_results.csv')
    if not data_calcs_csv_path.parent.exists():
        data_calcs_csv_path.parent.mkdir(parents = True)
    elif data_calcs_csv_path.exists():
        data_calcs_csv_path.unlink()

    if not psc_calcs_csv_path.parent.exists():
        psc_calcs_csv_path.parent.mkdir(parents = True)
    elif psc_calcs_csv_path.exists():
        psc_calcs_csv_path.unlink()

    maps_csv_path = pathlib.Path('/work/users/p/w/pwlin/output/maps.csv')
    if maps_csv_path.exists():
        maps_csv_path.unlink()
    with mp.Pool(25) as pool:
        results = pool.map(analyze_parquet, map_seeds)
        psc_results = pool.map(psc_analyze_parquet, map_seeds)
    pd.concat(results, axis = 0).to_csv(data_calcs_csv_path, header = True, index = False, mode = 'w')
    pd.concat(psc_results, axis = 0).to_csv(psc_calcs_csv_path, header = True, index = False, mode = 'w')
    
        # pool.map(analyze_parquet, map_seeds)
