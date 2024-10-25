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

file_names = [
    'run_0_100.csv',
    'run_100_200.csv',
    'run_200_300.csv',
    'run_300_400.csv',
    'run_400_500.csv',
    'run_500_600.csv',
    'run_600_700.csv',
    'run_700_800.csv',
    'run_800_900.csv',
    'run_900_1000.csv'
]

map_seeds = [i for i in range(1000)]

def analyze_output_file(filepath):
    i = int(filepath.split('_')[1])
    new_filepath = pathlib.Path(f'/proj/patellab/Sheps/output/{filepath}')
    for chunk in read_csv_with_header(new_filepath, chunksize = 7 * 3 * args.seeds * args.patients):
        chunk = process_chunks(chunk)
        single_map_analysis_output(chunk, 
                            map_number = i,
                            heatmap_diff = True,
                            save = True,
                            output_dir_str = 'results')
        i += 1

def analyze_parquet(map_num):
    file_name = f'map_{str(map_num).zfill(3)}.parquet'
    df = read_output(pathlib.Path(current_dir) / file_name, save_format = 'parquet')
    return single_map_analysis_output(df, map_number = map_num, heatmap_diff = True, save = True, output_dir_str = '/work/users/p/w/pwlin/output/results', line_errorbars = True) 

if __name__ == '__main__':
    # pool = mp.Pool(10)
    map_csv_path = pathlib.Path('/work/users/p/w/pwlin/output/maps.csv')
    if not map_csv_path.parent.exists():
        map_csv_path.parent.mkdir(parents = True)
    with mp.Pool(10) as pool:
        i = 0
        for result in pool.imap_unordered(analyze_parquet, map_seeds, chunksize = 100):
            result.to_csv(map_csv_path, header = False, index = False, mode = 'a')
            i += 1
    
        # pool.map(analyze_parquet, map_seeds)
