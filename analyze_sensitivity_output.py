from postprocess_simulation_results import *
import pathlib
import multiprocessing as mp

# os.chdir('/proj/patellab/Sheps/output')
# os.chdir('/work/users/p/w/pwlin')

# sim_results = read_output('run_0_100.csv')

parquet_files = pathlib.Path('/work/users/p/w/pwlin/full_output_sens/parquet_files')
map_files = [parquet_files.glob('*.parquet')]

data_calcs_csv_path = pathlib.Path('/work/users/p/w/pwlin/full_output_sens/all_numbers')

if not data_calcs_csv_path.exists():
    data_calcs_csv_path.mkdir(parents = True)

maps_csv_path = pathlib.Path('/work/users/p/w/pwlin/full_output_sens/maps.csv')
if maps_csv_path.exists():
    maps_csv_path.unlink()

def single_map_sensitivitiy_analyses(path):
    df = read_output(path, save_format = 'parquet')

    map_num = int(path.stem.split('_')[-1])
    sens_descriptor = '_'.join(path.stem.split('_')[:-1]).rstrip('_map')

    result = single_map_analysis_output(df, map_number = map_num, heatmap_diff = True, save = True, output_dir_str = '/work/users/p/w/pwlin/full_output_sens/results', line_errorbars = True, additional_file_name = sens_descriptor) 

    result.to_csv(data_calcs_csv_path / (path.stem + '.csv'), header = True, index = False, mode = 'w')

if __name__ == '__main__':
    with mp.Pool(20) as pool:
        pool.map(single_map_sensitivitiy_analyses, map_files)




# for path in map_files:
#     df = read_output(path, save_format = 'parquet')

#     map_num = int(path.stem.split('_')[-1])
#     sens_descriptor = '_'.join(path.stem.split('_')[:-1]).rstrip('_map')

#     result = single_map_analysis_output(df, map_number = map_num, heatmap_diff = True, save = True, output_dir_str = '/work/users/p/w/pwlin/output_sens/results', line_errorbars = True, additional_file_name = sens_descriptor) 

#     result.to_csv(data_calcs_csv_path / (path.stem + '.csv'), header = True, index = False, mode = 'w')
