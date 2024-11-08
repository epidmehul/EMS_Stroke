from postprocess_simulation_results import *
import pathlib

# os.chdir('/proj/patellab/Sheps/output')
# os.chdir('/work/users/p/w/pwlin')

# sim_results = read_output('run_0_100.csv')

parquet_files = pathlib.Path('/work/users/p/w/pwlin/output_sens/parquet_files')
map_files = parquet_files.glob('*.parquet')

for path in map_files:
    df = read_output(path, save_format = 'parquet')

    map_num = int(path.stem.split('_')[-1])

    data_calcs_csv_path = pathlib.Path('/work/users/p/w/pwlin/output_sens/map_scenario_threshold_seed_results.csv')
    if not data_calcs_csv_path.parent.exists():
        data_calcs_csv_path.parent.mkdir(parents = True)
    elif data_calcs_csv_path.exists():
        data_calcs_csv_path.unlink()

    maps_csv_path = pathlib.Path('/work/users/p/w/pwlin/output_sens/maps.csv')
    if maps_csv_path.exists():
        maps_csv_path.unlink()

    result = single_map_analysis_output(df, map_number = map_num, heatmap_diff = True, save = True, output_dir_str = '/work/users/p/w/pwlin/output_sens/results', line_errorbars = True, additional_file_name = '_'.join(path.stem.split('_')[:-1])) 

    if data_calcs_csv_path.exists():
        result.to_csv(data_calcs_csv_path, header = False, index = False, mode = 'a')
    else:
        result.to_csv(data_calcs_csv_path, header = True, index = False, mode = 'w')
