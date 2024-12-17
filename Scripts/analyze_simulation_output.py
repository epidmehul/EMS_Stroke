from postprocess_simulation_results import *
import os

# os.chdir('/proj/patellab/Sheps/output')
os.chdir('/work/users/p/w/pwlin')

# sim_results = read_output('run_0_100.csv')

filepath = 'run_0_100.csv'
i = int(filepath.split('_')[1])

for chunk in read_csv_with_header(filepath, chunksize = 7 * 3 * 40 * 1000):
    print(f'analyzing map {i}')
    chunk = process_chunks(chunk)
    single_map_analysis_output(chunk, 
                           map_number = i,
                           heatmap_diff = True,
                           save = True,
                           output_dir_str = 'results_test',
                           threshold = 20)
    i += 1
    if i>5:
        break
