from stroke_simulation import *
import os

# os.chdir('/proj/patellab/Sheps/output')
os.chdir('/work/users/p/w/pwlin/')

# sim_results = read_output('run_0_100.csv')

map_seeds = [i for i in range(100)]
run_map_simulations(map_seeds, save_format = 'parquet', output_dir = 'rewritten_simulation_output/parquet_output')
