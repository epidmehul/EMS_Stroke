from stroke_simulation import *
from postprocess_simulation_results import *
import argparse
import multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seeds', help = 'number of random seeds', type = int, default = 40)
parser.add_argument('-p', '--patients', help = 'number of patients', type = int, default = 1000)
parser.add_argument('-c', '--config', help = 'config file with simulation parameters', type = pathlib.Path, default = None)
parser.add_argument('-o','--output', help = 'output directory for parquet files', type = pathlib.Path, default = None)
args = parser.parse_args()

# os.chdir('/proj/patellab/Sheps/output')
# os.chdir('/work/users/p/w/pwlin/output')

# sim_results = read_output('run_0_100.csv')

# map_seeds = [711, 31, 244, 126, 701, 671, 93, 414, 984]
# map_seeds = [703]
map_seeds = [i for i in range(1000)]

if args.output is None:
    output_dir = pathlib.Path('/work/users/p/w/pwlin/output_sens/parquet_files')
else:
    output_dir = args.output


def run_single_map(map_seed):
    run_map_simulations([map_seed], num_patients = 1000, num_patient_seeds = 40, save_format = 'parquet', output_dir = output_dir, config = {'patients_lvo_ischemic': 0.141}, additional_file_name='low_lvo')

    run_map_simulations([map_seed], num_patients = 1000, num_patient_seeds = 40, save_format = 'parquet', output_dir = output_dir, config = {'patients_lvo_ischemic': 0.241}, additional_file_name='mid_lvo')

    run_map_simulations([map_seed], num_patients = 1000, num_patient_seeds = 40, save_format = 'parquet', output_dir = output_dir, config = {'patients_lvo_ischemic': 0.341}, additional_file_name='high_lvo')

if __name__ == '__main__':
    output_dir_path = pathlib.Path(output_dir)
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents = True)
    with mp.Pool(10) as pool:
        pool.map(run_single_map, map_seeds)
    

    # for map in map_seeds:
    #     run_map_simulations([map], num_patients = 1000, num_patient_seeds = 40, save_format = 'parquet', output_dir = output_dir, config = {'patients_lvo_ischemic': 0.141}, additional_file_name='low_lvo')

    #     run_map_simulations([map], num_patients = 1000, num_patient_seeds = 40, save_format = 'parquet', output_dir = output_dir, config = {'patients_lvo_ischemic': 0.241}, additional_file_name='mid_lvo')

    #     run_map_simulations([map], num_patients = 1000, num_patient_seeds = 40, save_format = 'parquet', output_dir = output_dir, config = {'patients_lvo_ischemic': 0.341}, additional_file_name='high_lvo')
