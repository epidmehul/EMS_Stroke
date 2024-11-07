from stroke_simulation import *
# from postprocess_simulation_results import *
import multiprocessing as mp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seeds', help = 'number of random seeds', type = int, default = 40)
parser.add_argument('-p', '--patients', help = 'number of patients', type = int, default = 1000)
parser.add_argument('-c', '--config', help = 'config file with simulation parameters', type = pathlib.Path, default = None)
args = parser.parse_args()

# os.chdir('/proj/patellab/Sheps/output')
# os.chdir('/work/users/p/w/pwlin/output')

# sim_results = read_output('run_0_100.csv')

map_seeds = [i for i in range(1000)]
output_dir = '/work/users/p/w/pwlin/output_sens/parquet_files'

def run_single_map(map_seed):
    run_map_simulations([map_seed], num_patients = args.patients, num_patient_seeds = args.seeds, save_format = 'parquet', output_dir = output_dir)

if __name__ == '__main__':
    output_dir_path = pathlib.Path(output_dir)
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents = True)
    run_map_simulations([711], num_patients = 1000, num_patient_seeds = 40, save_format = 'parquet', output_dir = output_dir, config = {'patients_lvo_ischemic': 0.141}, additional_file_name='low_lvo')

    run_map_simulations([711], num_patients = 1000, num_patient_seeds = 40, save_format = 'parquet', output_dir = output_dir, config = {'patients_lvo_ischemic': 0.341}, additional_file_name='high_lvo')
