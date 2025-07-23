#!/bin/bash

#SBATCH -N 1
#SBATCH -n 5
#SBATCH --mem=120g
#SBATCH -t 10:00:00
#SBATCH -o /nas/longleaf/home/pwlin/strokes/output.out
#SBATCH -J run_analyze_combo
#SBATCH --mail-type=end
#SBATCH --mail-user=pwlin@live.unc.edu

module purge
module load python/3.12.4
source ~/strokes_venv/bin/activate
# python ../Scripts/rng_to_config.py -m 0

# python ../Scripts/run_and_analyze_combinations.py -n 5 -c ../config_files/test2.yaml -d ../input_data/patients_0.csv -t ../input_data/map_0.csv -f ../config_files/cohort_nums.csv -o /work/users/p/w/pwlin/output2/parquet_files

python ../Scripts/run_and_analyze_combinations.py -n 5 -c ../config_files/test2.yaml -d ../input_data/patients_0.csv -t ../input_data/map_0.csv -f ../config_files/cohort_nums.csv -o /work/users/p/w/pwlin/output_map0/parquet_files

python ../Scripts/run_and_analyze_combinations.py -n 5 -c ../config_files/test2.yaml -d ../input_data/patients_0.csv -t ../input_data/map_1.csv -f ../config_files/cohort_nums.csv -o /work/users/p/w/pwlin/output_map1/parquet_files

python ../Scripts/run_and_analyze_combinations.py -n 5 -c ../config_files/test2.yaml -d ../input_data/patients_0.csv -t ../input_data/map_2.csv -f ../config_files/cohort_nums.csv -o /work/users/p/w/pwlin/output_map2/parquet_files

python ../Scripts/run_and_analyze_combinations.py -n 5 -c ../config_files/test2.yaml -d ../input_data/patients_0.csv -t ../input_data/map_3.csv -f ../config_files/cohort_nums.csv -o /work/users/p/w/pwlin/output_map3/parquet_files

python ../Scripts/run_and_analyze_combinations.py -n 5 -c ../config_files/test2.yaml -d ../input_data/patients_0.csv -t ../input_data/map_4.csv -f ../config_files/cohort_nums.csv -o /work/users/p/w/pwlin/output_map4/parquet_files

python ../Scripts/run_and_analyze_combinations.py -n 5 -c ../config_files/test2.yaml -d ../input_data/patients_0.csv -t ../input_data/map_5.csv -f ../config_files/cohort_nums.csv -o /work/users/p/w/pwlin/output_map5/parquet_files

python ../Scripts/run_and_analyze_combinations.py -n 5 -c ../config_files/test2.yaml -d ../input_data/patients_0.csv -t ../input_data/map_6.csv -f ../config_files/cohort_nums.csv -o /work/users/p/w/pwlin/output_map6/parquet_files

python ../Scripts/run_and_analyze_combinations.py -n 5 -c ../config_files/test2.yaml -d ../input_data/patients_0.csv -t ../input_data/map_7.csv -f ../config_files/cohort_nums.csv -o /work/users/p/w/pwlin/output_map7/parquet_files

python ../Scripts/visualize_ci_widths.py -f /work/users/p/w/pwlin/output_map0/avg_ci_half_widths.csv
python ../Scripts/visualize_ci_widths.py -f /work/users/p/w/pwlin/output_map1/avg_ci_half_widths.csv
python ../Scripts/visualize_ci_widths.py -f /work/users/p/w/pwlin/output_map2/avg_ci_half_widths.csv
python ../Scripts/visualize_ci_widths.py -f /work/users/p/w/pwlin/output_map3/avg_ci_half_widths.csv
python ../Scripts/visualize_ci_widths.py -f /work/users/p/w/pwlin/output_map4/avg_ci_half_widths.csv
python ../Scripts/visualize_ci_widths.py -f /work/users/p/w/pwlin/output_map5/avg_ci_half_widths.csv
python ../Scripts/visualize_ci_widths.py -f /work/users/p/w/pwlin/output_map6/avg_ci_half_widths.csv
python ../Scripts/visualize_ci_widths.py -f /work/users/p/w/pwlin/output_map7/avg_ci_half_widths.csv