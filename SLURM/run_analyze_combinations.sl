#!/bin/bash

#SBATCH -N 1
#SBATCH -n 5
#SBATCH --mem=120g
#SBATCH -t 1:00:00
#SBATCH -o /nas/longleaf/home/pwlin/strokes/output.out
#SBATCH -J run_analyze_combo
#SBATCH --mail-type=end
#SBATCH --mail-user=pwlin@live.unc.edu

module purge
module load python/3.12.4
source ~/strokes_venv/bin/activate
python ../Scripts/run_and_analyze_combinations.py -n 5 -c ../config_files/test2.yaml -d ../input_data/patient_0.csv -t ../input_data/map_1.csv -f ../config_files/cohort_nums.csv -o /work/p/w/pwlin/output_map1/parquet_files'
