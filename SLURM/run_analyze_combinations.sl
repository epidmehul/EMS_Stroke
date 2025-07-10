#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem=100g
#SBATCH -t 2:00:00
#SBATCH -o /nas/longleaf/home/pwlin/strokes/output.out
#SBATCH -J run_analyze_combo
#SBATCH --mail-type=end
#SBATCH --mail-user=pwlin@live.unc.edu

module purge
module load python/3.12.2
source ~/strokes_venv/bin/activate
python ../Scripts/run_and_analyze_combinations.py -n 10 -c ../config_files/test2.yaml -d ../input_data/test_patient2.csv -t ../input_data/test_dists2.csv -f ../config_files/cohort_nums.csv
