#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 25
#SBATCH --mem=10g
#SBATCH -t 01:00:00
#SBATCH -o /nas/longleaf/home/pwlin/strokes/output.out
#SBATCH -J sens_thresh
#SBATCH --mail-type=end
#SBATCH --mail-user=pwlin@live.unc.edu

module purge
module load python/3.12.2
source ~/strokes_venv/bin/activate
python ~/strokes/Scripts/calculate_sensitivity_time_thresholds.py --path /work/users/p/w/pwlin/full_output_sens/all_numbers --output optimal_thresholds
python ~/strokes/Scripts/calculate_sensitivity_time_thresholds.py --path /work/users/p/w/pwlin/full_output_sens/psc_all_numbers --output psc_optimal_thresholds