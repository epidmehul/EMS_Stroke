#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --mem=8g
#SBATCH -t 00:30:00
#SBATCH -o /nas/longleaf/home/pwlin/strokes/output.out
#SBATCH -J sens_thresh
#SBATCH --mail-type=end
#SBATCH --mail-user=pwlin@live.unc.edu

module purge
module load python/3.12.2
source ~/strokes_venv/bin/activate
python calculate_sensitivity_time_thresholds.py --path /work/users/p/w/pwlin/output_sens/all_numbers
