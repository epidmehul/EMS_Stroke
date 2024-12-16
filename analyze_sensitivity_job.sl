#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 26
#SBATCH --mem=36g
#SBATCH -t 09:00:00
#SBATCH -o /nas/longleaf/home/pwlin/strokes/output.out
#SBATCH -J full_sens
#SBATCH --mail-type=end
#SBATCH --mail-user=pwlin@live.unc.edu

module purge
module load python/3.12.2
source ~/strokes_venv/bin/activate
python analyze_sensitivity_output.py
