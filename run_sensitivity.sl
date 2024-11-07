#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=4g
#SBATCH -t 0:05:00
#SBATCH -o /nas/longleaf/home/pwlin/strokes/output.out
#SBATCH -J lvo_senss
#SBATCH --mail-type=end
#SBATCH --mail-user=pwlin@live.unc.edu

module purge
module load python/3.12.2
source ~/strokes_venv/bin/activate
python run_simulation_sensitivity_multiprocess.py -s 40 -p 1000
