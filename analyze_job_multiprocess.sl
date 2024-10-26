#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 11
#SBATCH --mem=32g
#SBATCH -t 2:00:00
#SBATCH -o /nas/longleaf/home/pwlin/strokes/output.out
#SBATCH -J analyze_maps
#SBATCH --mail-type=end
#SBATCH --mail-user=pwlin@live.unc.edu

module purge
module load python/3.12.2
source ~/strokes_venv/bin/activate
python analyze_multiprocess.py -s 40 -p 1000
