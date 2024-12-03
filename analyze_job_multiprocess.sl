#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 25
#SBATCH --mem=50g
#SBATCH -t 3:00:00
#SBATCH -o /nas/longleaf/home/pwlin/strokes/output.out
#SBATCH -J reanalyze_maps
#SBATCH --mail-type=end
#SBATCH --mail-user=pwlin@live.unc.edu

module purge
module load python/3.12.2
source ~/strokes_venv/bin/activate
python analyze_multiprocess.py -s 40 -p 1000
