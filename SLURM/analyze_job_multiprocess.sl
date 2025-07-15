#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=4g
#SBATCH -t 00:20:00
#SBATCH -o /nas/longleaf/home/pwlin/strokes/output.out
#SBATCH -J reanalyze_maps
#SBATCH --mail-type=end
#SBATCH --mail-user=pwlin@live.unc.edu

module purge
module load python/3.12.4
source ~/strokes_venv/bin/activate
python ../Scripts/analyze_multiprocess.py -s 40 -p 1000 -c 1
