#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 11
#SBATCH --mem=10g
#SBATCH -t 00:03:00
#SBATCH -o /nas/longleaf/home/pwlin/strokes/output.out
#SBATCH -J sens_maps
#SBATCH --mail-type=end
#SBATCH --mail-user=pwlin@live.unc.edu

module purge
module load python/3.12.2
source ~/strokes_venv/bin/activate
python ../Scripts/regenerate_maps_csv.py --path /work/users/p/w/pwlin/output/results
