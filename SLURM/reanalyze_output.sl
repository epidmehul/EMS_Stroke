#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 50
#SBATCH --mem=40g
#SBATCH -t 00:05:00
#SBATCH -o /nas/longleaf/home/pwlin/strokes/output.out
#SBATCH -J cohort_avgs
#SBATCH --mail-type=end
#SBATCH --mail-user=pwlin@live.unc.edu

module purge
module load python/3.12.4
source ~/strokes_venv/bin/activate
python ../Scripts/reanalyze_simulation_output.py -n 50 -i /proj/patellab/peter/output/parquet_files -o /proj/patellab/peter/output
