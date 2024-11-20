#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 11
#SBATCH --mem=10g
#SBATCH -t 02:00:00
#SBATCH -o /nas/longleaf/home/pwlin/strokes/output.out
#SBATCH -J full_sens_run
#SBATCH --mail-type=end
#SBATCH --mail-user=pwlin@live.unc.edu

module purge
module load python/3.12.2
source ~/strokes_venv/bin/activate
python run_simulation_sensitivity_multiprocess.py -s 40 -p 1000 -o /work/users/p/w/pwlin/full_output_sens/parquet_files
