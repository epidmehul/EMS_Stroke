#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16g
#SBATCH -t 00:10:00
#SBATCH -o /nas/longleaf/home/pwlin/strokes/output.out
#SBATCH -J sampson
#SBATCH --mail-type=end
#SBATCH --mail-user=pwlin@live.unc.edu

module purge
module load python/3.12.4
source ~/strokes_venv/bin/activate

python ../Scripts/run_sampson_simulation.py -s 100 -p 1000 -c ../config_files/sampson_test.yaml -t ../input_data/sampson_nsc_times.csv -o /work/users/p/w/pwlin/sampson_output
