#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=1g
#SBATCH -t 0:03:00
#SBATCH -o /nas/longleaf/home/pwlin/strokes/output.out
#SBATCH -J run_maps
#SBATCH --mail-type=end
#SBATCH --mail-user=pwlin@live.unc.edu

module purge
module load python/3.12.2
source ~/strokes_venv/bin/activate
python ../Scripts/run_simulation_multiprocess.py -s 40 -p 1000 -n 1 -c ../config_files/test2.yaml -d ../input_data/test_patient2.csv -t ../input_data/test_dists2.csv
