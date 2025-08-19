#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=8g
#SBATCH -t 00:10:00
#SBATCH -o /nas/longleaf/home/pwlin/strokes/output.out
#SBATCH -J run_analyze_combo
#SBATCH --mail-type=end
#SBATCH --mail-user=pwlin@live.unc.edu

module purge
module load python/3.12.4
source ~/strokes_venv/bin/activate
# python ../Scripts/rng_to_config.py -m 0

python ../Scripts/run_analyze_new_simulation.py -s 100 -p 1000 -c ../config_files/test3.yaml -d ../input_data/patients_0.csv -t ../input_data/map_0.csv -o /work/users/p/w/pwlin/new_output
