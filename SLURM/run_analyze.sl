#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=72g
#SBATCH -t 3:00:00
#SBATCH -o /nas/longleaf/home/pwlin/strokes/output.out
#SBATCH -J run_analyze
#SBATCH --mail-type=end
#SBATCH --mail-user=pwlin@live.unc.edu

module purge
module load python/3.12.4
source ~/strokes_venv/bin/activate
python ../scripts/run_and_analyze.py -s 100 -p 1000 -n 1 -c ../config_files/test2.yaml -d ../input_data/test_patient2.csv -t ../input_data/test_dists2.csv -m 0
python ../scripts/run_and_analyze.py -s 100 -p 5000 -n 1 -c ../config_files/test2.yaml -d ../input_data/test_patient2.csv -t ../input_data/test_dists2.csv -m 1
python ../scripts/run_and_analyze.py -s 100 -p 10000 -n 1 -c ../config_files/test2.yaml -d ../input_data/test_patient2.csv -t ../input_data/test_dists2.csv -m 2
python ../scripts/run_and_analyze.py -s 500 -p 1000 -n 1 -c ../config_files/test2.yaml -d ../input_data/test_patient2.csv -t ../input_data/test_dists2.csv -m 3
python ../scripts/run_and_analyze.py -s 500 -p 5000 -n 1 -c ../config_files/test2.yaml -d ../input_data/test_patient2.csv -t ../input_data/test_dists2.csv -m 4
python ../scripts/run_and_analyze.py -s 500 -p 10000 -n 1 -c ../config_files/test2.yaml -d ../input_data/test_patient2.csv -t ../input_data/test_dists2.csv -m 5
