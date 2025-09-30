#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16g
#SBATCH -t 00:10:00
#SBATCH -o /nas/longleaf/home/pwlin/strokes/output.out
#SBATCH -J county_test
#SBATCH --mail-type=end
#SBATCH --mail-user=pwlin@live.unc.edu

module purge
module load python/3.12.4
source ~/strokes_venv/bin/activate

python ../scripts/run_county_test_simulation.py -s 100 -p 1000 -c ../config_files/test4.yaml -d ../input_data/county_test_hexes2.csv -t ../input_data/county_test_all_times.csv -l ../input_data/county_test_hex_hosp_probs.csv -o /work/users/p/w/pwlin/county_output -m 0 -b 1

python ../scripts/run_county_test_simulation.py -s 100 -p 1000 -c ../config_files/test4.yaml -d ../input_data/county_test_hexes2.csv -t ../input_data/county_test_all_times.csv -l ../input_data/county_test_hex_hosp_probs.csv -o /work/users/p/w/pwlin/county_output -m 1 -b 2

python ../scripts/run_county_test_simulation.py -s 100 -p 1000 -c ../config_files/test4.yaml -d ../input_data/county_test_hexes2.csv -t ../input_data/county_test_all_times.csv -l ../input_data/county_test_hex_hosp_probs.csv -o /work/users/p/w/pwlin/county_output -m 2 -b 3