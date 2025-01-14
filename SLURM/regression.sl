#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=1g
#SBATCH -t 00:05:00
#SBATCH -o /nas/longleaf/home/pwlin/strokes/output.out
#SBATCH -J regress
#SBATCH --mail-type=end
#SBATCH --mail-user=pwlin@live.unc.edu

module purge
module load R/4.4.0

Rscript -e 'rmarkdown::render("~/strokes/Scripts/map_regression.Rmd", output_dir = "/work/users/p/w/pwlin/output/", clean = T)'
