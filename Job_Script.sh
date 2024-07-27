#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --account=an-tr043
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=nirmalkanna_kunasekaran_IPS-%j.out

# add modules
module load python/3.11.5 
module list

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index pandas numpy yfinance matplotlib statsmodels datetime warnings seaborn sklearn itertools dask time

python Python_Script.py

