#!/bin/bash

# ABSOLUTE PATH TO THE SCAN TO SEGMENT OR CSV WITH PATHS
# option 1) single scan
# PATH_INPUT=/path_to/scan.nii

# option 2) multiple scans (csv)
PATH_INPUT=/path_to/csv.csv


source activate .../envs/SALSA_env1
python ./SALSA_stepONE.py $PATH_INPUT

source activate .../envs/SALSA_env2
python ./SALSA_stepTWO.py $PATH_INPUT