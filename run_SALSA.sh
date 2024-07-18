#!/bin/bash

export nnUNet_raw="/nfs/rwork/mbalaguer/nnUNet_TALES/nnUNet_raw"
export nnUNet_preprocessed="/nfs/rwork/mbalaguer/nnUNet_TALES/nnUNet_preprocessed"
export nnUNet_results="/nfs/rwork/mbalaguer/nnUNet_TALES/nnUNet_results"



# ABSOLUTE PATH TO THE SCAN TO SEGMENT OR CSV WITH PATHS
# option 1) single scan
# PATH_INPUT=/nfs/rnas/projects/TALES/codes/pipeline/example0/scan_1.nii
# option 2) multiple scans (csv)
PATH_INPUT=/nfs/rnas/projects/TALES/codes/pipeline/csv_paths.csv


source activate /nfs/home/clmbalaguer/anaconda3/envs/total_seg-2
python /nfs/rnas/projects/TALES/codes/pipeline/SALSA_stepONE.py $PATH_INPUT

source activate /nfs/home/clmbalaguer/anaconda3/envs/nnunet
python /nfs/rnas/projects/TALES/codes/pipeline/SALSA_stepTWO.py $PATH_INPUT