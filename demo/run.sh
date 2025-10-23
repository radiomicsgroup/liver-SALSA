#!/bin/bash



# for total-segmentator
export nnUNet_preprocessed=/root/.totalsegmentator/nnunet/results
export nnUNet_results=/root/.totalsegmentator/nnunet/results
export nnUNet_raw=/root/.totalsegmentator/nnunet/results

# for nnU-Net
export nnUNet_raw="/nfs/rwork/SALSA/nnUNet_TALES/nnUNet_raw"
export nnUNet_preprocessed="/nfs/rwork/SALSA/nnUNet_TALES/nnUNet_preprocessed"
export nnUNet_results="/nfs/rwork/SALSA/nnUNet_TALES/nnUNet_results"

source activate /nfs/home/clmbalaguer/anaconda3/envs/salsa

cd /nfs/rnas/projects/SALSA-liver/codes/internal_pipeline/GITHUB/
python inference.py --input_csv ./demo/paths_demo.csv
