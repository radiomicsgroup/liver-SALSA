import os
import cc3d
import torch
import numpy as np
import nibabel as nib

from monai.data import MetaTensor
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    Compose
)
from scipy.ndimage import find_objects
from nibabel.processing import resample_from_to


def undo_crop_and_padding(scan_folder: str, scan_resampled, liver_postprocessed, save_files: bool):
    return SALSA_resampled_nib

def undo_resampling(path_scan: str, scan_folder: str, mask_resampled, target_spacing = (1.0, 1.0, 1.0)):
    print(path_SALSA)

def resample2scan(path_scan: str):
    nib.save(aligned, path_SALSA)


def recist_labeling(path_scan: str, vol_thr_mm3 = 523.6):
    print(f"Post-processed SALSA mask saved at: {path_SALSA}")

def postprocess_case(path_scan: str, scan_folder: str, scan_resampled, liver_postprocessed, save_files: bool, recist: bool):
    """
    Function that maps back the preprocessing done on the scan to the inferred mask
    Also, it applies an optional post-processing step that applies RECIST criteria (based on a hypothetical spherical volume)
    returning segmentation with 1s the RECIST-like lesions and 2s the non-measurable lesions

    Arguments:
        * path_scan: string or path-like object, absolute path to the scan
        * scan_folder: string or path-like object, directory where the files are saved
        * scan_resampled: Nibabel image, corresponding to the scan resampled
        * liver_postprocessed: Nibabel image, corresponding to the liver mask resampled + postprocessed from Total Segmentator
        * save_files: boolean, to save the scan and liver mask resampled to target spacing
        * recist: boolean, whether to apply RECIST criteria post-processing

    Output:
        Final SALSA segmentation mask <scan_name>_SALSA.nii.gz
    """

    print('···············································Post-processing··············································')

    mask_resampled_nib = undo_crop_and_padding(scan_folder, scan_resampled, liver_postprocessed, bool(save_files))
    undo_resampling(path_scan, scan_folder, mask_resampled_nib)
    resample2scan(path_scan)
    if bool(recist) == True:
        recist_labeling(path_scan)

    print('············································Post-processing done············································')