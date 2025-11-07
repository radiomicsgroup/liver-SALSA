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
    """
    Function that undoes the cropping (+padding) of the predicted segmentation mask inferred by SALSA.

    Arguments:
        * scan_folder: string or path-like object, directory where the files are saved
        * scan_resampled: Nibabel image, corresponding to the scan resampled
        * liver_postprocessed: Nibabel image, corresponding to the liver mask resampled + postprocessed from Total Segmentator
        * save_files: boolean, to save the predicted segmentation mask resampled to target spacing

    Returns:
        * mask_SALSA_resampled: Nibabel image, corresponding to the predicted segmentation mask inferred by SALSA resampled
    """

    # recompute the bounding box used during cropping
    liver_data = liver_postprocessed.get_fdata().astype(np.uint8)

    bbox = find_objects(liver_data > 0)[0] # tuple of slices

    shape_cropped = liver_data[bbox].shape

    # load mask predicted cropped and remove padding
    SALSA_padded = nib.load(os.path.join(scan_folder, 'cropped', "mask_0000.nii.gz"))
    SALSA_padded_data = SALSA_padded.get_fdata()

    SALSA_cropped = SALSA_padded_data[
        :shape_cropped[0],
        :shape_cropped[1],
        :shape_cropped[2]]

    # insert back into original scan shape
    SALSA_resampled = np.zeros(scan_resampled.shape, dtype = SALSA_cropped.dtype)
    SALSA_resampled[bbox] = SALSA_cropped

    # save
    SALSA_resampled_nib = nib.Nifti1Image(SALSA_resampled.astype(np.uint8), affine = scan_resampled.affine)
    nib.save(SALSA_resampled_nib, os.path.join(scan_folder, 'mask_resampled.nii.gz'))
    if save_files == True:
        print(f"Mask resampled saved at: {os.path.join(scan_folder, 'mask_resampled.nii.gz')}")
    return SALSA_resampled_nib

def undo_resampling(path_scan: str, scan_folder: str, mask_resampled, target_spacing = (1.0, 1.0, 1.0)):
    """
    Function that undoes the resampling to the target spacing of the predicted segmentation mask inferred by SALSA.

    Arguments:
        * path_scan: string or path-like object, absolute path to the scan
        * scan_folder: string or path-like object, directory where the intermediate files are saved
        * scan_resampled: Nibabel image, corresponding to the scan resampled
        * target_spacing: tuple, default set to 1x1x1
        * mask_resampled: Nibabel image, corresponding to the predicted segmentation mask not cropped or padded

    Output:
        Saves ./<scan_name>_SALSA.nii.gz
    """
    path_mask_resampled = os.path.join(scan_folder, 'mask_resampled.nii.gz')

    # load original scan and resampled mask
    scan_orig = nib.load(path_scan)

    data_dict = {"scan": path_scan, "mask": path_mask_resampled}

    transforms = Compose([
        LoadImaged(keys = ["scan", "mask"]),
        EnsureChannelFirstd(keys = ["scan", "mask"]),
        Orientationd(keys = ["scan", "mask"], axcodes = "RAS"),
        Spacingd(keys = ["mask"], pixdim = scan_orig.header.get_zooms(), mode = "nearest"),
    ])

    processed = transforms(data_dict)
    scan_tensor = processed["scan"]
    mask_tensor = processed["mask"]

    mask_original = mask_tensor.as_tensor().cpu().numpy()[0]

    mask_original_nib = nib.Nifti1Image(mask_original.astype(np.uint8), affine = scan_tensor.meta["affine"])

    path_SALSA = path_scan.replace('.nii.gz', '_SALSA.nii.gz')
    nib.save(mask_original_nib, path_SALSA)
    print(f"**********************************************SALSA mask saved*********************************************")
    print(path_SALSA)

def resample2scan(path_scan: str):
    """
    Function that ensures: same shape, same affine, same orientation in the predcited segmentation mask

    Arguments:
        * path_scan: string or path-like object, absolute path to the scan

    Output:
        Saves ./<scan_name>_SALSA.nii.gz
    """
    path_SALSA = path_scan.replace('.nii.gz', '_SALSA.nii.gz')
    scan = nib.load(path_scan)
    mask = nib.load(path_SALSA)
    aligned = resample_from_to(mask, scan)
    nib.save(aligned, path_SALSA)


def recist_labeling(path_scan: str, vol_thr_mm3 = 523.6):
    """
    Function that classifies lesions in a binary mask by physical volume using connected components.
    Based on RECIST-criteria: we are classifying into: 1s (measurable disease) and 2s (non-measurable disease)

    Arguments:
        * path_scan: string or path-like object, absolute path to the scan
        * vol_thr_mm3: integer, volume threshold in mm^3 for the RECIST criteria based on a 10mm diameter sphere (4/3*pi*5^3) = 523.5987755982989

    Output:
        * mask_SALSA_recist: Nibabel image, corresponding to the predicted segmentation mask inferred by SALSA resampled with 1s (measurable disease) and 2s (non-measurable disease)
    """

    path_SALSA = path_scan.replace('.nii.gz', '_SALSA.nii.gz')
    mask = nib.load(path_SALSA)
    mask_data = mask.get_fdata().astype(np.uint8)

    voxel_spacing = np.abs(mask.header.get_zooms()[:3])  # (x, y, z) spacing
    voxel_volume = np.prod(voxel_spacing)

    labels_out, num_labels = cc3d.connected_components(mask_data, connectivity = 26, return_N = True)
    print(f"SALSA has predicted {num_labels} lesions")

    voxel_counts = np.bincount(labels_out.ravel())

    classified_mask = np.zeros_like(mask_data, dtype=np.uint8)

    for i in range(1, num_labels + 1):
        lesion_volume = voxel_counts[i] * voxel_volume  # mm³
        if lesion_volume < vol_thr_mm3:
            classified_mask[labels_out == i] = 2  # small lesion
        else:
            classified_mask[labels_out == i] = 1  # large lesion

    # print summary
    n_small = np.sum(voxel_counts[1:] * voxel_volume < vol_thr_mm3)
    n_large = np.sum(voxel_counts[1:] * voxel_volume >= vol_thr_mm3)
    print(f"Out of the {num_labels} predicted lesions: {n_large} are measurable by RECIST criteria {n_small} are considered non-measurable disease")

    classified_img = nib.Nifti1Image(classified_mask, affine = mask.affine)
    nib.save(classified_img, path_SALSA)

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