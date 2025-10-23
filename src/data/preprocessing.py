import os
import sys
import cc3d
import torch
import shutil
import subprocess
import numpy as np
import nibabel as nib


from math import ceil
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    Compose
)
from monai.data import MetaTensor
from scipy.ndimage import binary_closing, binary_dilation, generate_binary_structure, find_objects
from totalsegmentator.python_api import totalsegmentator

def run_total_segmentator_subprocess(path_scan: str, scan_folder: str, save_files: bool):
    """
    Function that applies Total Segmentator to the indicated scan and returns the liver mask.
    The liver mask is saved as it is needed for the postprocessing ('./<scan_name>_SALSA/liver.nii.gz').

    Arguments:
        * path_scan: string or path-like object, absolute path to the scan
        * scan_folder: string or path-like object, directory where the output liver mask will be saved
        * save_files: boolean, to save the scan and liver mask resampled to target spacing

    Output:
        The liver mask from Total Segmentator saved as './<scan_name>_SALSA/liver.nii.gz'
    """

    if os.path.exists(scan_folder) == False:
        os.makedirs(scan_folder)

    print('Running Total Segmentator')
    device = "gpu" if torch.cuda.is_available() else "cpu"
    cmd = ['TotalSegmentator', '-i', path_scan, '-o', scan_folder, '--roi_subset', 'liver', '--device', device]

    subprocess.run(cmd, check = True,
                   stdout = subprocess.DEVNULL,
                   stderr = subprocess.DEVNULL)


    print('Liver mask generated')

    # build the liver mask and save it
    path_liver = os.path.join(scan_folder, 'liver.nii.gz')

    if not os.path.exists(path_liver):
        raise FileNotFoundError("TotalSegmentator did not produce a liver mask.")

    else:
        liver = nib.load(path_liver)
    if save_files == True:
        print(f'Liver mask saved at: {path_liver}')

def resampling(path_scan: str, scan_folder: str, save_files: bool, target_spacing = (1.0, 1.0, 1.0)):
    """
    Function that gets the scan and the Total Segmentator liver mask, to resample them.

    Arguments:
        * path_scan: string or path-like object, absolute path to the scan
        * scan_folder: string or path-like object, directory where the output scan and liver mask resampled will be saved
        * save_files: boolean, to save the scan and liver mask resampled to target spacing
        * target_spacing: tuple, default set to 1x1x1 mm

    Returns:
        * scan_resampled: Nibabel image, corresponding to the resampled scan
        * liver_resampled: Nibabel image, corresponding to the resampled liver mask from Total Segmentator
    """

    path_liver = os.path.join(scan_folder, 'liver.nii.gz')
    data_dict = {"scan": path_scan, "liver": path_liver}

    transforms = Compose([
        LoadImaged(keys = ["scan", "liver"]),
        EnsureChannelFirstd(keys = ["scan", "liver"]),
        Orientationd(keys = ["scan", "liver"], axcodes = "RAS"),
        Spacingd(keys = ["scan", "liver"], pixdim = target_spacing, mode = ("bilinear", "nearest")),
        # ScaleIntensityRanged(keys = ["scan"], a_min = -100, a_max = 400, b_min = 0.0, b_max = 1.0, clip = True)
    ])

    processed = transforms(data_dict)
    scan_tensor, liver_tensor = processed["scan"], processed["liver"]

    scan_resampled = scan_tensor.as_tensor().cpu().numpy()[0]
    liver_resampled = liver_tensor.as_tensor().cpu().numpy()[0]

    scan_resampled_nib = nib.Nifti1Image(scan_resampled, affine = scan_tensor.affine)
    liver_resampled_nib = nib.Nifti1Image(liver_resampled.astype(np.uint8), affine = liver_tensor.affine)

    if save_files == True:
        path_scan_resampled = os.path.join(scan_folder, 'scan_resampled.nii.gz')
        path_liver_resampled = os.path.join(scan_folder, 'liver_resampled.nii.gz')

        nib.save(scan_resampled_nib, path_scan_resampled)
        print(f'Scan resampled and saved at: {path_scan_resampled}')

        nib.save(liver_resampled_nib, path_liver_resampled)
        print(f'Liver mask resampled and saved at: {path_liver_resampled}')
        return scan_resampled_nib, liver_resampled_nib
    else:
        print('Scan and liver mask resampled')
        return scan_resampled_nib, liver_resampled_nib

def postprocess_liver(liver_resampled, scan_folder: str, save_files: bool):
    """
    Function that post-processes the liver mask from Total Segmentator (already resampled), it applies dilation,
    connected components and checks for holes. It allows you to save it too (turned off by Default)

    Arguments:
        * liver_resampled: Nibabel image, corresponding to the liver mask from Total Segmentator resampled
        * scan_folder: string or path-like object, directory where the output liver mask post-processed will be saved
        * save_files: boolean, save the postprocessed mask as './<scan_name>_SALSA/liver_resampled_postprocessed.nii.gz'

    Returns:
        * liver_postprocessed: Nibabel image, corresponding to the postprocessed liver mask resampled
    """

    # define 3D structuring element
    structure = generate_binary_structure(3, 2)  # 3D, connectivity 2
    # 1: binary closing
    closed = binary_closing(liver_resampled.get_fdata(), structure = structure, iterations = 10)
    # 2: dilation to fill gaps
    dilated = binary_dilation(closed, structure = structure, iterations = 2)
    # 3: largest connected component
    labeled_array = cc3d.connected_components(dilated, connectivity = 6, binary_image = True)
    largest_cc = (labeled_array == np.argmax(np.bincount(labeled_array.flat)[1:]) + 1)
    mask = largest_cc.astype(np.uint8)
    liver_postprocessed = nib.Nifti1Image(mask, affine = liver_resampled.affine, header = liver_resampled.header)

    if save_files == True:
        path_liver_postprocessed = os.path.join(scan_folder, 'liver_postprocessed.nii.gz')
        nib.save(liver_postprocessed, path_liver_postprocessed)
        print(f'Liver mask postprocessed and saved at: {path_liver_postprocessed}')
    else:
        print('Liver mask postprocessed')
    return liver_postprocessed

def crop_and_padding(scan_resampled, liver_postprocessed, scan_folder: str, save_files: bool):
    """
    Function that crops the resampled scan to the resampled and postprocessed liver mask from Total Segmentator.
    Then pad to a shape divisible by 64 and saves the result for nnU-Net inference.
    This image is saved as './<scan_name>_SALSA/scan_0000.nii.gz' and will be the input to the model to run inference

    Arguments:
        * scan_resampled: Nibabel image, corresponding to the original scan resampled
        * liver_postprocessed: Nibabel image, corresponding to the liver mask resampled + postprocessed from Total Segmentator
        * scan_folder: string or path-like object, directory where the files are saved
        * save_files: boolean, to save the scan and liver mask resampled to target spacing

    Output:
        Saves ./<scan_name>_SALSA/scan_0000.nii.gz
    """

    # load data
    scan_data = scan_resampled.get_fdata().astype(np.float32)
    liver_data = liver_postprocessed.get_fdata().astype(np.uint8)

    # mask out non-liver regions
    scan_data[liver_data == 0] = -1024.0  # background air

    # compute bounding box of liver region
    bbox = find_objects(liver_data > 0)
    if not bbox or bbox[0] is None:
        raise ValueError("Liver mask appears empty — cannot crop.")

    cropped_scan = scan_data[bbox[0]]

    # pad scan to nearest multiple of 64
    pad_sizes = []
    for dim in cropped_scan.shape:
        total = int(ceil(dim / 64) * 64)
        pad_sizes.append((0, total - dim))

    # pad in z, y, x order (reverse for torch)
    pad_flat = [p for pair in pad_sizes[::-1] for p in pair]
    scan_padded = torch.nn.functional.pad(torch.tensor(cropped_scan),
                                          pad = pad_flat,
                                          mode = "constant",
                                          value = -1024.0).numpy()

    # save padded scan as input to nnU-Net inference
    if os.path.exists(os.path.join(scan_folder, 'scan_to_segment')) == False:
        os.makedirs(os.path.join(scan_folder, 'scan_to_segment'))

    scan_padded_nib = nib.Nifti1Image(scan_padded, affine = scan_resampled.affine)
    path_scan_padded = os.path.join(scan_folder, 'scan_to_segment', "scan_0000.nii.gz")
    nib.save(scan_padded_nib, path_scan_padded)

    if save_files == True:
        print(f'Input to model saved at: {os.path.join(path_scan_padded)}')

def preprocess_case(path_scan: str, scan_folder: str, save_files: bool):
    """
    Function that maps back the preprocessing done on the scan to the inferred mask

    Arguments:
        * path_scan: string or path-like object, absolute path to the scan
        * scan_folder: string or path-like object, directory where the files are saved
        * save_files: boolean, to save intermediate files

    Output:
        * scan_resampled: Nibabel image, corresponding to the original scan resampled
        * liver_postprocessed: Nibabel image, corresponding to the liver mask resampled and postprocessed from Total Segmentator
        (THESE TWO FILES ARE NEEDED FOR POST-PROCESSING IN CASE THEY ARE NOT SAVED, WE NEED THEM IN MEMORY
    """
    print('···············································Pre-processing···············································')
    run_total_segmentator_subprocess(path_scan, scan_folder, bool(save_files))
    scan_resampled, liver_resampled = resampling(path_scan, scan_folder, bool(save_files))
    liver_postprocessed = postprocess_liver(liver_resampled, scan_folder, bool(save_files))
    crop_and_padding(scan_resampled, liver_postprocessed, scan_folder, bool(save_files))
    print(f"···········································Pre-processing done·············································")
    return scan_resampled, liver_postprocessed