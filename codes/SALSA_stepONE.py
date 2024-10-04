import os
import sys
import cc3d
import numpy as np
import nrrd
import torch
import shutil

import pandas as pd
import nibabel as nib

from math import ceil
# from monai.data import MetaTensor
from monai.transforms import Spacingd
from scipy.ndimage import binary_dilation
from totalsegmentator.python_api import totalsegmentator


def read_scan(path_scan: str):
    """
    Function that reads a path corresponding to an image, it makes sure the file format is accepted first

    Arguments:
        * path_scan: string or path-like object, absolute path to the scan

    Returns:
        * scan: Nibabel image, corresponding to the original scan
    """
    
    filename = path_scan.split('/')[-1]
    file_extension = filename.split('.')[-1]
    accepted_extensions = ['nii', 'gz']
    
    if file_extension in accepted_extensions:
        if file_extension == 'nii':
            scan = nib.load(path_scan)
        elif file_extension == 'gz':
            if filename[-6:] == 'nii.gz':
                scan = nib.load(path_scan)
    else:
        print('File format not supported (Only NIfTIs allowed: .nii or .nii.gz)')
        
    return scan


def run_TS(path_scan: str, folder: str, scan_filename: str, save_all_segmentations = False):
    """
    Function that applies Total Segmentator to the indicated scan and returns the liver mask. 
    The liver mask is saved as it is needed for the postprocessing ('./scan_filename_SALSA/liver.nii.gz'), 
    all the output from Total Segmentator can be saved too (default == False)

    Arguments:
        * path_scan: string or path-like object, absolute path to the scan
        * folder: string or path-like object, directory where the scan is stored
        * scan_filename: string or path-like object, corresponding to the filename of the scan to be segmented
        * save_all_segmentations: boolean, save ALL the segmentations from Total Segmentator inside a new folder called './scan_filename_SALSA/TS_seg.nii.gz'

    Returns:
        * liver: Nibabel image, corresponding to the liver mask from Total Segmentator
    """
    if os.path.exists('/root/.totalsegmentator/nnunet/results/nnUNet/3d_fullres/') == False:
        shutil.copytree('/nfs/rwork/software/soft_python/TotalSegmentator/pretrained_weights', '/root/.totalsegmentator/nnunet/results/nnUNet/3d_fullres/')
    folder_name = scan_filename + '_SALSA'
    if os.path.exists(os.path.join(folder, folder_name)) == False:
        os.makedirs(os.path.join(folder, folder_name))
    print('----------------------------------Running Total Segmentator----------------------------------')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # run TS without printing
    sys.stdout = open(os.devnull, 'w')
    seg = totalsegmentator(path_scan, os.path.join(folder, folder_name, 'TS_seg.nii.gz'), True, quiet = True, fast = True, device = device)
    sys.stdout = sys.__stdout__
    # remove intermediate stuff
    if save_all_segmentations == False:
        os.remove(os.path.join(folder, folder_name, 'TS_seg.nii.gz'))
    # build the liver mask and save it
    liver = nib.Nifti1Image((seg.get_fdata() == 5).astype(np.uint16), seg.affine)
    path_liver = os.path.join(folder, folder_name, 'liver.nii.gz')
    nib.save(liver, path_liver)
    print(f'----------------------------------Liver mask saved at {path_liver}----------------------------------')
    return liver


def resample_scan_liver(scan, liver, folder: str, scan_filename: str, save_files = False):
    """
    Function that resamples the original scan and the liver mask from Total Segmentator, to 1x1x1 mm^3. 
    It allows you to save the resampled scan and liver mask (default == False)
    
    Arguments:
        * scan:  Nibabel image, corresponding to the original scan
        * liver:  Nibabel image, corresponding to the liver mask from Total Segmentator
        * folder: string or path-like object, directory where the scan is stored
        * scan_filename: string or path-like object, corresponding to the filename of the scan to be segmented
        * save_files: boolean, save the 2 resampled files (default == False) as './scan_filename_SALSA/scan_resampled.nii.gz' and './scan_filename_SALSA/liver_resampled.nii.gz'

    Returns:
        * image: Nibabel image, corresponding to the original scan resampled
        * mask: Nibabel image, corresponding to the liver mask resampled from Total Segmentator
    """
    spx, spy, spz = scan.header.get_zooms()
    transform_spacing = Spacingd(keys = ["scan", "liver"],
                             pixdim = (1.0/spx, 1.0/spy, 1.0/spz),
                             mode = ("bilinear", "nearest"))
    dic = {"scan": torch.tensor(scan.get_fdata()).unsqueeze(0), "liver": torch.tensor(liver.get_fdata()).unsqueeze(0)}
    dic = transform_spacing(dic)
    image, mask = dic['scan'], dic['liver']
    if save_files == True:
        scan_resampled = nib.Nifti1Image(image[0].numpy(), np.eye(4))
        liver_resampled = nib.Nifti1Image(mask[0].numpy(), np.eye(4))
        folder_name = scan_filename + '_SALSA'
        path_scan_resampled = os.path.join(folder, folder_name, 'scan_resampled.nii.gz')
        path_liver_resampled = os.path.join(folder, folder_name, 'liver_resampled.nii.gz')
        nib.save(scan_resampled, path_scan_resampled)
        nib.save(liver_resampled, path_liver_resampled)
        print(f'----------------------------------Scan and liver mask resampled and saved in {path_scan_resampled} and {path_liver_resampled}----------------------------------')
    else:
        print('----------------------------------Scan and liver mask resampled----------------------------------')
    return image, mask


def postprocess_liver(mask, image, folder, scan_filename, save_mask = False):
    """
    Function that postprocesses the resampled liver mask from Total Segmentator, applies dilation,
    connected components and checks for holes. It allows you to save it too (turned off by Default)
    
    Arguments:
        * mask: Nibabel image, corresponding to the liver mask resampled from Total Segmentator
        * image: Nibabel image, corresponding to the original scan resampled
        * folder: string or path-like object, directory where the scan is stored
        * scan_filename: string or path-like object, corresponding to the filename of the scan to be segmented
        * save_mask: boolean, save the postprocessed mask (default == False) as './scan_filename_SALSA/liver_resampled_postprocessed.nii.gz'

    Returns:
        * mask: Nibabel image, corresponding to the postprocessed liver mask resampled
    """
    # a) apply a dilation to the liver mask
    mask = binary_dilation(mask, iterations = 15)
    # b) make sure the liver is composed of only one connected component
    cc = np.expand_dims(cc3d.connected_components(mask[0]),0)
    N = len(np.unique(cc))
    if N>2:
        liver_cc = [np.sum(cc==c) for c in range(1, N)]
        largest_cc = np.where(np.array(liver_cc)==max(liver_cc))[0][0]+1
        mask = (cc==largest_cc).astype(int)
        print('Deleting small liver parts')
    # c) make sure the liver has no holes
    cc = np.expand_dims(cc3d.connected_components(1-mask[0]),0)
    N = len(np.unique(cc))
    if N>2:
        liver_cc = [np.sum(cc==c) for c in range(1, N)]
        largest_cc = np.where(np.array(liver_cc)==max(liver_cc))[0][0]+1
        mask = 1-(cc==largest_cc).astype(np.uint8)
        print('Deleting small holes in the liver')
    if save_mask == True:
        mask_resampled = nib.Nifti1Image(image[0].numpy(), np.eye(4))
        folder_name = scan_filename + '_SALSA'
        path_liver_post = os.path.join(folder, folder_name, 'liver_resampled_postprocessed.nii.gz')
        nib.save(mask_resampled, path_liver_post)
        print(f'----------------------------------Liver mask postprocessed and saved in {path_liver_post}----------------------------------')
    else:
        print('----------------------------------Liver mask postprocessed----------------------------------')
    return mask


def crop_and_padding(image, mask, folder, scan_filename):
    """
    Function that crops the resampled scan to the resampled and postprocessed mask from Total Segmentator.
    This image is saved as './scan_filename_SALSA/scan_to_segment/scan_0000.nii.gz' and will be the input to the model to run inference
    
    Arguments:
        * image: Nibabel image, corresponding to the original scan resampled
        * mask: Nibabel image, corresponding to the liver mask resampled and postprocessed from Total Segmentator
        * folder: string or path-like object, directory where the scan is stored
        * scan_filename: string or path-like object, corresponding to the filename of the scan to be segmented
    """
    # cropping the values
    mask = torch.tensor(mask)
    image[mask==0] = -1024
    # cropping the bounding box
    # tmp_im = torch.tensor(mask)
    tmp_im = mask.clone().detach()
    tmp_dims = []
    to_keep = None
    for k in range(len(tmp_im.shape)):
        tmp_tk = torch.tensor(binary_dilation(torch.sum(tmp_im, dim=[i for i in range(len(tmp_im.shape)) if i!=k]), iterations=10)>0)
        tmp_dims.append(torch.sum(tmp_tk).item())
        for i in range(k): tmp_tk=tmp_tk.unsqueeze(0)
        for i in range(k, len(tmp_im.shape)-1): tmp_tk=tmp_tk.unsqueeze(-1)
        if to_keep is None:
            to_keep = tmp_tk
        else:
            to_keep = to_keep*tmp_tk
    image_cropped = image[to_keep].view(tmp_dims).squeeze(0).numpy()
    scan_cropped = nib.Nifti1Image(image_cropped, affine = np.eye(4))
    # do padding
    padding = [[0, int(ceil(i/64)*64 - i)] for i in scan_cropped.shape[-3:]]
    padding.reverse()
    padding = [a for b in padding for a in b]
    scan_img = torch.nn.functional.pad(torch.tensor(scan_cropped.get_fdata()), pad=padding, mode='constant', value=-1024.).numpy()
    # save input to model!
    scan_padding = nib.Nifti1Image(scan_img, scan_cropped.affine)
    folder_name = scan_filename + '_SALSA'
    if os.path.exists(os.path.join(folder, folder_name, 'scan_to_segment')) == False:
        os.makedirs(os.path.join(folder, folder_name, 'scan_to_segment'))
    scan_padding_path = os.path.join(folder, folder_name, 'scan_to_segment', 'scan_0000.nii.gz')
    nib.save(scan_padding, scan_padding_path)
    print(f'----------------------------------Input to model saved as {scan_padding_path}----------------------------------')
    

def main():
    scan_path = sys.argv[1]
    folder_dir = scan_path.rsplit('/',1)[0]
    scan_name = scan_path.split('/')[-1].split('.')[0]
    file_extension = scan_path.split('/')[-1].split('.')[-1]

    if file_extension == 'csv':
        df = pd.read_csv(scan_path)
        for value in df['PATHS']:
            scan_path = value
            folder_dir = scan_path.rsplit('/',1)[0]
            scan_name = scan_path.split('/')[-1].split('.')[0]

            # A) preprocessing
            scan = read_scan(scan_path)
            liver = run_TS(scan_path, folder_dir, scan_name, save_all_segmentations = False) 
            image, mask = resample_scan_liver(scan, liver, folder_dir, scan_name, save_files = False)
            mask = postprocess_liver(mask, image, folder_dir, scan_name, save_mask = False)
            cropped = crop_and_padding(image, mask, folder_dir, scan_name)

    else:
        # A) preprocessing
        scan = read_scan(scan_path)
        liver = run_TS(scan_path, folder_dir, scan_name, save_all_segmentations = True) 
        image, mask = resample_scan_liver(scan, liver, folder_dir, scan_name, save_files = True)
        mask = postprocess_liver(mask, image, folder_dir, scan_name, save_mask = True)
        cropped = crop_and_padding(image, mask, folder_dir, scan_name)



if __name__ == "__main__":
    main()

