import os
import cv2
import cc3d
import torch
import numpy as np
import pandas as pd
import nibabel as nib

from PIL import Image
from monai.transforms import ScaleIntensityRange
from scipy.ndimage import binary_closing, binary_dilation, generate_binary_structure

def generate_gif(input: str):
    if input.endswith('.csv'):
        generate_gif_csv(input)
    elif input.endswith('.nii.gz') or input.endswith('.nii'):
        generate_gif_nifti(input)


def generate_gif_csv(input: str):
    """
    Function to generate GIFs either from the input csv (same as to run SALSA).
    It also has the option to plot the liver mask (in case it has been saved from the running of SALSA)

    Arguments:
    * input: csv, with paths to each scan (only one column named 'PATHS', it can be the same as the one to run SALSA)

    Output:
        The GIF generated saved as './<scan_name>_SALSA.gif'

    """
    # define transform for scan
    transform_pre = ScaleIntensityRange(a_min = -1024,
                                    a_max = 3071,
                                    b_min = 0.0,
                                    b_max = 1.0,
                                    clip = True)
    
    df = pd.read_csv(input)
    for value in df['PATHS']:
        path_scan = value
        folder_dir = path_scan.rsplit('/',1)[0]
        scan_name = path_scan.split('/')[-1].split('.')[0]
        scan_folder = os.path.join(folder_dir, scan_name + '_SALSA/')

        filename = path_scan.replace('.nii.gz', '_SALSA.gif') # output

        # load and preprocess images
        scan = nib.load(path_scan).get_fdata()
        scan = transform_pre(torch.tensor(scan)).numpy()
        path_SALSA = path_scan.replace('.nii.gz', '_SALSA.nii.gz')
        salsa = nib.load(path_SALSA).get_fdata()

        scan, salsa = np.flip(scan, 2), np.flip(salsa, 2)

        if nib.load(path_scan).affine[1, 1] > 0: 
            scan, salsa = np.flip(scan, 1), np.flip(salsa, 1)

        scan, salsa = np.transpose(scan, (1, 0, 2)), np.transpose(salsa, (1, 0, 2))

        if os.path.exists(os.path.join(scan_folder, 'liver.nii.gz')):
            liver = nib.load(os.path.join(scan_folder, 'liver.nii.gz')).get_fdata()
            structure = generate_binary_structure(3, 2)  # 3D, connectivity 2
            # 1: binary closing
            closed = binary_closing(liver, structure = structure, iterations = 10)
            # 2: dilation to fill gaps
            dilated = binary_dilation(closed, structure = structure, iterations = 2)
            # 3: largest connected component
            labeled_array = cc3d.connected_components(dilated, connectivity = 6, binary_image = True)
            largest_cc = (labeled_array == np.argmax(np.bincount(labeled_array.flat)[1:]) + 1)
            liver = largest_cc.astype(np.uint8)

            liver = np.flip(liver, 2)
            if nib.load(path_scan).affine[1, 1] > 0: 
                liver = np.flip(liver, 1)
            liver = np.transpose(liver, (1, 0, 2))
        else:
            liver = None

        create_gif(scan, salsa, liver, filename)


def generate_gif_nifti(input):
    """
    Function to generate GIFs either from just one scan (.nii or .nii.gz).
    It also has the option to plot the liver mask (in case it has been saved from the running of SALSA)

    Arguments:
    * input: str or path-like object, with the paths to the scan we want the gif

    Output:
        The GIF generated saved as './<scan_name>_SALSA.gif'

    """
    path_scan = input
    folder_dir = path_scan.rsplit('/',1)[0]
    scan_name = path_scan.split('/')[-1].split('.')[0]
    scan_folder = os.path.join(folder_dir, scan_name + '_SALSA/')


    # define transform for scan
    transform_pre = ScaleIntensityRange(a_min = -1024,
                                    a_max = 3071,
                                    b_min = 0.0,
                                    b_max = 1.0,
                                    clip = True)

    filename = path_scan.replace('.nii.gz', '_SALSA.gif')

    # load and preprocess images
    scan = nib.load(path_scan).get_fdata()
    scan = transform_pre(torch.tensor(scan)).numpy()
    path_SALSA = path_scan.replace('.nii.gz', '_SALSA.nii.gz')
    salsa = nib.load(path_SALSA).get_fdata()

    scan, salsa = np.flip(scan, 2), np.flip(salsa, 2)

    if nib.load(path_scan).affine[1, 1] > 0: 
        scan, salsa = np.flip(scan, 1), np.flip(salsa, 1)

    scan, salsa = np.transpose(scan, (1, 0, 2)), np.transpose(salsa, (1, 0, 2))

    if os.path.exists(os.path.join(scan_folder, 'liver.nii.gz')):
        liver = nib.load(os.path.join(scan_folder, 'liver.nii.gz')).get_fdata()
        structure = generate_binary_structure(3, 2)  # 3D, connectivity 2
        # 1: binary closing
        closed = binary_closing(liver, structure = structure, iterations = 10)
        # 2: dilation to fill gaps
        dilated = binary_dilation(closed, structure = structure, iterations = 2)
        # 3: largest connected component
        labeled_array = cc3d.connected_components(dilated, connectivity = 6, binary_image = True)
        largest_cc = (labeled_array == np.argmax(np.bincount(labeled_array.flat)[1:]) + 1)
        liver = largest_cc.astype(np.uint8)

        liver = np.flip(liver, 2)
        if nib.load(path_scan).affine[1, 1] > 0: 
            liver = np.flip(liver, 1)
        liver = np.transpose(liver, (1, 0, 2))
    else:
        liver = None

    create_gif(scan, salsa, liver, filename)



def create_gif(scan, salsa, liver, filename):
    """
    Create GIF for each input scan + SALSA segmentation (+ liver mask)

    Arguments:
        * scan: Nibabel image, corresponding to the scan, processed for the gif generation
        * salsa: Nibabel image, corresponding to the SALSA-liver segmentation, processed for the gif generation
        * liver: Nibabel image, corresponding to the liver postprocessed mask, processed for the gif generation or None (if the file does not exist)
        * filename: string or path-like object, where the gif will be stored

    Output:
        The GIF generated saved as './<scan_name>_SALSA.gif'

    """
    # define colors
    red = [200, 0, 0] # SALSA   
    yellow = [255, 255, 0] # liver


    scan = 255*scan
    eq_scan = (scan - 55) / 25 * 255
    eq_scan[eq_scan < 0] = 0
    eq_scan[eq_scan > 255] = 255
    slices = []
    if liver is None:
        for slice in range(eq_scan.shape[-1]):
            s = np.expand_dims(eq_scan[:,:,slice], -1) # scan
            p = np.float32(salsa[:,:,slice] > 0.5) # SALSA prediction

            kernel = np.ones((3, 3), np.uint8) 
            p = cv2.dilate(p, kernel, iterations = 1) - p

            img = np.uint8(np.concatenate((s, s, s), 2))
            img[p>0] = red
            slices.append(Image.fromarray(img))

    else:
        for slice in range(eq_scan.shape[-1]):
            s = np.expand_dims(eq_scan[:,:,slice], -1) # scan
            p = np.float32(salsa[:,:,slice] > 0.5) # SALSA prediction
            l = liver[:,:,slice] # liver mask

            kernel = np.ones((3, 3), np.uint8) 
            p = cv2.dilate(p, kernel, iterations = 1) - p
            l = cv2.dilate(l, kernel, iterations = 1) - l

            img = np.uint8(np.concatenate((s, s, s), 2))
            img[p>0] = red
            img[l>0] = yellow
            slices.append(Image.fromarray(img))

        
    frame_one = slices[0]

    d = 10000/scan.shape[2]
    print(f'saving gif as:', filename)

    frame_one.save(filename, format = "GIF", append_images = slices[1:], save_all = True, duration = d, loop = 0) # duration of each frame in milliseconds