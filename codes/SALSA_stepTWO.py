import os
import sys
import cc3d
import nrrd
import torch
import shutil

import numpy as np
import pandas as pd
import nibabel as nib

from math import ceil
from monai.transforms import Spacing
from scipy.ndimage import binary_dilation

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def run_inference(scan_filename: str, folder: str):
    """
    Function that runs inference on the nnU-Net and saves the corresponding output masks
    
    Arguments:
        * scan_filename: string or path-like object, corresponding to the filename of the original scan to be segmented
        * folder: string or path-like object, directory where the scan is stored / working directory
    """
    nnUNet_results = '/nfs/rwork/mbalaguer/nnUNet_TALES/nnUNet_results'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    
    folder_name = scan_filename + '_SALSA'
    folder_to_segment = os.path.join(folder, folder_name, 'scan_to_segment')
    print('----------------------------------Running inference----------------------------------')
    # 1st low res --------------
    if os.path.exists(os.path.join(folder, folder_name, 'SALSA_segmentations_3dl')) == False:
        os.makedirs(os.path.join(folder, folder_name, 'SALSA_segmentations_3dl'))
        os.makedirs(os.path.join(folder, folder_name, 'SALSA_segmentations_3dc'))

    output_folder_1 =  os.path.join(folder, folder_name, 'SALSA_segmentations_3dl')
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        os.path.join(nnUNet_results, 'Dataset001_TALES/nnUNetTrainer__nnUNetPlans__3d_lowres'),
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )
    predictor.predict_from_files(folder_to_segment, 
                                output_folder_1,
                                save_probabilities=False, overwrite=False,
                                num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
    # 2nd cascade --------------
    output_folder_2 =  os.path.join(folder, folder_name, 'SALSA_segmentations_3dc')
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        os.path.join(nnUNet_results, 'Dataset001_TALES/nnUNetTrainer__nnUNetPlans__3d_cascade_fullres'),
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )
    predictor.predict_from_files(folder_to_segment,
                                output_folder_2,
                                save_probabilities=False, overwrite=False,
                                num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                folder_with_segs_from_prev_stage=output_folder_1, num_parts=1, part_id=0)


def find_mask_corner(mask_tensor):
    # Iterate through each dimension of the tensor
    corner = []
    for dim in range(mask_tensor.dim()):
        # Find the index of the first non-zero element along the current dimension
        non_zero_indices = (mask_tensor != 0).nonzero(as_tuple=True)
        if non_zero_indices[dim].shape[0] > 0:
            corner.append(non_zero_indices[dim][0].item())
        else:
            corner.append(0)
    return tuple(corner)


def adjust_shape_to_match(scan, mask):
    """
    Adjust the shape of the mask tensor to match the shape of the scan tensor.
    If the shape of the mask tensor is smaller, it will be padded with zeros.
    If the shape of the mask tensor is larger, it will be cropped.

    Args:
    - scan: PyTorch tensor, the reference tensor with the desired shape.
    - mask: PyTorch tensor, the tensor to be adjusted to match the shape of scan.

    Returns:
    - adjusted_mask: PyTorch tensor, the mask tensor with adjusted shape.
    """

    # Get the shapes of the scan and mask tensors
    scan_shape = scan.shape
    mask_shape = mask.shape

    # First let's crop if any axis is too big
    mask = mask[:min(scan_shape[0], mask_shape[0]), :min(scan_shape[1], mask_shape[1]), :min(scan_shape[2], mask_shape[2])]

    # Second let's pad the axis that are too short
    padding = [[0, int(scan_shape[i]-mask_shape[i])] for i in range(len(mask_shape))]
    padding.reverse()
    padding = [a for b in padding for a in b]

    mask = torch.nn.functional.pad(mask, pad=padding, mode='constant', value=0)

    return mask


def postprocess_case(case):
    '''
    This function generates the masks in the original space; with the same shape and voxel size from the original scan.
    Input:
        case: a dictionary with three fields:
            * 'scan' the absolute path to the original scan
            * 'mask' the absolute path to the predicted mask to be processed
            * 'liver' the absolute path to the liver mask generated during the preprocessing
    '''
    # Let's load everything
    # The variables with 111 in the name have 1mm3 voxels. The others do not
    mask111 = nib.load(case['mask'])
    scan = nib.load(case['scan'])
    liver = nib.load(case['liver'])

    ## We'll start by getting the area that was croped from the original image; this was done from the liver mask
    # This is an exact copy of the transformation applied to the liver mask in the preprocessing code
    # Reshape the mask to 1x1x1 mm3
    print('----------------------------------Postprocessing mask----------------------------------')
    spx, spy, spz = scan.header.get_zooms()
    transform_spacing = Spacing(
                    pixdim = (1/spx, 1/spy, 1/spz),
                    mode = "nearest")
    liver111 = transform_spacing(torch.tensor(liver.get_fdata()).unsqueeze(0))
    liver111 = binary_dilation(liver111, iterations=15)

    # Make sure the liver is composed of only one connected component
    cc = np.expand_dims(cc3d.connected_components(liver111[0]),0)
    N = len(np.unique(cc))
    if N>2:
        liver_cc = [np.sum(cc==c) for c in range(1, N)]
        largest_cc = np.where(np.array(liver_cc)==max(liver_cc))[0][0]+1
        liver111 = (cc==largest_cc).astype(int)
        print('Deleting small liver parts')

    # Make sure the liver has no holes
    cc = np.expand_dims(cc3d.connected_components(1-liver111[0]),0)
    N = len(np.unique(cc))
    if N>2:
        liver_cc = [np.sum(cc==c) for c in range(1, N)]
        largest_cc = np.where(np.array(liver_cc)==max(liver_cc))[0][0]+1
        liver111 = 1-(cc==largest_cc).astype(np.uint8)
        print('Deleting small holes in the liver')

    # Finally get a mask with the bounding box around the liver
    tmp_im = torch.tensor(liver111)
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

    # Once we have the mask and its dimentsions, find the corner where it starts
    corner = find_mask_corner(to_keep)
    # In the latest step of the preprocessing we padded the croped region to multiples of 64 so the shape of the mask shall suffer the same fate
    mask_dims = [1] + [int(ceil(i/64)*64) for i in tmp_dims[1:]]
    assert mask_dims[1:] == list(mask111.shape), 'There is some mismatch among the shapes of the predicted mask and the target shape extracted from the liver mask.'
    #to_keep[:, corner[1]:corner[1]+tmp_dims[1], corner[2]:corner[2]+tmp_dims[2], corner[3]:corner[3]+tmp_dims[3]] # This shall only contain 1s

    # Let's now place our partial mask into a full mask in the 111 space
    full_mask = torch.zeros(liver111.shape)
    # Beware that the padding applied during the preprocessing may have extended the mask beyond the limit of the scan
    mask_dims[1:] = [min(mask_dims[i], full_mask.shape[i]-corner[i]) for i in range(1, len(mask_dims))]
    full_mask[:, corner[1]:corner[1]+mask_dims[1], corner[2]:corner[2]+mask_dims[2], corner[3]:corner[3]+mask_dims[3]] = torch.tensor(mask111.get_fdata())[:mask_dims[1],:mask_dims[2],:mask_dims[3]]

    # Finally, we reshape the full mask from the 111 space to the original one
    transform_spacing = Spacing(
                    pixdim = (spx, spy, spz),
                    mode = "nearest")
    mask_reshaped = transform_spacing(full_mask).squeeze(0)
    if not  mask_reshaped.shape==scan.shape:
        print(f"There's a mismatch among the reshaped mask shape {mask_reshaped.shape} and the scan shape {scan.shape}.")
        mask_reshaped = adjust_shape_to_match(scan, mask_reshaped)

    return nib.Nifti1Image(mask_reshaped.numpy(), affine=scan.affine)


def niigz_to_segnrrd(path_mask, folder, scan_filename):
    """
    Function that converts the output mask from a .nii.gz to .seg.nrrd

    Arguments:
        * path_mask: string or path-like object, absolute path to the mask
        * folder: string or path-like object, directory where the scan is stored
        * scan_filename: string or path-like object, corresponding to the filename of the scan to be segmented

    Returns:
        * liver: Nibabel image, corresponding to the final segmentation
    """
    nifti_image = nib.load(path_mask)
    # Get the data array from the NIfTI image
    data = nifti_image.get_fdata()

    # CONNECTED COMPONENTS TO SPLIT INTO LABELMAPS
    labelmaps = cc3d.connected_components(data, connectivity=26)

    
    # Get the header and affine information from the NIfTI image
    affine = nifti_image.affine
    header = nifti_image.header
    
    # Extract space directions (rotation and scaling part of affine matrix)
    space_directions = affine[:3, :3].tolist()
    
    # Extract space origin (translation part of affine matrix)
    space_origin = affine[:3, 3].tolist()
    
    # Create NRRD header dictionary
    nrrd_header = {
        'type': 'float',
        'dimension': data.ndim,
        'space': 'right-anterior-superior',
        'sizes': data.shape,
        'space directions': space_directions,
        'kinds': ['domain', 'domain', 'domain'],
        'endian': 'little',
        'encoding': 'gzip',
        'space origin': space_origin
    }

    # Write the NRRD file using pynrrd
    mask_name2 = scan_filename + '_SALSA.seg.nrrd'
    path_mask2 = os.path.join(folder, mask_name2)
    nrrd.write(path_mask2, labelmaps, header=nrrd_header)
    os.remove(path_mask)
    print(f'----------------------------------Final mask saved as {path_mask2}----------------------------------')
 

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

            # B) inference
            run_inference(scan_name, folder_dir)
            

            # C) postprocessing
            folder_name = scan_name + '_SALSA'

            case = {'scan': scan_path, 
                'mask': os.path.join(folder_dir, folder_name, 'SALSA_segmentations_3dc', "scan.nii.gz"), 
                'liver': os.path.join(folder_dir, folder_name, 'liver.nii.gz')}

            mask_reshaped = postprocess_case(case)
            mask_name = scan_name + '_SALSA.nii.gz'
            nib.save(mask_reshaped, os.path.join(folder_dir, mask_name))
            niigz_to_segnrrd(os.path.join(folder_dir, mask_name), folder_dir, scan_name)

            # %%%%%%%%%%%%%%%%%%%%%%% CHANGE IF YOU WANT TO KEEP THEM!!!!!!!!!!!!! %%%%%%%%%%%%%%%%%%%%%%% 
            remove_folder_intermediate_files = True

            if remove_folder_intermediate_files == True:
                if os.path.exists(os.path.join(folder_dir, folder_name)):
                    shutil.rmtree(os.path.join(folder_dir, folder_name))

            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 


    else:
        # B) inference
        run_inference(scan_name, folder_dir)
        
        # C) postprocessing
        folder_name = scan_name + '_SALSA'

        case = {'scan': scan_path, 
            'mask': os.path.join(folder_dir, folder_name, 'SALSA_segmentations_3dc', "scan.nii.gz"), 
            'liver': os.path.join(folder_dir, folder_name, 'liver.nii.gz')}


        mask_reshaped = postprocess_case(case)
        mask_name = scan_name + '_SALSA.nii.gz'
        nib.save(mask_reshaped, os.path.join(folder_dir, mask_name))
        niigz_to_segnrrd(os.path.join(folder_dir, mask_name), folder_dir, scan_name)


        # %%%%%%%%%%%%%%%%%%%%%%% CHANGE IF YOU WANT TO KEEP THEM!!!!!!!!!!!!! %%%%%%%%%%%%%%%%%%%%%%% 
        remove_folder_intermediate_files = True

        if remove_folder_intermediate_files == True:
            if os.path.exists(os.path.join(folder_dir, folder_name)):
                shutil.rmtree(os.path.join(folder_dir, folder_name))

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 



if __name__ == "__main__":
    main()

