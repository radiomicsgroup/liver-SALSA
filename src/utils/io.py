import nibabel as nib
import numpy


def read_scan(path_scan: str):
    """
    Function that reads a path corresponding to an image, it makes sure the file format is accepted first

    Arguments:
        * path_scan: string or path-like object, absolute path to the scan

    Returns:
        * scan: Nibabel image, corresponding to the loaded (original) scan or None if no scan was found
    """

    filename = path_scan.split('/')[-1]
    file_extension = filename.split('.')[-1]
    accepted_extensions = ['nii', 'gz']

    if file_extension in accepted_extensions:
        print('Reading scan from ' + path_scan)
        if file_extension == 'nii':
            scan = nib.load(path_scan)
        elif file_extension == 'gz':
            if filename[-6:] == 'nii.gz':
                scan = nib.load(path_scan)
                print('Correctly loaded:', filename)
                return scan
    else:
        print('File format not supported (Only NIfTIs allowed: .nii or .nii.gz)')
        return None



# def save_scan(array, affine, header, path_scan: str):
#     """
#     Function that reads a path corresponding to an image, it makes sure the file format is accepted first
#
#     Arguments:
#         * path_scan: string or path-like object, absolute path to the scan
#
#     Returns:
#         * scan: Nibabel image, corresponding to the loaded (original) scan or None if no scan was found
#     """
#
#     filename = path_scan.split('/')[-1]
#     file_extension = filename.split('.')[-1]
#     accepted_extensions = ['nii', 'gz']
#
#     if file_extension in accepted_extensions:
#         if file_extension == 'nii':
#             scan = nib.load(path_scan)
#         elif file_extension == 'gz':
#             if filename[-6:] == 'nii.gz':
#                 scan = nib.load(path_scan)
#                 return scan
#     else:
#         print('File format not supported (Only NIfTIs allowed: .nii or .nii.gz)')
#         return None


# def niigz_to_segnrrd(path_mask, folder, scan_filename):
#     """
#     Function that converts the output mask from a .nii.gz to .seg.nrrd
#
#     Arguments:
#         * path_mask: string or path-like object, absolute path to the mask
#         * folder: string or path-like object, directory where the scan is stored
#         * scan_filename: string or path-like object, corresponding to the filename of the scan to be segmented
#
#     Returns:
#         * liver: Nibabel image, corresponding to the final segmentation
#     """
#     nifti_image = nib.load(path_mask)
#     # Get the data array from the NIfTI image
#     data = nifti_image.get_fdata()
#
#     # CONNECTED COMPONENTS TO SPLIT INTO LABELMAPS
#     labelmaps = cc3d.connected_components(data, connectivity=26)
#
#     # Get the header and affine information from the NIfTI image
#     affine = nifti_image.affine
#     header = nifti_image.header
#
#     # Extract space directions (rotation and scaling part of affine matrix)
#     space_directions = affine[:3, :3].tolist()
#
#     # Extract space origin (translation part of affine matrix)
#     space_origin = affine[:3, 3].tolist()
#
#     # Create NRRD header dictionary
#     nrrd_header = {
#         'type': 'float',
#         'dimension': data.ndim,
#         'space': 'right-anterior-superior',
#         'sizes': data.shape,
#         'space directions': space_directions,
#         'kinds': ['domain', 'domain', 'domain'],
#         'endian': 'little',
#         'encoding': 'gzip',
#         'space origin': space_origin
#     }
#
#     # Write the NRRD file using pynrrd
#     mask_name2 = scan_filename + '_SALSA.seg.nrrd'
#     path_mask2 = os.path.join(folder, mask_name2)
#     nrrd.write(path_mask2, labelmaps, header=nrrd_header)
#     os.remove(path_mask)
#     print(f'----------------------------------Final mask saved as {path_mask2}----------------------------------')
