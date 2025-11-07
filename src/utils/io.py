import nibabel as nib
import numpy


def check_scan(path_scan: str):
    """
    Function that reads a path corresponding to an image, and makes sure the file format is accepted

    Arguments:
        * path_scan: string or path-like object, absolute path to the scan

    Returns:
        * None: if read == False
    """

    if path_scan.endswith(".nii.gz"):
        return True
    elif path_scan.endswith(".nii"):
        return True
    else:
        print('File format not supported (Only NIfTIs allowed: .nii or .nii.gz)')
        return False



def read_scan(path_scan: str):
    """
    Function that reads a path corresponding to an image, it makes sure the file format is accepted first

    Arguments:
        * path_scan: string or path-like object, absolute path to the scan

    Returns:
        * scan: Nibabel image, corresponding to the loaded (original) scan or None if no scan was found
    """

    filename = path_scan.split('/')[-1]
    
    if check_scan(path_scan):
        scan = nib.load(path_scan)
        print('Correctly loaded:', filename)
        return scan
    else:
        print('File format not supported (Only NIfTIs allowed: .nii or .nii.gz)')
        return None