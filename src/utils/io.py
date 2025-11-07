import nibabel as nib
import numpy


def check_scan(path_scan: str):
    """
    Function that reads a path corresponding to an image, and makes sure the file format is accepted first

    Arguments:
        * path_scan: string or path-like object, absolute path to the scan

    Returns:
        * None: if read == False
    """

    filename = path_scan.split('/')[-1]
    file_extension = filename.split('.')[-1]
    accepted_extensions = ['nii', 'gz']

    if file_extension in accepted_extensions:
        if file_extension == 'nii':
            return True
        elif file_extension == 'gz':
            if filename[-6:] == 'nii.gz':
                return True
            else:
                print('File format not supported (Only NIfTIs allowed: .nii or .nii.gz)')
                return False
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