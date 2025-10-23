import os
import shutil
import argparse
import pandas as pd

from src.utils.io import read_scan
from src.data.preprocessing import preprocess_case
from src.model.load_model import load_model
from src.model.inference import run_inference, clean_up
from src.data.postprocessing import postprocess_case

def main():
    parser = argparse.ArgumentParser(description = "Run SALSA-liver (nnU-Net v2) inference")
    parser.add_argument("--input_csv", required = True, help = "Path to input csv with paths to NIfTI scans")
    # output folder is by default same as input scan adding "./<scan_name>_SALSA"
    parser.add_argument("--model_dir", default = "/nfs/rwork/SALSA/nnUNet_TALES/nnUNet_results", help = "Path to nnU-Net model directory")
    parser.add_argument("--device", default = "cuda", help = "'cuda' or 'cpu'")
    parser.add_argument("--keep_intermediate_files", default = False, help = "True or False: turn to True for debugging purposes")
    parser.add_argument("--recist", default = True, help = "True or False: turn to False if you do not wnat to apply post-processing based on RECIST-criteria")
    args = parser.parse_args()

    if args.input_csv.split('.')[-1] != 'csv':
        raise ValueError("Input csv must have '.csv' extension")
    else:
        df = pd.read_csv(args.input_csv)

        for value in df['PATHS']:
            path_scan = value
            folder_dir = path_scan.rsplit('/',1)[0]
            scan_name = path_scan.split('/')[-1].split('.')[0]
            scan_folder = os.path.join(folder_dir, scan_name + '_SALSA/')
            print('------------------------------------------------------------------------------------------------------------')
            print('Absolute path to the scan:', path_scan)
            print('Folder for temporary files:', scan_folder)
            print('------------------------------------------------------------------------------------------------------------')

            print(f'\t \t \t \t ¡¡¡¡¡¡¡¡ Keeping intermediate files: {args.keep_intermediate_files} !!!!!!!!')

            # A) PRE-PROCESSING
            scan_resampled, liver_postprocessed = preprocess_case(path_scan, scan_folder, save_files = bool(args.keep_intermediate_files))

            # B) INFERENCE
            model_dir = load_model(args.model_dir)
            run_inference(scan_folder, model_dir, args.device)

            # C) POST-PROCESSING
            postprocess_case(path_scan, scan_folder, scan_resampled, liver_postprocessed, save_files = bool(args.keep_intermediate_files), recist = bool(args.recist))

            # D) CLEAN UP!!!!!
            if os.path.exists(os.path.join(folder_dir, scan_name + '_SALSA.nii.gz')):
                if args.keep_intermediate_files == False: # turn to True for debugging
                    print('Removing intermediate files at:', scan_folder)
                    if os.path.exists(scan_folder):
                        shutil.rmtree(scan_folder)

            print('Done with:', path_scan)
            print('------------------------------------------------------------------------------------------------------------')
            print('############################################################################################################')

if __name__ == "__main__":
    main()