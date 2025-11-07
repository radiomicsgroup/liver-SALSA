import os
import shutil

import torch
import warnings

warnings.filterwarnings("ignore",
                        message = "Detected old nnU-Net plans format.*") # I know I have nnU-Net old plans
# this is only to suppress unnecessary prints, as we are only running inference
os.environ["nnUNet_raw"] = "/tmp"
os.environ["nnUNet_preprocessed"] = "/tmp"
os.environ["nnUNet_results"] = "/tmp"

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def run_inference(scan_folder: str, model_dir: str, device: str, clean: bool = True):
    """
    Function that runs inference on the nnU-Net and saves the corresponding output mask.
    As it is a 3D cascade configuration, it needs two steps: low-res and full-res.

    Arguments:
        * scan_folder: string or path-like object, directory where the segmentation mask will be saved
        * model_dir: (default = "./src/model/model_weights/") string or path-like object, directory where the nnU-Net model is stored
        * device: str (default = 'cuda'), which device to run inference on
        * clean: bool (default = True), to remove the intermediate low-res mask after inference

    Output:
        Saves ./<scan_name>_SALSA/mask_0000.nii.gz
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size = 0.5,
        use_gaussian = True,
        use_mirroring = True,
        perform_everything_on_device = True,
        device = device,
        verbose = False,
        verbose_preprocessing = False,
        allow_tqdm = False
    )

    scan_to_segment = os.path.join(scan_folder, 'scan_to_segment')
    print('············································Running inference··············································')

    # 1) run low res configuration as a first step
    if os.path.exists(os.path.join(scan_folder, 'SALSA_segmentations_3dl')) == False:
        os.makedirs(os.path.join(scan_folder, 'SALSA_segmentations_3dl'))

    output_folder_1 = os.path.join(scan_folder, 'SALSA_segmentations_3dl')


    print(f"·····Running nnU-Net 3D low-res model·····")

    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        os.path.join(model_dir, 'nnUNetTrainer__nnUNetPlans__3d_lowres'),
        use_folds = (0,),
        checkpoint_name = 'checkpoint_final.pth',
    )
    predictor.predict_from_files(scan_to_segment,
                                 output_folder_1,
                                 save_probabilities = False,
                                 overwrite = False,
                                 num_processes_preprocessing = 2,
                                 num_processes_segmentation_export = 2,
                                 folder_with_segs_from_prev_stage = None,
                                 num_parts = 1,
                                 part_id = 0)


    # 2) next run the cascade configuration as a final step
    if os.path.exists(os.path.join(scan_folder, 'SALSA_segmentations_3dc')) == False:
        os.makedirs(os.path.join(scan_folder, 'SALSA_segmentations_3dc'))

    output_folder_2 = os.path.join(scan_folder, 'SALSA_segmentations_3dc')
    print(f"···········Low-res output saved···········")

    print(f"·····Running nnU-Net 3D cascade model·····")

    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        os.path.join(model_dir, 'nnUNetTrainer__nnUNetPlans__3d_cascade_fullres'),
        use_folds = (0,),
        checkpoint_name = 'checkpoint_final.pth',
    )
    predictor.predict_from_files(scan_to_segment,
                                 output_folder_2,
                                 save_probabilities = False,
                                 overwrite = False,
                                 num_processes_preprocessing = 2,
                                 num_processes_segmentation_export = 2,
                                 folder_with_segs_from_prev_stage = output_folder_1,
                                 num_parts = 1,
                                 part_id = 0)

    print(f"············SALSA output saved···········")
    print('················································Inference done··············································')



    if clean == True:
        print("Cleaning up folders")
        clean_up(scan_folder)
    else:
        print(f"SALSA output saved at: {(os.path.join(scan_folder, 'cropped', 'mask_0000.nii.gz'))}")


def clean_up(scan_folder: str):
    """
    Function that cleans up the nnU-Net inference results folder

    Arguments:
        * scan_folder: string or path-like object, directory where the intermediate files are saved

    Output:
        A cleaned up folder :) (being tidy is important in life)
    """

    os.rename(os.path.join(scan_folder, 'SALSA_segmentations_3dc', 'scan.nii.gz'),
              os.path.join(scan_folder, 'SALSA_segmentations_3dc', 'mask_0000.nii.gz'))

    shutil.move(os.path.join(scan_folder, 'SALSA_segmentations_3dc', 'mask_0000.nii.gz'),
                os.path.join(scan_folder, 'scan_to_segment', 'mask_0000.nii.gz'))

    shutil.rmtree(os.path.join(scan_folder, 'SALSA_segmentations_3dl'))
    shutil.rmtree(os.path.join(scan_folder, 'SALSA_segmentations_3dc'))

    if os.path.exists(os.path.join(scan_folder, 'cropped')):
        shutil.rmtree(os.path.join(scan_folder, 'cropped'))

    os.rename(os.path.join(scan_folder, 'scan_to_segment'),
              os.path.join(scan_folder, 'cropped'))


