# SALSA: System for Automatic liver Lesion Segmentation And detection

This is the code repository of the liver tumor automatic detection and segmentation tool by Dr. Raquel Perez-Lopez and colleagues from the Radiomics group at the Vall d'Hebron Institute of Oncology (VHIO), Barcelona, Spain.

<p align="center">
    <img src=imgs/workflow.png width="700" height="561.97">
</p>




## Requirements
Download or clone this repository as follows, and navigate into the new folder /liver-SALSA:

`git clone https://github.com/radiomicsgroup/liver-SALSA`

You may create a new Python environment, we will use anaconda/miniconda:

`conda create --name SALSA_env1 python=3.8`

`conda activate SALSA_env1`

Install python packages specified in the file requirements.txt from the liver-SALSA folder:

`pip install -r requirements1.txt`

Repeat this same process for a second environment.

Two environments are needed as Total Segmentator (used as a pre-processing step), uses nnunetv1, while the model's has been trained with a newer version nnunetv2. Total Segmentator is being updated to nnunetv2, as soon as its done it will be adapted into this repository too, and only one environment will be needed.


A Docker file is in development for a more swift implementation

## Content

- The folder `codes` contains the pre-processing, inference and post-processing parts of the pipeline: 
    - `SALSA_stepONE.py`: code that contains the functions to load and preprocess the data.
    - `SALSA_stepTWO.py`: code for running our trained model on the preprocessed data and to save and postprocess the final mask.
    - `run_SALSA.py`: code to run the pipeline.

Download the model's weights from Google Drive or Hugging Face and place in the `models` folder


## Usage
To run the pipeline, call `bash run_SALSA.sh`:

Inside you need to modify the input to the pipeline, which can be:
* The path to the image to be segmented in NIfTI format (.nii or .nii.gz). 
* A csv with all the desired paths of the images to be segmented.


The output segmentation file will also be in NIfTI format. The new file will be saved in the same directory at the same level as the image, with the same filename adding '_SALSA' at the end.


## License
Please, see `license.txt`


## Citation
The paper is currently under review, a preprint version of the article is available at http://dx.doi.org/10.2139/ssrn.4890104


If you have questions please contact Dr. Raquel Perez-Lopez (rperez@vhio.net), Maria Balaguer (mbalaguer@vhio.net) or Adrià Marcos (adriamarcos@vhio.net).

To know more about our group, visit us at https://radiomicsgroup.github.io