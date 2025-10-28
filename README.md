# SALSA: System for Automatic liver Lesion Segmentation And detection

This is the code repository of the liver tumor automatic detection and segmentation tool by Dr. Raquel Perez-Lopez and colleagues from the Radiomics group at the Vall d'Hebron Institute of Oncology (VHIO), Barcelona, Spain.

<p align="center">
    <img src=imgs/workflow.png width="700" height="561.97">
</p>



The results obtained with our pipeline look like this, where the blue contour is the manual ground truth segmentation (not necessary to run SALSA) and the red contour is the automatic prediction by SALSA.


<p align="center">
    <img src=imgs/results.gif>
</p>


### 📢📢📢 **UPDATE (October 2025)** 

In order to obtain SALSA model's weights, please fill out this [form!](https://apps.docusign.com/api/maestro/v1/accounts/a17d1967-477b-4476-b7f3-c8d47b53a934/workflow_definitions/51999945-59bc-43c2-ba4a-2ce689d4f8cf/trigger?hash=ZjEwYWIwOGZmZDk3MWU0ZTg0NzI3MDg4Y2I2ZGVjMzFkMDdmYjI4NTc1ODU4MmM3OTgzN2M3MmIyZmZiNWI2NGNkMGM5ZDI0MjA4ZWIyZmZhZjdlYmJmOGQ2ZDU3OWRkZWE4YmZkYzhiYjMyYWRlYWQ3MWEyNTJjMjkyNjkyZWNhNGNmZWVmODhiMzNhMjMxODNmNjM3MTliOWZmNTBmOWJmMjNhMjk2M2VlNzdlN2M3ZmIxODA3MmJmODliNTVhNjcxMzQzN2E4Mzg3OTFjMDcxMWY0N2I2MDRjNGFkYWVlY2M1Y2Q3OGRlMGM3ODI3M2JlYmRlYzYwZGI2OTAwNA==) (please, use institutionals emails🙏🏼)


## Requirements
Download or clone this repository as follows, and navigate into the new folder /liver-SALSA:

`git clone https://github.com/radiomicsgroup/liver-SALSA`

You may create a new Python environment, we will use anaconda/miniconda:

`conda create --name SALSA_env1 python=3.10`

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
The paper is out!! Read the full article at https://doi.org/10.1016/j.xcrm.2025.102032

This code can be cited as [![DOI:10.5281/zenodo.14644657](http://img.shields.io/badge/DOI-10.5281/zenodo.14644657-0E7FC0.svg)](https://doi.org/10.5281/zenodo.14644657)


If you have questions please contact Dr. Raquel Perez-Lopez (rperez@vhio.net), Maria Balaguer (mbalaguer@vhio.net) or Adrià Marcos (adriamarcos@vhio.net).

To know more about our group, visit us at https://radiomicsgroup.github.io