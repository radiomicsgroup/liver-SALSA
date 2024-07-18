# SALSA: System for Automatic liver Lesion Segmentation And detection

This is the code repository of the liver tumor automatic detection and segmentation tool by Raquel Perez-Lopez and colleagues from the Vall d'Hebron Institute of Oncology (VHIO), Barcelona, Spain.



**🚧 WORK IN PROGRESS 🚧**



## Requirements
Download or clone this repository as follows, and navigate into the new folder /liver-SALSA:

`git clone https://github.com/radiomicsgroup/liver-SALSA`

You may create a new Python environment, we will use anaconda/miniconda:

`conda create --name myenv python=3.8`

`conda activate myenv`

Install python packages specified in the file requirements.txt from the liver-SALSA folder:

`pip install -r requirements.txt`


A Docker file is in development for a more swift implementation

## Content
- `SALSA_stepONE.py`: code that contains the functions to load and preprocess the data.
- `SALSA_stepTWO.py`: code for running our trained model on the preprocessed data and to save and postprocess the final mask.
- `run_SALSA.sh`: code to run the pipeline.

Download the weights of our model [HERE!](https://drive.google.com/file/d/1-OcpWwafshk51qnlUT-qQ1o_472A_14F/view?usp=sharing)

## Installation



## Usage



## License
Please, see license.txt
