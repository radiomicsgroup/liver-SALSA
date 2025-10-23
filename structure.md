SALSA-liver/
│
├── README.md
├── license.txt
├── opensource_manifest.txt
├── requirements.txt
├── environment.yml             # *optional (for conda setup)*
│
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── load_model.py       # handles model loading (e.g. torch, keras)
│   │   ├── inference.py        # core inference logic
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocess.py       # normalization, resizing, etc.
│   │   ├── postprocess.py      # thresholding, mask cleaning, etc.
│   │
│   ├── utils/
│   │   ├── io.py               # reading/writing images, nii files, etc.
│   │   ├── visualization.py    # overlay masks, generate figures
│   │   ├── metrics.py          # *optional: dice, IOU*
│
├── inference.py                # single entry point script (CLI or function)
│
├── demo/
│   ├── scan1.nii.gz
│   ├── scan1_SALSA.nii.gz
│   ├── scan1_SALSA/   
│   │   ├── liver.nii.gz
│   │   ├── 
│   │
│   ├── scan2.nii.gz
│   ├── scan2_SALSA.nii.gz
│   ├── scan2_SALSA/
│   │   ├── liver.nii.gz
│   │   ├── 
│   │
    └── inference_example.ipynb