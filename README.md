# SNN-RFI-SUPER
Supervised SNN for RFI detection

Contains:
- Implementation of supervised SNN training for RFI detection
- Optuna hyperparameter optimization code
- Runfiles for training on HPC centre 

All code files are in `src/`.

## Installation
```bash
conda create -n snn-rfi-super python=3.10
conda activate snn-rfi-super
pip install -r src/requirements.txt
```
You may need extra instructions for installing PyTorch with your specific CPU/GPU combination.

### Data Dependencies
The data used in this project is not included in this repository.
You will need to download the HERA dataset from [zenodo](https://zenodo.org/record/6724065) and unzip
them into `/data`.

## Licensing
This code is licensed under the MIT License. See the LICENSE file for details.