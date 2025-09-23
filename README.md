# Contact Interpretation System (Localization)

## Overview  
This repository implements **contact detection and localization** for robotic manipulators.  
It supports dataset handling, training detection/localization models, and robot control (Franka Panda via `frankX`, UR5e via RTDE).

## Repository Structure  

| Directory | Purpose |
|-----------|---------|
| `dataset/` | Datasets for source (Franka main) and target robots (Franka MindLab, UR5e). |
| `pipelines/` | Training & evaluation pipelines for contact detection and localization. |
| `src/` | Robot control code, APIs (Franka Panda & UR5e), data labeling, utilities. |
| `deployment/` | Scripts for deploying trained models. |
| `environment.yml`, `requirements.txt` | Environment setup. |

## Dependencies  

Install with:  
```bash
conda env create -f environment.yml
conda activate contact-interpretation
# or
pip install -r requirements.txt
```

## Setup

### Franka Panda (via `frankX`)
1. Power on robot → yellow light.
2. Connect to console (`ROBOT IP`).
3. Unlock → blue light → activate FCI.
4. Run control PC setup
### UR5e (via RTDE)

1. Enable remote control on teach pendant.

## Training

Pipelines for training are in `pipelines/`:

- **Data preprocessing:** `pipelines/data_pipeline.py`
- **Contact detection:** `pipelines/src/training_contact_detection.py`
- **Localization:** `pipelines/src/training_contact_localization.py`
- **BaseLine:** `pipelines/ref25_training_testing.ipynb`


