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
| `environment.yml`, `requirements.txt`, `installation.sh` | Environment setup. |

## Dependencies  

- Python 3.8+  
- `numpy==1.19.5`  
- `torch==1.10.1`  
- `torchvision==0.11.2`  
- `torchmetrics==0.8.2`  
- `pandas==1.1.5`  
- `canlib` (for communication)  
- `frankX` (instead of `frankapy`)  

Install with:  
```bash
conda env create -f environment.yml
conda activate contact-interpretation
#or
pip install -r requirements.txt
```

## Setup

### Franka Panda (via `frankX`)
1. Power on robot → yellow light.
2. Connect to console (`172.16.0.2` or `192.168.15.33`).
3. Unlock → blue light → activate FCI.
4. Run control PC setup:
```bash
bash src/robotAPI/frankX/bash_scripts/start_control_pc.sh
```

5. Launch glove node for data labeling:
```bash
source /opt/ros/noetic/setup.bash
conda activate frankxenv
python src/dataLabeling/digitalGloveNode.py
```

6. Start control & data saving:
```bash
python src/frankX/main.py
python src/frankX/saveDataNode.py
```
>>>>>>> 	modified:   README.md

### UR5e (via RTDE)

1. Enable remote control on teach pendant.

2. Run data collection:
```bash
python src/urRobot/main_ur10.py
```

## Training


Pipelines for training are in `pipelines/`:


- **Data preprocessing:** `pipelines/data_pipeline.py`
- **Contact detection:** `pipelines/training_pipeline_contact_detection.py`
- **Localization:** `pipelines/training_pipeline_localization.py`


### Example Run
```bash
python pipelines/training_pipeline_contact_detection.py
```

## Deployment


Use scripts under `deployment/` to run trained models on robots.


### Example Run
```bash
python deployment/run_model_franka.py # For Franka Panda
python deployment/run_model_ur5.py # For UR5e
```
