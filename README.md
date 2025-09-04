# Contact Interpretation System

This repository contains code and documentation for a contact interpretation system, including contact detection and localization functionalities for robotic manipulators.

## Dependencies

numpy==1.19.5
torch==1.10.1
pandas==1.1.5
torchvision==0.11.2
torchmetrics==0.8.2
canlib
## Contact Detection Datasets:

The following datasets are available for contact detection and localization:

* [https://github.com/MindLabZHAW/contactInterpretation/tree/localization/dataset/franka_main](https://github.com/MindLabZHAW/contactInterpretation/tree/localization/dataset/franka_main): Dataset from the source Franka Emika Panda robot used for initial contact detection and localization experiments.
* [https://github.com/MindLabZHAW/contactInterpretation/tree/localization/dataset/franka_mindlab](https://github.com/MindLabZHAW/contactInterpretation/tree/localization/dataset/franka_mindlab): Dataset collected from the target Franka Emika Panda robot (7 degrees of freedom).
* [https://github.com/MindLabZHAW/contactInterpretation/tree/localization/dataset/ur5](https://github.com/MindLabZHAW/contactInterpretation/tree/localization/dataset/ur5): Dataset collected from the target Universal Robots UR5e robot (6 degrees of freedom).

## AI Models:
https://github.com/MindLabZHAW/contactInterpretation/blob/localization/AIModels/data_pipeline.ipynb: This notebook provides the data pipeline to convert raw sensor data into the desired datasets for training and evaluation.
https://github.com/MindLabZHAW/contactInterpretation/blob/localization/AIModels/training_pipeline_contact_detection.ipynb: This notebook contains the code for training and testing contact detection models on both the source and target robots.
https://github.com/MindLabZHAW/contactInterpretation/blob/localization/AIModels/training_pipeline_localization.ipynb: This notebook contains the code for training and testing contact localization models on both the source and target robots.

## Robot APIs:

### Franka Emika Panda

* franka-interface: [https://iamlab-cmu.github.io/franka-interface/](https://iamlab-cmu.github.io/franka-interface/)
* frankapy: [https://iamlab-cmu.github.io/frankapy/](https://iamlab-cmu.github.io/frankapy/)

### Universal Robots

* Universal Robots RTDE: [https://sdurobotics.gitlab.io/ur_rtde/](https://sdurobotics.gitlab.io/ur_rtde/)

## Run robot and contact detection

### Franka Robot:

#### 1st Step: unlock robot

* Turn on the robot (wait until it has a solid yellow light).
* Connect to the robot desk with the ID (172.16.0.2 or 192.168.15.33).
* Unlock the robot.
* The robot light should turn blue.
* Unlock the robot and activate FCI.

#### 2nd Step: run frankapy

Open a terminal and execute:

```bash
conda activate frankapyenv
bash robotAPI/frankapy/bash_scripts/start_control_pc.sh -i localhost
```
#### 3rd Step: run digital glove node
Open another terminal and execute:

open another temrinal

```bash
	source /opt/ros/noetic/setup.bash
	$HOME/miniconda/envs/frankapyenv/bin/python dataLabeling/digitalGloveNode.py
```
#### 4th Step: run robot node
Open another terminal and execute:
```bash
	conda activate frankapyenv
	source /opt/ros/noetic/setup.bash
	source robotAPI/franka-interface/catkin_ws/devel/setup.bash --extend
	source robotAPI/frankapy/catkin_ws/devel/setup.bash --extend
	
	$HOME/miniconda/envs/frankapyenv/bin/python3 frankaRobot/main.py
```
#### 5th Step: run save data node

Open another terminal and execute:
```bash
	conda activate frankapyenv
	source /opt/ros/noetic/setup.bash
	source robotAPI/franka-interface/catkin_ws/devel/setup.bash --extend
	source robotAPI/frankapy/catkin_ws/devel/setup.bash --extend

	$HOME/miniconda/envs/frankapyenv/bin/python3 frankaRobot/saveDataNode.py
```
### To change the publish rate of frankastate:
Edit the following file: 
```bash
sudo nano robotAPI/franka-interface/catkin_ws/src/franka_ros_interface/launch/franka_ros_interface.launch
```

### UR Robot

#### 1st  Step: activate remote control from robot teach pendant
#### 2nd Step: run program
Open a terminal and execute:
```bash
	conda activate frankapyenv
	source /opt/ros/noetic/setup.bash
	$HOME/miniconda/envs/frankapyenv/bin/python3 urRobot/main_ur10.py
```

## Record new motion for Franka robot:

### 1st  Step and 2nd Step like above.

### 3rd Step: run this code

Open a terminal and execute:
````bash
	conda activate frankapyenv
	source /opt/ros/noetic/setup.bash
	source robotAPI/franka-interface/catkin_ws/devel/setup.bash --extend
	source robotAPI/frankapy/catkin_ws/devel/setup.bash --extend
	
	$HOME/miniconda/envs/frankapyenv/bin/python3 frankaRobot/recordNewMotion.py
````
