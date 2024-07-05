# Contact Interpretation System

numpy==1.19.5

torch==1.10.1

pandas==1.1.5

torchvision==0.11.2

torchmetrics==0.8.2

canlib

## robot APIS: 

### Franka Emika Panda

franka-interface: https://iamlab-cmu.github.io/franka-interface/ 


frankapy: https://iamlab-cmu.github.io/frankapy/ 


### Universal Robots
Universal Robots RTDE: https://sdurobotics.gitlab.io/ur_rtde/

## Run robot and contact detection

### Franka Robot:

#### 1st  Step: unlock robot
	-turn on the robot (wait until it has a solid yellow)
	-connect to the robot desk with the ID (172.16.0.2 or 192.168.15.33)
	-unlock the robot
	-the robot light should be blue
	-unlock the robot and activate FCI

#### 2nd Step: run frankapy

open an terminal

	## conda activate frankapyenv
	conda activate frankapyenv
	bash robotAPI/frankapy/bash_scripts/start_control_pc.sh -i localhost


#### 3nd Step: run digital glove node

open another temrinal

	source /opt/ros/noetic/setup.bash
	$HOME/miniconda/envs/frankapyenv/bin/python dataLabeling/digitalGloveNode.py

#### 4th Step: run robot node

open another terminal 

	conda activate frankapyenv
	source /opt/ros/noetic/setup.bash
	source robotAPI/franka-interface/catkin_ws/devel/setup.bash --extend
	source robotAPI/frankapy/catkin_ws/devel/setup.bash --extend
	
	$HOME/miniconda/envs/frankapyenv/bin/python3 frankaRobot/main.py

#### 5th Step: run save data node

open another terminal

	conda activate frankapyenv
	source /opt/ros/noetic/setup.bash
	source robotAPI/franka-interface/catkin_ws/devel/setup.bash --extend
	source robotAPI/frankapy/catkin_ws/devel/setup.bash --extend

	$HOME/miniconda/envs/frankapyenv/bin/python3 frankaRobot/saveDataNode.py


### to chage publish rate of frankastate go to : 
sudo nano robotAPI/franka-interface/catkin_ws/src/franka_ros_interface/launch/franka_ros_interface.launch


### UR Robot

#### 1st  Step: activate remote control from robot teach pandanent


#### 2nd Step: run program

open a terminal

	conda activate frankaenv
	source /opt/ros/noetic/setup.bash
	$HOME/miniconda/envs/frankaenv/bin/python3 urRobot/main_ur10.py



## record new motion for franka robot:

### 1st  Step: unlock robot
	-turn on the robot (wait until it has a solid yellow)
	-connect to the robot desk with the ID (172.16.0.2 or 192.168.15.33)
	-unlock the robot
	-the robot light should be blue
	-unlock the robot and activate FCI

### 2nd Step: run frankapy

open an terminal

	conda activate frankaenv
	bash robotAPI/frankapy/bash_scripts/start_control_pc.sh -i localhost

### 3nd Step: run this code

open another terminal 

	conda activate frankaenv
	source /opt/ros/noetic/setup.bash
	source robotAPI/franka-interface/catkin_ws/devel/setup.bash --extend
	source robotAPI/frankapy/catkin_ws/devel/setup.bash --extend
	
	$HOME/miniconda/envs/frankaenv/bin/python3 frankaRobot/recordNewMotion.py
