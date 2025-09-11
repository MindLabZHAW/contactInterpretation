#!/usr/bin/env python3
"""
By Maryam Rezayat

This code continuously records data from the Franka Panda robot until manually stopped. It prompts the user to enter a tag name for labeling the data.

Files Created:
1. all_data.txt: Contains all data published by the Franka Panda robot.
2. true_label.csv: Contains data of true labels acquired through CANBUS communication.
3. model_result.csv: Presents the model output and data window.

How to Run:
1. Unlock the Robot:
	-turn on the robot (wait until it has a solid yellow)
	-connect to the robot desk with the ID (172.16.0.2 or 192.168.15.33)
	-unlock the robot
	-the robot light should be blue
	-unlock the robot and activate FCI

2. Connecting to the Robot (Running Frankapy):
	-open an terminal
		conda activate frankapyenv
		bash robotAPI/frankapy/bash_scripts/start_control_pc.sh -i localhost

3. Specify the Output Folder (PATH) - (line 48 in the code).

4. Run the Program:
    - Open another terminal:
        conda activate frankapyenv
        source /opt/ros/noetic/setup.bash
        source robotAPI/franka-interface/catkin_ws/devel/setup.bash --extend
        source robotAPI/frankapy/catkin_ws/devel/setup.bash --extend
        $HOME/miniconda/envs/frankapyenv/bin/python3 frankaRobot/saveDataNode.py
"""

## Import required libraries 
import numpy as np
import rospy
from std_msgs.msg import Float64
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from franka_interface_msgs.msg import RobotState
import csv
import datetime
import os

# Set the default PATH
PATH = 'frankaRobot/DATA/target_robot/k_1/'
PATH = 'dataset/franka_mindlab/raw_data/'

# Prompt the user to enter a tag name for data labeling
folder_name = input('Enter tag name: ')

class LogData:
    def __init__(self, PATH: str) -> None:
        # Initialize ROS node
        rospy.init_node('log_data')
        print('ROS node is initiated!')
        
        # Create a folder for saving data
        self.PATH = PATH + folder_name
        os.makedirs(self.PATH)
        print("Directory '%s' created" % self.PATH)

        # Create empty files for saving data
        self.file_all = open(self.PATH + '/all_data.txt', 'w')
        self.file_index = csv.writer(open(self.PATH + '/true_label.csv', 'w'))
        self.file_index.writerow(('time_sec', 'time_nsec', 'timestamp', 'DATA0', 'DATA1', 'DATA2', 'DATA3', 'DATA4'))

        self.log_model_result = csv.writer(open(self.PATH + '/model_result.csv', 'w'))
        self.log_model_result.writerow(('Time_sec', 'Time_nsec', 'prediction_duration', 'contact_out', 'collision_out', 'localization_out'))
        print('*** Four empty text files are created in ', self.PATH, ' ***')

        # Subscribe to relevant ROS topics
        rospy.Subscriber(name="/model_output", data_class=numpy_msg(Floats), callback=self.save_model_output)
        rospy.Subscriber(name="/robot_state_publisher_node_1/robot_state", data_class=RobotState, callback=self.save_robot_state)
        rospy.Subscriber(name="/contactTimeIndex", data_class=numpy_msg(Floats), callback=self.save_contact_index)

    def save_contact_index(self, data):
        # Save contact index data to true_label.csv
        data_row = np.array(data.data)
        self.file_index.writerow(data_row)

    def save_model_output(self, data):
        # Save model output data to model_result.csv
        data_row = np.array(data.data)
        self.log_model_result.writerow(data_row)

    def save_robot_state(self, data):
        # Save robot state data to all_data.txt
        self.file_all.write(str(data))

if __name__ == "__main__":
    # Create an instance of the LogData class
    log_data_instance = LogData(PATH)
    
    # Keep the program running to listen for ROS messages
    rospy.spin()
