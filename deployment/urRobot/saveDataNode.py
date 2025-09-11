#!/usr/bin/env python3
"""
By Maryam Rezayat

This code continuously records data from the Franka Panda robot until manually stopped. It prompts the user to enter a tag name for labeling the data.

Files Created:
1. all_data.txt: Contains all data published by the Franka Panda robot.
2. true_label.csv: Contains data of true labels acquired through CANBUS communication.
3. model_result.csv: Presents the model output and data window.

How to Run:
  -open a terminal
        conda activate frankapyenv
        source /opt/ros/noetic/setup.bash
        $HOME/miniconda/envs/frankapyenv/bin/python3 urRobot/saveDataNode.py
"""

## Import required libraries 
import numpy as np
import rospy
from std_msgs.msg import Float64
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
import csv
import datetime
import os

# Set the default PATH

main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'+'/dataset/ur5/raw_data/'
robot_dof = 6
num_features = 4*robot_dof
time_seq = 28
num_columns = time_seq * num_features 
class LogData:
    def __init__(self, PATH: str, folder_name:str) -> None:
        # Initialize ROS node
        rospy.init_node('log_data')
        print('ROS node is initiated!')
        
        # Create a folder for saving data
        self.PATH = PATH + folder_name
        os.makedirs(self.PATH, exist_ok=True)
        print("Directory '%s' created" % self.PATH)


        # Create empty file for saving data
        header = ['Time_sec', 'Time_nsec', 'prediction_duration', 'contact_out', 'collision_out', 'localization_out'] + [f'val_{i}' for i in range(num_columns)]
        self.log_model_result = csv.writer(open(self.PATH + '/model_out.csv', 'w'))
        self.log_model_result.writerow(header)
        self.file_index = csv.writer(open(self.PATH + '/true_label.csv', 'w'))
        self.file_index.writerow(('time_sec', 'time_nsec', 'timestamp', 'DATA0', 'DATA1', 'DATA2', 'DATA3', 'DATA4'))
        print('*** empty text files are created in ', self.PATH, ' ***')

        # Subscribe to relevant ROS topics
        rospy.Subscriber(name="/model_output", data_class=numpy_msg(Floats), callback=self.save_model_output)
        rospy.Subscriber(name="/contactTimeIndex", data_class=numpy_msg(Floats), callback=self.save_contact_index)

    def save_contact_index(self, data):
        # Save contact index data to true_label.csv
        data_row = np.array(data.data)
        self.file_index.writerow(data_row)

    def save_model_output(self, data):
        # Save model output data to model_result.csv
        data_row = np.array(data.data)
        self.log_model_result.writerow(data_row)


if __name__ == "__main__":

    # Prompt the user to enter a tag name for data labeling
    folder_name = input('Enter tag name: ')
    
    # Create an instance of the LogData class
    log_data_instance = LogData(main_path, folder_name)
    
    # Keep the program running to listen for ROS messages
    rospy.spin()
