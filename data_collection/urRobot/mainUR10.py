#!/usr/bin/env python3
"""
This version tries to have clear structure


By Maryam Rezayati
# How to run?

1. activate remote control from robot teach pandanent

2. run program

    -open a terminal
        conda activate frankapyenv
        source /opt/ros/noetic/setup.bash
        $HOME/miniconda/envs/frankapyenv/bin/python3 urRobot/mainUR10.py

"""

## import required libraries 
import os
import numpy as np
import pandas as pd
import time

import torch
from torchvision import transforms

import rospy
from std_msgs.msg import Float64
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from threading import Thread
from threading import Event
from import_model import import_lstm_models
from import_model import import_lstm_models_old

from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_control import RTDEControlInterface

main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'

num_features_lstm = 4
#contact_detection_path= main_path +'AIModels/trainedModels/contactDetection/trainedModel_06_30_2023_10:16:53.pth'
contact_detection_path= main_path +'AIModels/trainedModels/contactDetection/trainedModel_01_24_2024_11:18:01.pth'

#collision_detection_path = main_path + 'AIModels/trainedModels/collisionDetection/trainedModel_06_30_2023_09:07:24.pth'
collision_detection_path = main_path + 'AIModels/trainedModels/collisionDetection/trainedModel_01_24_2024_11:12:30.pth'

#localization_path = main_path + 'AIModels/trainedModels/localization/trainedModel_06_30_2023_09:08:08.pth'
localization_path = main_path + 'AIModels/trainedModels/localization/trainedModel_01_24_2024_11:15:06.pth'


window_length = 28
dof = 6
features_num = dof*4

robot_ip = '172.0.0.2'
frequency = 100
# load model

model_contact, labels_map_contact, num_features_lstm_0 = import_lstm_models(PATH=contact_detection_path)
contact_detection_path= main_path +'AIModels/trainedModels/contactDetection/'
model_contact = import_lstm_models_old(PATH=contact_detection_path,num_classes=2, network_type='main', model_name='trainedModel_LSTM_ur.pth')

model_collision, labels_map_collision, num_features_lstm_1 = import_lstm_models(PATH=collision_detection_path)
model_localization, labels_map_localization, num_features_lstm_2 = import_lstm_models(PATH=localization_path)

if num_features_lstm_2==num_features_lstm_1 and num_features_lstm_1==num_features_lstm_0 :
    num_features_lstm = num_features_lstm_0
else:
    print('check num_lstm features number in your model')
    exit()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device.type == "cuda":
	torch.cuda.get_device_name()


model_contact = model_contact.to(device)
model_collision = model_collision.to(device)
model_localization = model_localization.to(device)

transform = transforms.Compose([transforms.ToTensor()])
window = np.zeros([window_length, features_num])



def contact_detection(data_object:RTDEReceive, event: Event):
    global window, publish_output
    
    rate = rospy.Rate(frequency)
    while  not rospy.is_shutdown():    
        try:
            if event.is_set():
                break
            start_time = rospy.get_time()

            q_desired = np.array(data_object.getTargetQ()) 
            q = np.array(data_object.getActualQ())

            dq_desired = np.array(data_object.getTargetQd())
            dq = np.array(data_object.getActualQd())

            i_desired = np.array(data_object.getTargetCurrent())
            i_actual = np.array(data_object.getActualCurrent())

            #TODO: external torque should be calculated
            tau_J = np.array(data_object.getTargetMoment())
            tau_ext = np.array( i_desired - i_actual)
            tau_ext = np.multiply(tau_ext, 0.7)

            e_q = np.array( q_desired - q)
            e_dq = np.array(dq_desired - dq)

            new_row = np.hstack((tau_J, tau_ext, e_q, e_dq))
            new_row = new_row.reshape((1,features_num))

            window = np.append(window[1:,:],new_row, axis=0)

            #change the order for lstm	
            lstmDataWindow = []
            for j in range(dof):
                # tau(t), tau_ext(t), e(t), de(t)

                if num_features_lstm == 4:
                    column_index = [j, j+dof, j+dof*2, j+dof*3 ]
                elif num_features_lstm ==2:
                    column_index = [j+dof*2, j+dof*3 ]
                elif num_features_lstm ==3:
                    column_index = [j+dof, j+dof*2, j+dof*3 ]

                join_data_matix = window[:, column_index]
                lstmDataWindow.append(join_data_matix.reshape((1, num_features_lstm*window_length)))

            lstmDataWindow = np.vstack(lstmDataWindow)

            with torch.no_grad():
                data_input = transform(lstmDataWindow).to(device).float()
                model_out = model_contact(data_input)
                model_out = model_out.detach()
                output = torch.argmax(model_out, dim=1)

            contact = output.cpu().numpy()[0]
            if contact == 1:
                with torch.no_grad():
                    model_out = model_collision(data_input)
                    model_out = model_out.detach()
                    output = torch.argmax(model_out, dim=1)
                    collision = output.cpu().numpy()[0]

                    model_out = model_localization(data_input)
                    model_out = model_out.detach()
                    output = torch.argmax(model_out, dim=1)
                    localization = output.cpu().numpy()[0]
                detection_duration  = rospy.get_time()-start_time
                rospy.loginfo('detection duration: %f, There is a: %s on %s',detection_duration, labels_map_collision[collision], labels_map_localization[localization])

            else:
                detection_duration  = rospy.get_time()-start_time
                rospy.loginfo('detection duration: %f, there is no contact',detection_duration)

            publish_output.publish(lstmDataWindow)

            #TODO: publish all data as a msg
            #rospy.loginfo(lstmDataWindow)
            rate.sleep()
            
        except Exception as e:
            print(e)
            event.set()
            break



def save_data():
	#TODO: save data!
	pass



def main():
    global publish_output
    event = Event()
    rospy.init_node('ur_robot_data')
    publish_output = rospy.Publisher("/showOutput", numpy_msg(Floats), queue_size= 1)

    # create robot controller instance
    robot = RTDEControlInterface(robot_ip)
    data_object = RTDEReceive(robot_ip, frequency)


    detection_thread = Thread(target= contact_detection, args = (data_object,event, ))
    print('waiting for the models to be loaded............ \n')
    detection_thread.start()
    time.sleep(2)

    #move_robot(robot, event)

    #def move_robot(robot:RTDEControlInterface, event: Event):
    velocity = 0.15
    acceleration = 0.05
    blend_1 = 0.0
    blend_2 = 0.02
    blend_3 = 0.0
    path_pose1 = [0.581, -0.233, 0.359, 2.421, -2.137, 0.161, velocity, acceleration, blend_1 ]
    path_pose2 = [0.581, -0.233, 0.225, 2.421, -2.137, 0.161, velocity, acceleration, blend_1 ]
    path_pose3 = [0.581, 0, 0.359, 2.421, -2.137, 0.161, velocity, acceleration, blend_1 ]
    path_pose4 = [0.581, 0, 0.225, 2.421, -2.137, 0.161, velocity, acceleration, blend_1 ]

    path = [path_pose1, path_pose2, path_pose1, path_pose3, path_pose4, path_pose3, path_pose1]

    # Send a linear path with blending in between - (currently uses separate script)
    while  not rospy.is_shutdown():
        try:
            robot.moveL(path)
        except:
            print('hi')
            robot.stopScript()
            event.set()
            break

    robot.stopScript()
    event.set()

    print('Finished :)')


if __name__ == "__main__":
    main()