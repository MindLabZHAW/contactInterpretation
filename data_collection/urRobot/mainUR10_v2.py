import os
import numpy as np
import pandas as pd
import time
import signal
import torch
from torchvision import transforms

import rospy
from std_msgs.msg import Float64
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from threading import Thread, Event
from importModel import import_lstm_models

from rtde_receive import RTDEReceiveInterface as RTDEReceive
#from rtde_control import RTDEControlInterface

robot_ip = '192.168.125.5'
frequency = 200

main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'
joints_data_path = main_path + 'urRobot/robotMotionPoints/UR5_06_10_2024_10:00:22.csv'
data_path = main_path + '/urRobot/DATA/UR10/robot_data/'

contact_detection_path = main_path + 'AIModels/trainedModels/contactDetection/trainedModel_01_24_2024_11:18:01.pth'
collision_detection_path = main_path + 'AIModels/trainedModels/collisionDetection/trainedModel_01_24_2024_11:12:30.pth'
localization_path = main_path + 'AIModels/trainedModels/localization/trainedModel_01_24_2024_11:15:06.pth'

window_length = 28
dof = 6
features_num = dof * 4

k = [0.75, 0.73, 0.9, 0.89, 0.6, 0.65]
#k=0.7

model_contact, labels_map_contact, num_features_lstm_0 = import_lstm_models(PATH=contact_detection_path)
model_collision, labels_map_collision, num_features_lstm_1 = import_lstm_models(PATH=collision_detection_path)
model_localization, labels_map_localization, num_features_lstm_2 = import_lstm_models(PATH=localization_path)

if num_features_lstm_2 == num_features_lstm_1 and num_features_lstm_1 == num_features_lstm_0:
    num_features_lstm = num_features_lstm_0
else:
    print('Check num_lstm features number in your model')
    exit()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_contact = model_contact.to(device)
model_collision = model_collision.to(device)
model_localization = model_localization.to(device)

transform = transforms.Compose([transforms.ToTensor()])
window = np.zeros([window_length, features_num])

model_msg = Floats()

def check_rtde_connection(rtde_object: RTDEReceive, event: Event):
    while not event.is_set():
        if not rtde_object.isConnected():
            print("RTDE connection lost. Attempting to reconnect...")
            rtde_object.reconnect()
            event.set()
        time.sleep(1)  # Adjust sleep duration as needed

def contact_detection(data_object: RTDEReceive, event: Event):
    global window, model_pub, big_time_digits, connection_thread
    collision = -1
    localization = -1
    # Start a thread to monitor RTDE connection
    #connection_thread = Thread(target=check_rtde_connection, args=(data_object, event))
    #connection_thread.start()
    
    rate = rospy.Rate(frequency)
    while not rospy.is_shutdown() and not event.is_set():
        start_time = rospy.get_time()

        try:
            q_desired = np.array(data_object.getTargetQ())
            q = np.array(data_object.getActualQ())

            dq_desired = np.array(data_object.getTargetQd())
            dq = np.array(data_object.getActualQd())

            i_desired = np.array(data_object.getTargetCurrent())
            i_actual = np.array(data_object.getActualCurrent())

            tau_J = np.array(data_object.getTargetMoment())
            tau_ext = np.array(i_desired - i_actual)
            tau_ext = np.multiply(tau_ext, k)

            e_q = np.array(q_desired - q)
            e_dq = np.array(dq_desired - dq)

            new_row = np.hstack((tau_J, tau_ext, e_q, e_dq))
            new_row = new_row.reshape((1, features_num))

            window = np.append(window[1:, :], new_row, axis=0)

            lstmDataWindow = []
            for j in range(dof):
                if num_features_lstm == 4:
                    column_index = [j, j + dof, j + dof * 2, j + dof * 3]
                elif num_features_lstm == 2:
                    column_index = [j + dof * 2, j + dof * 3]
                elif num_features_lstm == 3:
                    column_index = [j + dof, j + dof * 2, j + dof * 3]

                join_data_matrix = window[:, column_index]
                lstmDataWindow.append(join_data_matrix.reshape((1, num_features_lstm * window_length)))

            lstmDataWindow = np.vstack(lstmDataWindow)

            with torch.no_grad():
                data_input = transform(lstmDataWindow).to(device).float()
                model_out = model_contact(data_input)
                model_out = model_out.detach()
                output = torch.argmax(model_out, dim=1)

            contact = output.cpu().numpy()[0]
            if contact == 1:
                '''with torch.no_grad():
                    model_out = model_collision(data_input)
                    model_out = model_out.detach()
                    output = torch.argmax(model_out, dim=1)
                    collision = output.cpu().numpy()[0]

                    model_out = model_localization(data_input)
                    model_out = model_out.detach()
                    output = torch.argmax(model_out, dim=1)
                    localization = output.cpu().numpy()[0]'''

                detection_duration = rospy.get_time() - start_time
                rospy.loginfo('Detection duration: %f, There is a contact', detection_duration)
            else:
                detection_duration = rospy.get_time() - start_time
                rospy.loginfo('Detection duration: %f, There is no contact', detection_duration)

            # Publish model output
            start_time = np.array(start_time).tolist()
            time_sec = int(start_time)
            time_nsec = start_time - time_sec
            model_msg.data = np.append(np.array([time_sec - big_time_digits, time_nsec, detection_duration, contact, collision, localization], dtype=np.complex128), np.hstack(window))
            model_pub.publish(model_msg)
        
        except Exception as e:
            rospy.logerr(f"Error in contact_detection: {e}")
        
        rate.sleep()

def signal_handler(sig, frame):
    print('Stopping the robot...')
    global running
    running = False

if __name__ == "__main__":
    global model_pub, big_time_digits, running,connection_thread
    
    signal.signal(signal.SIGINT, signal_handler)
    event = Event()
    event.clear()
    rospy.init_node('ur_robot_data')
    model_pub = rospy.Publisher("/model_output", numpy_msg(Floats), queue_size=1)
    scale = 1000000
    big_time_digits = int(rospy.get_time() / scale) * scale

    #robot = RTDEControlInterface(robot_ip)
    #print(robot.reuploadScript())
    #print(robot.isConnected())

    data_object = RTDEReceive(robot_ip, frequency)

    running = True
    i = 0
    file_name = input('ENTER DATA TAG NAME: ')
    file_name = data_path + file_name
    os.makedirs(file_name, exist_ok=True)
    file_name = file_name + '/' + str(rospy.Time.now().to_sec()) + '.txt'

    detection_thread = Thread(target=contact_detection, args=(data_object, event,))
    print('Waiting for the models to be loaded...')
    detection_thread.start()
    time.sleep(1)

    try:
        joints = pd.read_csv(joints_data_path)
        print(joints.head())

        data_object.startFileRecording(file_name)
        '''
        while running and not rospy.is_shutdown() and robot.isConnected():
            robot.moveL(np.array(joints.iloc[i]), speed=0.25, acceleration=0.2)
            if i < joints.shape[0] - 1:
                i = i + 1
            else:
                i = 0
        '''
        rospy.spin()
        event.set()
        detection_thread.join(timeout=1)
    except KeyboardInterrupt:
        print('Interrupted by user')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
    finally:
        print('Cleaning up resources...')
        #robot.stopScript()
        event.set()
        detection_thread.join(timeout=1)
        data_object.stopFileRecording()
        print('Program stopped.')

    #connection_thread.join(timeout=1)  # Ensure connection thread is terminated cleanly
