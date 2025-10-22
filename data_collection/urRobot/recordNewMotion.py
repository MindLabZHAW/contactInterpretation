'''
By Maryam Rezayati

How to run?
1. unlock robot and acctivate remote control

    
2. run this program

	conda activate frankapyenv

'''
import os
import numpy as np
import pandas as pd
import time


from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_control import RTDEControlInterface

robot_ip = '192.168.125.5'
frequency = 100
robotType= 'UR10'
main_path = os.path.dirname(os.path.abspath(__file__))+'/'
print(main_path)



if __name__ == '__main__':

    # create robot controller instance
    robot = RTDEControlInterface(robot_ip)
    data_object = RTDEReceive(robot_ip, frequency)
    robot.teachMode()

    # Lists to store joint data
    joints = []


    counter = 0
    state = 'None'
    # Manual input of desired robot positions
    while (state !='exit'):
        state = input('Press enter when the robot is in a desired position or type exit:   ')
        #dummy = data_object.getActualQ()
        dummy = data_object.getActualTCPPose()
        print('\n', dummy)
        joints.append(dummy)
        counter += 1
        print(counter, 'poses are selected.\n')
    
    robot.endTeachMode()
    robot.stopScript()

    # Create a DataFrame and save joint data to a CSV file
    df = pd.DataFrame(data=np.squeeze(joints))
    named_tuple = time.localtime()
    filename = main_path + '/robotMotionPoints/'+robotType + str(time.strftime("_%m_%d_%Y_%H:%M:%S", named_tuple)) + '.csv'
    df.to_csv(filename,index=False)
    print('Data are saved in', filename)


