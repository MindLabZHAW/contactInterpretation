from frankx import Affine, LinearRelativeMotion, Robot, JointMotion
from frankx import Gripper
import numpy as np
import time 
robot_ip = "192.168.15.33"
robot_ip = "10.10.10.150"

gripper = Gripper(robot_ip)
robot = Robot(robot_ip)

# Recover from errors
robot.recover_from_errors()
robot.set_default_behavior()
gripper.move(0.03)

# Set velocity, acceleration and jerk to 5% of the maximum
robot.set_dynamic_rel(0.05)

#motion = LinearRelativeMotion(Affine(0, 0, 0.02))
#robot.move(motion)

print('\nPose: ', robot.current_pose())

state = robot.read_once()
print('Joints: ', state.q)
print('Joints_d: ', state.q_d)

feature_vector = np.array(state.q_d)- np.array(state.q)# Using external torques as an example

print('error: ', feature_vector)

joint_motion = state.q
joint_ID = 5
joint_motion[joint_ID] = joint_motion[joint_ID]-0.05

robot.move_async(JointMotion(joint_motion))
print('Joints: ', state.q)
print('Joints_d: ', state.q_d)
time.sleep(2)
'''
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_control import RTDEControlInterface as RTDEControl

ip_address = "192.168.163.11"
frequency = 125

robot_control = RTDEControl(ip_address)
robot_receive = RTDEReceive(ip_address, frequency)

target_joints = robot_receive.getActualQ()
print(target_joints)

target_joints[0] = target_joints[0]+0.1
print(target_joints)

robot_control.moveL_FK(target_joints, 0.2, 0.1, True)
print('Joints: ', robot_receive.getActualQ())
#robot_control.stopL()
print('Joints: ', robot_receive.getActualQ())
'''