from frankx import Affine, LinearRelativeMotion, Robot, JointMotion
from frankx import Gripper
import numpy as np
import time 
gripper = Gripper("192.168.15.33")
robot = Robot("192.168.15.33")
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
joint_motion[0] = joint_motion[0]+0.05

robot.move_async(JointMotion(joint_motion))
print('Joints: ', state.q)
print('Joints_d: ', state.q_d)
time.sleep(2)
