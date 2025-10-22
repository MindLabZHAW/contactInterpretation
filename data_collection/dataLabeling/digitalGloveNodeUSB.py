#!/usr/bin/env python3
'''
By Maryam Rezayati

How to run:

conda activate frankapyenv
source /opt/ros/noetic/setup.bash
$HOME/miniconda/envs/frankapyenv/bin/python dataLabeling/digitalGloveNodeUSB.py

'''
## import required libraries 

import numpy as np
from canlib import canlib
import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

class digital_glove():
	def __init__(self, channel_number = 0):
		rospy.init_node('digital_glove', anonymous=True)
		rospy.loginfo("***  ros node is created  ***")
		self.pub = rospy.Publisher("/contactTimeIndex", numpy_msg(Floats), queue_size = 10)

		print("Opening channel %d" % (channel_number))
		# Open CAN channel, virtual channels are considered ok to use
		self.ch = canlib.openChannel(channel_number)#, canlib.canOPEN_ACCEPT_VIRTUAL)

		print("Setting bitrate to 1 Mb/s")
		self.ch.setBusParams(canlib.canBITRATE_1M)

		self.ch.busOn()
		rospy.loginfo("   ID     DLC      DATA     Timestamp    computerTime")
		#self.pub.publish("time_sec, time_nsec, timestamp, DATA, timestamp")
		scale = 1000000
		self.big_time_digits = int(rospy.get_time()/scale)*scale

	def readPublish_data(self):
		try:
			self.frame = self.ch.read(timeout=1)
			read_time = rospy.get_time()
			#rospy.loginfo(read_time)
			"""Prints a message to screen"""
			frame = self.frame
			if (frame.flags & canlib.canMSG_ERROR_FRAME == 0):
		
				msg = Floats()
				start_time = np.array(read_time).tolist()
				time_sec = int(start_time)
				time_nsec = start_time-time_sec
				data = np.array([time_sec-self.big_time_digits, time_nsec, frame.timestamp])
				for i in frame.data:
					data = np.append(data,i)
				msg.data = data
				#rospy.loginfo(msg.data[3:])
				self.pub.publish(msg)
		except(canlib.canNoMsg) as ex:
			None
		except (canlib.canError) as ex:
			print(ex)
		
	
	def close(self):
		self.ch.busOff()
		self.ch.close()

if __name__ == '__main__':
	rate_value = 1000
	dg = digital_glove(channel_number=0)
	rate = rospy.Rate(rate_value)

	while not rospy.is_shutdown():
		try:
			dg.readPublish_data()
			#rate.sleep()
		except:
			dg.close()
			print('done')
	dg.close()
	rospy.signal_shutdown("Finished ........  ")

