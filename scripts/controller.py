#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist, PoseArray
from std_msgs.msg import Float32MultiArray, String
import tf
import numpy as np
from numpy import linalg
from utils import wrapToPi


class Controller:

    def __init__(self):
        rospy.init_node('turtlebot_controller', anonymous=True)
	rospy.Subscriber('/gazebo/model_states', ModelStates, self.gazebo_callback)
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def gazebo_callback(self, data):
        pose = data.pose[data.name.index("turtlebot3_burger")]
        twist = data.twist[data.name.index("turtlebot3_burger")]
        self.x = pose.position.x
        self.y = pose.position.y
        quaternion = (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.theta = euler[2]

    def get_ctrl_output(self):
        # use self.x self.y and self.theta to 
		# compute the right control input here
	
		### YOUR CODE HERE ###
		x_g = 1
		y_g = 1
		th_g = 0

	    #Define control inputs (V,om) - without saturation constraints
	    #Define Control constants k1,k2,k3
		k1 = 0.8
		k2 = 0.8
		k3 = 0.2
	    
		alpha = wrapToPi(np.arctan2((y_g - self.y),(x_g - self.x)) - self.theta)
		ro = ((y_g - self.y)**2 + (x_g - self.x)**2)**0.5
		delta = wrapToPi(np.arctan2((y_g - self.y),(x_g - self.x)) - th_g)
		V = k1 * ro * np.cos(alpha)
		om = k2 * alpha + k1 * np.sinc(alpha/np.pi) * np.cos(alpha) *(alpha + k3*delta)

	    # Apply saturation limits
		V = np.sign(V)*min(0.5, np.abs(V))
		om = np.sign(om)*min(1, np.abs(om))	

		if ((((self.x-x_g)**2 + (self.y-y_g)**2)**0.5) <= 0.01):
			[V,om] = [0,0]

		#Define X_DOT??
		cmd_x_dot = V * np.cos(self.theta)
		#Define theta_dot??
		cmd_theta_dot = om
		### END OF YOUR CODE ###        

		cmd = Twist()
		cmd.linear.x = cmd_x_dot
		cmd.angular.z = cmd_theta_dot
		return cmd

    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            ctrl_output = self.get_ctrl_output()
            self.pub.publish(ctrl_output)
            rate.sleep()

if __name__ == '__main__':
    ctrl = Controller()
    ctrl.run()
