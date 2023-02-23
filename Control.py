#!/usr/bin/env python
# -*- coding: <utf-8> -*-

from time import time
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState
from math import *
import numpy as np
from tf_transformations import euler_from_quaternion, quaternion_from_euler
import math

# Q-learning speed parameters
CONST_LINEAR_SPEED_FORWARD = 0.8  #0.08
CONST_ANGULAR_SPEED_FORWARD = 0.8  #0.0
CONST_LINEAR_SPEED_TURN = 0.8  #0.06
CONST_ANGULAR_SPEED_TURN = 0.8  #0.4
 
# Feedback control parameters
K_RO = 2 #2 
K_ALPHA = 15 #15
K_BETA = -1 #-1
V_CONST = 0.45 # 0.45[m/s]

# Goal reaching threshold
GOAL_DIST_THRESHOLD = 0.1 # [m]
GOAL_ANGLE_THRESHOLD = 15 # [degrees]

# Get theta in [radians]
def getRotation(odomMsg):
    orientation_q = odomMsg.pose.pose.orientation
    orientation_list = [ orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
    return yaw

# Get (x,y) coordinates in [m]
def getPosition(odomMsg):
    x = odomMsg.pose.pose.position.x
    y = odomMsg.pose.pose.position.y
    return ( x , y)

# Get linear speed in [m/s]
def getLinVel(odomMsg):
    return odomMsg.twist.twist.linear.x

# Get angular speed in [rad/s] - z axis
def getAngVel(odomMsg):
    return odomMsg.twist.twist.angular.z

# Create rosmsg Twist()
def createVelMsg(v,w):
    velMsg = Twist()
    velMsg.linear.x = float(v)
    velMsg.linear.y = 0.
    velMsg.linear.z = 0.
    velMsg.angular.x = 0.
    velMsg.angular.y = 0.
    velMsg.angular.z = float(w)
    return velMsg

# Go forward command
def robotGoForward(velPub):
    velMsg = createVelMsg(CONST_LINEAR_SPEED_FORWARD,CONST_ANGULAR_SPEED_FORWARD)
    velPub.publish(velMsg)

# Turn left command
def robotTurnLeft(velPub,velstop,theta,pi):
    velMsg = createVelMsg( 0 , +CONST_ANGULAR_SPEED_TURN )
    velPub.publish(velMsg)
    if abs(theta-(pi/2)) < 0.15:
        robotStop(velstop)

# Turn right command
def robotTurnRight(velPub,velstop,theta,pi):
    velMsg = createVelMsg( 0 , -CONST_ANGULAR_SPEED_TURN )
    velPub.publish(velMsg)
    if abs(theta+(pi/2)) < 0.15:
        robotStop(velstop)

# Stop command
def robotStop(velPub):
    velMsg = createVelMsg(0.0,0.0)
    velPub.publish(velMsg)

def robotTurn(velPub):
    velMsg = createVelMsg( -CONST_ANGULAR_SPEED_TURN , -CONST_ANGULAR_SPEED_TURN )
    velPub.publish(velMsg)


def robotTurnB(velPub,velstop,theta,pi):
    velMsg = createVelMsg( 0 , -CONST_ANGULAR_SPEED_TURN )
    velPub.publish(velMsg)
    if abs(theta-(pi)) < 0.15:
        robotStop(velstop)

def checkTurn(lidar): #================================================================
    lidar_horizon = np.concatenate((lidar[(ANGLE_MIN + HORIZON_WIDTH):(ANGLE_MIN):-1],lidar[(ANGLE_MAX):(ANGLE_MAX - HORIZON_WIDTH):-1]))
    W = np.linspace(1.2, 1, len(lidar_horizon) // 2)
    W = np.append(W, np.linspace(1, 1.2, len(lidar_horizon) // 2))
    
    if np.min( W * lidar_horizon ) < COLLISION_DISTANCE:
        return True
    else:
        return False  

def robotDoAction(velPub, action):
    status = 'robotDoAction => OK'
    if action == 0:
        robotGoForward(velPub)
    elif action == 1:
        robotTurnLeft(velPub)
    elif action == 2:
        robotTurnRight(velPub)
    else:
        status = 'robotDoAction => INVALID ACTION'
        robotGoForward(velPub)

    return status

def robotFeedbackControl(velPub, x, y, theta, x_goal, y_goal, theta_goal):
    # Normalize theta_goal to range (-pi, pi]
    theta_goal_norm = math.atan2(math.sin(theta_goal), math.cos(theta_goal))

    # Calculate distance and heading to goal
    dx = x_goal - x
    dy = y_goal - y
    distance_to_goal = math.sqrt(dx**2 + dy**2)
    goal_heading = math.atan2(dy, dx)

    # Calculate angle errors
    alpha = (goal_heading - theta + math.pi) % (2 * math.pi) - math.pi
    beta = (theta_goal_norm - goal_heading + math.pi) % (2 * math.pi) - math.pi

    if distance_to_goal < GOAL_DIST_THRESHOLD and math.degrees(abs(theta - theta_goal_norm)) < GOAL_ANGLE_THRESHOLD:
        # Goal position reached, stop moving
        status = 'Goal position reached!'
        v = 0
        w = 0
    else:
        # Move towards goal
        status = 'Goal position not reached!'
        v = K_RO * distance_to_goal
        w = K_ALPHA * alpha + K_BETA * beta

    if v == 0:
        v_scal = 0
        w_scal = 0
    else:
        v_scal = math.copysign(V_CONST, v)
        w_scal = w * (V_CONST / abs(v))

    # Publish velocity message
    velMsg = createVelMsg(v_scal, w_scal)
    velPub.publish(velMsg)
    return status

# Stability Condition
def check_stability(k_rho, k_alpha, k_beta):
    return k_rho > 0 and k_beta < 0 and k_alpha > k_rho

# Strong Stability Condition
def check_strong_stability(k_rho, k_alpha, k_beta):
    return k_rho > 0 and k_beta < 0 and k_alpha + 5 * k_beta / 3 - 2 * k_rho / np.pi > 0
