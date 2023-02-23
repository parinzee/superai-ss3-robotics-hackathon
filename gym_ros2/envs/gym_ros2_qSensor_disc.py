import gym

from gym import spaces

import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty

from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.signals import SignalHandlerGuardCondition
from rclpy.utilities import timeout_sec_to_nsec

from std_msgs.msg import String

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

import numpy as np

from geometry_msgs.msg import Twist

from tf_transformations import euler_from_quaternion, quaternion_from_euler

from std_srvs.srv import Empty
from std_srvs.srv._empty import Empty_Request
from time import sleep
from gym.envs.registration import register
import math
import threading
from queue import Queue
from gazebo_msgs.msg import ModelState
import torch
from Control import *

torch.autograd.set_detect_anomaly(True)

# Set Lidar segmetation
LIDAR_SEGMENT = 90


# AUXILAIRY FUNCTIONS
def check_crash(ranges_arr, min_range):
    """
    : Description : Check robot is crashing the obstacle or not
    : Input :
        ranges_arr - List(float32) => List of Lidar sensor
        min_range  - float32       => Minimum range for detecting crashing
    : Output :
        is_crash   - Boolean       => Robot is crashing (True) / Not crashing (False)
    """
    is_crash = min(ranges_arr) < min_range
    return is_crash

def get_goal_distance(x_now, y_now, x_goal, y_goal):
    """
    : Description : Calculate current distance from robot to goal position
    : Input :
        x_now   - float32   => Current position of robot in x-axis
        y_now   - float32   => Current position of robot in y-axis
        x_goal  - float32   => Position of goal in x-axis
        y_goal  - float32   => Position of goal in y-axis
    : Output :
        distance - float32  => Current distance from robot to goal
    """
    distance = math.hypot(x_goal - x_now, y_goal - y_now)
    return distance

def check_win(x_now, y_now, x_goal, y_goal, goal_r):
    """
    : Description : Check robot is on Goal area or not
    : Input :
        x_now   - float32   => Current position of robot in x-axis
        y_now   - float32   => Current position of robot in y-axis
        x_goal  - float32   => Position of goal in x-axis
        y_goal  - float32   => Position of goal in y-axis
        goal_r  - float32   => Radius of goal for describing goal area
    : Output :
        is_win   - Boolean  => Robot is on Goal area (True) / Not on Goal area (False)
        distance - float32  => Current distance from robot to goal
    """
    current_distance = get_goal_distance(x_now, y_now, x_goal, y_goal)
    if current_distance < goal_r:
        is_win = True
    else:
        is_win = False
    return is_win

def quantized_lidar(ranges_arr, sensor_section):
    """
    ADD by Senmee 24 FEB 2023

    : Description : Finding minimum value distance from lidar sensor in each sectors
    : Input :   ranges_arr = numpy.ndarray that contain 450 or 360 or any numbers float values of distance each angles in meter unit
    : Output :  section_list = list of float minimum value distance of each sector
    
    : Example : ranges_arr = numpy.ndarray of 451 value float of lidar
                section_list = [0.92, 0.89, 1.68, 1.80, 1.20, 1.02, 0.71, 0.54, 0.51, 0.52, 0.60, 0.91, 1.58, 0.93, 0.63]
    """

    lidar_value = []

    # Handle value received from sensor
    for i in range(len(ranges_arr)):
        if ranges_arr[i] == float('Inf') or ranges_arr[i] == float('inf'):      # Convert Inf to 3.5
            lidar_value.append(3.5)
        elif np.isnan(ranges_arr[i]) or ranges_arr[i] == float('nan'):          # Convert NaN to 0
            lidar_value.append(0)
        elif ranges_arr[i] > 3.5:                                               # Set maximum value to 3.5
            lidar_value.append(3.5)
        elif ranges_arr[i] < 0:                                                 # Set minimum value to 0
            lidar_value.append(0)                                   
        else:                                                                   # else: collect that value
            lidar_value.append(ranges_arr[i])

    section_list = []

    if len(lidar_value) >= 450 :                                                # for real lidar sensor
        pre_transform  = lidar_value[:450]                                      # decreasing an list from 451 point to 450 point
    else:
        pre_transform   = lidar_value[:360]
    num_sub_section = len(pre_transform)//sensor_section                        # len of sublist 
    section_list = [pre_transform[i:i+num_sub_section] for i in range(0, len(pre_transform), num_sub_section)] # make sub section 
    section_list = [min(sub_section) for sub_section in section_list]           # get the min value out of each section to make it as an representative of each section

    segmented_arr =list(map(lambda x: 3.5 if x>3.5 else x, section_list))

    return segmented_arr

def normalized(number, high, low):
    """
    ADD by Senmee 24 FEB 2023

    : Description : Convert range [low, high] to range [0, 1]
    """
    number = (number - low) / (high - low)
    number = np.clip(number, 0, 1)
    return number

# ======== 

class LearningNode(Node):
    def __init__(self, x_init, y_init, x_target, y_target, min_range, env2ros, ros2env):
        super().__init__('wowowow')
        self.velPub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.velStop = self.create_publisher(Twist, 'cmd_vel', 0)
        self.dummy_req = Empty_Request()
        self.reset = self.create_client(Empty, '/reset_simulation')
        self.reset.call_async(self.dummy_req)

        self.timer_period = .5 # seconds
        self.timer = self.create_timer(self.timer_period, self.callback)

        self.x_init = x_init
        self.y_init = y_init
        self.x_target = x_target
        self.y_target = y_target
        self.min_range = min_range

        self.env2ros = env2ros
        self.ros2env = ros2env

        self.x_step = 0
        self.y_step = 0
        self.yaw = 0
        self.position
    
    def callback(self):
        if not self.env2ros.empty():
            # env2ros = "get_state", "reset_world", [v, w]
            cmd = self.env2ros.get()


            if cmd == "get_state":
                self.ros2env.put(self.get_state())


            elif cmd == "reset_world":
                self.reset_world()
                self.ros2env.put(self.get_state())

            else:
                self.publisher_vel(cmd[0], cmd[1])
        
    def odom_receive(self):

        _, msg_odom = self.wait_for_message('/odom', Odometry)
   
         
        x = msg_odom.pose.pose.position.x
        y = msg_odom.pose.pose.position.y
        # z = msg_odom.pose.pose.position.z
        self.position = msg_odom.pose.pose.position
        
        
        orientation_q = msg_odom.pose.pose.orientation
        orientation_list = [ orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.yaw = yaw
        
        return x,y,yaw
    
        
    def scan_receive(self):
        _,msg_scan=self.wait_for_message('/scan', LaserScan)
        
        ranges_arr = np.array(msg_scan.ranges)
        # len_ranges = len(ranges_arr)
        range_max = msg_scan.range_max 
        # range_min = msg_scan.range_min
        angle_max = msg_scan.angle_max
        # angle_min = msg_scan.angle_min
        angle_increment = msg_scan.angle_increment
        # scan_time = msg_scan.scan_time
        # time_increment = msg_scan.time_increment
        # intensities = msg_scan.intensities
        # header = msg_scan.header
        
        return ranges_arr, range_max, angle_max, angle_increment
    
    def get_state(self):
        x,y,yaw = self.odom_receive()

        ranges_arr, _, _, _ = self.scan_receive()

        ### ADD BY SENMEE 24 FEB 2023
        ranges_arr = quantized_lidar(ranges_arr, LIDAR_SEGMENT)
        ranges_arr = normalized(ranges_arr, 3.5, 0)
        return ranges_arr
        
    
    def publisher_vel(self,v,w):
        velMsg = Twist()
        velMsg.linear.x = v
        # velMsg.linear.y = 0.
        # velMsg.linear.z = 0.
        # velMsg.angular.x = 0.
        # velMsg.angular.y = 0.
        velMsg.angular.z = w
        self.velPub.publish(velMsg)
    
    def reset_world(self):

        self.reset.call_async(self.dummy_req)

        checkpoint = ModelState()
        checkpoint.model_name = 'turtlebot3_burger'
        checkpoint.pose.position.x = float(self.x_init)
        checkpoint.pose.position.y = float(self.y_init)
        checkpoint.pose.position.z = 0.0

    def wait_for_message(
        node,
        topic: str,
        msg_type,
        time_to_wait=-1
    ):
        """
        Wait for the next incoming message.
        :param msg_type: message type
        :param node: node to initialize the subscription on
        :param topic: topic name to wait for message
        :time_to_wait: seconds to wait before returning
        :return (True, msg) if a message was successfully received, (False, ()) if message
            could not be obtained or shutdown was triggered asynchronously on the context.
        """
        context = node.context
        wait_set = _rclpy.WaitSet(1, 1, 0, 0, 0, 0, context.handle)
        wait_set.clear_entities()

        sub = node.create_subscription(msg_type, topic, lambda _: None, 1)
        wait_set.add_subscription(sub.handle)
        sigint_gc = SignalHandlerGuardCondition(context=context)
        wait_set.add_guard_condition(sigint_gc.handle)

        timeout_nsec = timeout_sec_to_nsec(time_to_wait)
        wait_set.wait(timeout_nsec)

        subs_ready = wait_set.get_ready_entities('subscription')
        guards_ready = wait_set.get_ready_entities('guard_condition')

        if guards_ready:
            if sigint_gc.handle.pointer in guards_ready:
                return (False, None)

        if subs_ready:
            if sub.handle.pointer in subs_ready:
                msg_info = sub.handle.take_message(sub.msg_type, sub.raw)
                return (True, msg_info[0])

        return (False, None)

class ROS2Env(gym.Env):
    """Custom Gym environment for ROS2"""
    metadata = {'render.modes': ['human']}

    def __init__(self, x_init = 0, y_init = 0, x_goal = 1.85, y_goal = 0.619, min_range = 0.15, train=True):
        super(ROS2Env, self).__init__()

        # Action Space = Left, Right, Forward, Northeast, Northwest
        self.action_space = spaces.Discrete(5)

        # Observation Space = Laserscan + Robot Pose + distance + target
        self.observation_space = spaces.Box(shape=(LIDAR_SEGMENT), dtype=np.float64, low=-0, high=3.5)

        # Initialize data variables
        self.x_init = x_init
        self.y_init = y_init
        self.x_goal = x_goal
        self.y_goal = y_goal
        self.min_range = min_range
        # self.initial_dis = get_goal_distance(x_init, y_init, x_goal, y_goal)
        self.train = train
        
        # Queues for communication
        self.env2ros = Queue()
        self.ros2env = Queue()

        rclpy.init()
        self.executor = rclpy.executors.MultiThreadedExecutor()

        if self.train:
            self.node = LearningNode(self.x_init, self.y_init, self.x_goal, self.y_goal, self.min_range, self.env2ros, self.ros2env)
            self.executor.add_node(self.node)
        else:
            raise NotImplementedError
        
        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()

        self.count_step = 0
    
    def get_reward(self, state):

        if check_crash(state[:LIDAR_SEGMENT], self.min_range):
            print("Crashed")
            return -100
        
        if check_win(self.position.x, self.position.y, self.x_goal, self.y_goal, self.min_range):
            print(self.position.x, self.position.y)
            print(self.x_goal)
            print(self.y_goal)
            print("WON!")
            return 100
        
        # dis_to_goal = get_goal_distance(*state[LIDAR_SEGMENT:LIDAR_SEGMENT+2], self.x_goal, self.y_goal)
        self.count_step = self.count_step + 1

        return -self.count_step

    
    def step(self, action):

        # Action Here
        if action == 0:                     # Forward
            robotGoForward(self.velPub)
        elif action == 1:                   # Left
            robotTurnLeft(self.velPub,self.velstop,self.yaw,math.pi)
        elif action == 2:                   # Right
            robotTurnRight(self.velPub,self.velstop,self.yaw,math.pi)
        else:                               # Backward
            robotTurnB(self.velPub, self.velstop, self.yaw, math.pi)
        
        # Start Running
        # self.env2ros.put([raw_action[0], raw_action[1]])
        # self.env2ros.put([0.0, 0.0])

        # Get Next State
        self.env2ros.put("get_state")

        next_state = self.ros2env.get(block=True)

        # Reward
        reward = self.get_reward(next_state)

        # Done 
        if reward == 100 or reward == -100:
            done = True
        else:
            done = False

        info = {
            'x': self.position.x,
            'y': self.position.y,
            'yaw': self.yaw,
            'dis': get_goal_distance(self.position.x, self.position.y, self.x_goal, self.y_goal),
            'reward': reward,
        }

        return next_state, reward, done, info

    def reset(self):
        self.env2ros.put("reset_world")

        state = self.ros2env.get()

        return state

    def render(self, mode='human'):
        pass

    def close(self):
        self.executor_thread.join()
        # self.node.destroy_node()
        rclpy.shutdown()