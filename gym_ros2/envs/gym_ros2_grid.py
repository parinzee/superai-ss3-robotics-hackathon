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

torch.autograd.set_detect_anomaly(True)

SENSOR_SECTION = 360

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


def minimum_segmented_lidar(ranges_arr):
    """
    : Description : Finding minimum value distance from lidar sensor in each sectors
    : Input :   ranges_arr = numpy.ndarray that contain 450 or 360 or any numbers float values of distance each angles in meter unit
    : Output :  section_list = list of float minimum value distance of each sector
    
    : Example : ranges_arr = numpy.ndarray of 451 value float of lidar
                section_list = [0.92, 0.89, 1.68, 1.80, 1.20, 1.02, 0.71, 0.54, 0.51, 0.52, 0.60, 0.91, 1.58, 0.93, 0.63]
    """
    lidar_value=list(ranges_arr)
    section_list = []
    if len(lidar_value) >= 450 : # for real lidar sensor
        pre_transform  = lidar_value[:450] # decreasing an list from 451 point to 450 point
        num_sub_section = len(pre_transform)// SENSOR_SECTION # len of sublist 
        section_list = [pre_transform[i:i+num_sub_section] for i in range(0, len(pre_transform), num_sub_section)] # make sub section 
        section_list = [min(sub_section) for sub_section in section_list]# get the min value out of each section to make it as an representative of each section
    else : # for simulator sensor
        pre_transform  = lidar_value[:450] #old value 360 # decreasing an list from 451 point to 450 point
        num_sub_section = len(pre_transform)// SENSOR_SECTION # len of sublist 
        section_list = [pre_transform[i:i+num_sub_section] for i in range(0, len(pre_transform), num_sub_section)] # make sub section 
        section_list = [min(sub_section) for sub_section in section_list ] # get the min value out of each section to make it as an representative of each section

    segmented_arr = list(map(lambda x:15 if x>15 else x,section_list))

    return segmented_arr

# ========

class LearningNode(Node):
    def __init__(self, x_init, y_init, x_target, y_target, min_range, env2ros, ros2env):
        super().__init__('wowowow')
        self.velPub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.dummy_req = Empty_Request()
        self.reset = self.create_client(Empty, '/reset_simulation')
        self.reset.call_async(self.dummy_req)

        self.timer_period = 2 # seconds
        self.timer = self.create_timer(self.timer_period, self.callback)

        self.x_init = x_init
        self.y_init = y_init
        self.x_target = x_target
        self.y_target = y_target
        self.min_range = min_range

        self.env2ros = env2ros
        self.ros2env = ros2env
    
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
        
        orientation_q = msg_odom.pose.pose.orientation
        orientation_list = [ orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        
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

        ranges_arr = minimum_segmented_lidar(ranges_arr)

        return ranges_arr, range_max, angle_max, angle_increment
    
    def get_state(self):
        x,y,yaw = self.odom_receive()

        ranges_arr, _, _, _ = self.scan_receive()
        distance = get_goal_distance(x, y, self.x_target, self.y_target)

        # Front back left right
        fblr = [ranges_arr[0], ranges_arr[179], ranges_arr[89], ranges_arr[269]]

        # If the distance is less than the 0.3, set it to 0 else 1
        fblr = [1 if x >= 0.35 else 0 for x in fblr]

        curr_row = x // 0.5
        curr_col = y // 0.5

        target_row = self.x_target // 0.5
        target_col = self.y_target // 0.5

        merged = np.array(list(fblr) + [curr_row, curr_col, target_row, target_col])

        # Remove nan
        merged = np.nan_to_num(merged, posinf=4.0, neginf=0.0)

        return merged
        
    
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
        self.action_space = spaces.Discrete(4) 

        # Observation Space = Laserscan + Robot Pose + distance + target
        self.observation_space = spaces.Box(shape=(4 + 2 + 2,), dtype=np.float64, low=-999.0, high=999.0)

        # Initialize data variables
        self.x_init = x_init
        self.y_init = y_init
        self.x_goal = x_goal
        self.y_goal = y_goal
        self.min_range = min_range
        self.initial_dis = get_goal_distance(x_init, y_init, x_goal, y_goal)
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
    
    def get_reward(self, state):

        if check_crash(state[:4], self.min_range):
            print("Crashed")
            return -100
        
        # If row and col are the same, then we have reached the goal
        if state[4] == state[6] and state[5] == state[7]:
            print("WON!")
            return 100
        
        dis_to_goal = get_goal_distance(*state[360:362], self.x_goal, self.y_goal)

        return self.initial_dis - dis_to_goal

    
    def step(self, action):

        # Action Here
        if action == 0:
            # Forward
            raw_action = [0.25, 0.0]
            self.env2ros.put([raw_action[0], raw_action[1]])
            self.env2ros.put([0.0, 0.0])
        elif action == 1:
            # Backward
            raw_action = [-0.25, 0.0]
            self.env2ros.put([raw_action[0], raw_action[1]])
            self.env2ros.put([0.0, 0.0])
        elif action == 2:
            # Left
            raw_action = [0.0, 3.14 / 4]
            self.env2ros.put([raw_action[0], raw_action[1]])
            self.env2ros.put([0.25, 0.0])
            self.env2ros.put([-raw_action[0], -raw_action[1]])
            self.env2ros.put([0.0, 0.0])
        else:
            # Right
            raw_action = [0.0, -(3.14 / 4)]
            self.env2ros.put([raw_action[0], raw_action[1]])
            self.env2ros.put([0.25, 0.0])
            self.env2ros.put([-raw_action[0], -raw_action[1]])
            self.env2ros.put([0.0, 0.0])

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
            # 'x': next_state[360],
            # 'y': next_state[361],
            # 'yaw': next_state[362],
            # 'dis': get_goal_distance(*next_state[360:362], self.x_goal, self.y_goal),
            # 'reward': reward,
        }

        return next_state, reward, done, info
    
    def action_masks(self):
        self.env2ros.put("reset_world")
        state = self.ros2env.get()
        return np.array(state[:4])

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