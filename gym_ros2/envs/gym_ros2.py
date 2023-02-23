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

# ========

class LearningNode(Node):
    def __init__(self, x_init, y_init, x_target, y_target, min_range, env2ros, ros2env):
        super().__init__('wowowow')
        self.velPub = self.create_publisher(Twist, 'cmd_vel', 10)
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
        
        return ranges_arr, range_max, angle_max, angle_increment
    
    def get_state(self):
        x,y,yaw = self.odom_receive()

        ranges_arr, _, _, _ = self.scan_receive()
        distance = get_goal_distance(x, y, self.x_target, self.y_target)

        merged =  np.array(list(ranges_arr) + [x, y, yaw, distance, self.x_target, self.y_target])

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
        self.action_space = spaces.Discrete(5) 

        # Observation Space = Laserscan + Robot Pose + distance + target
        self.observation_space = spaces.Box(shape=(360 + 3 + 1 + 2,), dtype=np.float64, low=-999.0, high=999.0)

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

        if check_crash(state[:360], self.min_range):
            print("Crashed")
            return -100
        
        if check_win(*state[360:362], self.x_goal, self.y_goal, self.min_range):
            print(state[360:362])
            print(self.x_goal)
            print(self.y_goal)
            print("WON!")
            return 100
        
        dis_to_goal = get_goal_distance(*state[360:362], self.x_goal, self.y_goal)

        return self.initial_dis - dis_to_goal

    
    def step(self, action):

        # Action Here
        if action == 0:
            raw_action = [0.0, 3.14]
        elif action == 1:
            raw_action = [0.0, 3.14 / 2]
        elif action == 2:
            raw_action = [1.5, 0.0]
        elif action == 3:
            raw_action = [0.0, -3.14 / 2]
        else:
            raw_action = [0.0, -3.14]
        
        # Start Running
        self.env2ros.put([raw_action[0], raw_action[1]])
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
            'x': next_state[360],
            'y': next_state[361],
            'yaw': next_state[362],
            'dis': get_goal_distance(*next_state[360:362], self.x_goal, self.y_goal),
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