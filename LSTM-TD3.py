import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty

from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.signals import SignalHandlerGuardCondition
from rclpy.utilities import timeout_sec_to_nsec

from std_msgs.msg import String

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from tf_transformations import euler_from_quaternion, quaternion_from_euler


import numpy as np

from gazebo_msgs.msg import ModelState

from geometry_msgs.msg import Twist, Pose

from math import *

from std_srvs.srv import Empty
from std_srvs.srv._empty import Empty_Request

# Import more library
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.optim as optim
from torch.distributions import Normal

import math
from collections import deque
import random
import os
import json
import time
import sys
import copy


### ============= INITIAL RARAMETERS ============= ###

X_INIT          = 0.
Y_INIT          = 0.
THETA_INIT      = 0.
X_GOAL          = 0.542760
Y_GOAL          = 0.520140
SENSOR_SECTION  = 20
MIN_RANGE       = 0.136
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DIS_START_GOAL = np.sqrt((X_INIT - X_GOAL)**2 + (Y_INIT - Y_GOAL)**2)

### ============================================== ###


### ================ ROBOT SECTION ================ ###
def robotSetPos(setPosPub, x, y, theta):
    checkpoint = ModelState()

    checkpoint.model_name = 'turtlebot3_burger'

    checkpoint.pose.position.x = float(x)
    checkpoint.pose.position.y = float(y)
    checkpoint.pose.position.z = 0.0

    [x_q,y_q,z_q,w_q] = quaternion_from_euler(0.0,0.0,radians(theta))

    checkpoint.pose.orientation.x = x_q
    checkpoint.pose.orientation.y = y_q
    checkpoint.pose.orientation.z = z_q
    checkpoint.pose.orientation.w = w_q

    checkpoint.twist.linear.x = 0.0
    checkpoint.twist.linear.y = 0.0
    checkpoint.twist.linear.z = 0.0

    checkpoint.twist.angular.x = 0.0
    checkpoint.twist.angular.y = 0.0
    checkpoint.twist.angular.z = 0.0

    setPosPub.publish(checkpoint)
    return ( x , y , theta )

### ============================================== ###


### ============= RELATIVE FUNCTION ============== ###
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
    return is_win, current_distance

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
        num_sub_section = len(pre_transform)//SENSOR_SECTION # len of sublist 
        section_list = [pre_transform[i:i+num_sub_section] for i in range(0, len(pre_transform), num_sub_section)] # make sub section 
        section_list = [min(sub_section) for sub_section in section_list]# get the min value out of each section to make it as an representative of each section
    else : # for simulator sensor
        pre_transform  = lidar_value[:450] #old value 360 # decreasing an list from 451 point to 450 point
        num_sub_section = len(pre_transform)//SENSOR_SECTION # len of sublist 
        section_list = [pre_transform[i:i+num_sub_section] for i in range(0, len(pre_transform), num_sub_section)] # make sub section 
        section_list = [min(sub_section) for sub_section in section_list ] # get the min value out of each section to make it as an representative of each section

    segmented_arr =list(map(lambda x:15 if x>15 else x,section_list))

    return segmented_arr

def init_weight(w):
    """
    : Description : Set initial weight of neural network base on xavier uniform
    """
    if isinstance(w, nn.Linear):                            # Check w is a subset of nn.Linear or not
        nn.init.xavier_uniform_(w.weight, gain=1)           # Fill the input Tensor with values according to uniform distribution
        nn.init.constant_(w.bias, 0)                        # Fill bias of Tensor as 0

def hard_update(target, source):
    """
    : Description : Update weight by copying source's parameter to target
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_update(target, source, tau):
    """
    : Description : Update weight by copying source's parameter to target with some noise
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + (param.data * tau))

def action_unnormalized(action, high, low):
    """
    : Description : Convert Normalized value [-1, 1] to range [low, high]
    """
    action = low + (action + 1.0) / 2 * (high - low)
    action = np.clip(action, low, high)
    return action

### ============================================== ###

### MORE CLASS ###
class ReplayBuffer:
    """
    : Description : Storing experience of robot state and action for stopping the temporal correlations 
                    between different episodes in the training phase
    """
    def __init__(self, capacity):
        self.capacity = capacity                    # Set maximum size of buffer storage
        self.buffer = []                            # Set initial buffer be empty set
        self.position = 0                           # Set initial position to zero

    def push(self, state, action, reward, next_state, done):
        """
        : Description : Add new experience to buffer, if buffer is full the oldest one will be rewrite
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        : Description : Get random sample of experience from Replay buffer based on batch size
                        ! Note: The sample should be used when ReplayBuffer's size > batch size !
        """
        batch = random.sample(self.buffer, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(np.stack, zip(*batch))
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
    
    def __len__(self):
        """
        : Description : Get length of Replay buffer
        """
        return len(self.buffer)

class ActorNetwork(nn.Module):
    """
    : Description : The network for giving Policy
    """
    def __init__(self, state_dim, action_dim, hidden_dim, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, action_dim)        # Get output as mean
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)     # Get another output as log standard deviation

        self.apply(init_weight)                                     # Initial weight of model
    
    def forward(self, state):
        """
        : Description : Feed forward of Actor network for creating policy
        : Input : 
            state - The current state of robot including 
                    [n] Current LiDAR sensor, [1] previous V, [1] previous W, [1] Current distance from target, 
                    and [1] Current angle from target
        : Output : 
            mean        - The value for assign mean of action data
            log_stad    - The value for assing log standard deviation that bounded between log_std_min and log_std_max
        """
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)  # Set limit of minimum and maximum of log_std
        return mean, log_std                                                        # Return Mean and Log stad
    
    def sample(self, state, epsilon=1e-6):
        """
        : Description : Get sample of action from normal distribution based on mean and log_std value from feed forward
        : Input : 
            state   - List(float32) =>  The current state of robot including 
                                        Current LiDAR sensor, previous V, previous W, Current distance from target, 
                                        Current angle from target
            epsilon - List(float32) =>  Constant value from calculating log probrability
        : output : 
            action  - List(float32) =>  Predicted action for Robot moving including [1] Linear velocity (v) and 
                                        [1] Angular velocity (w)
            log_prob- float32       =>  Logarithm of policy probrability
            mean    - float32       =>  Average of policy
            log_std - float32       =>  Log of standard deviation of policy
        """
        mean, log_std = self.forward(state)                             # Get mean and log std from feed forward
        std = log_std.exp()                                             # Get standard deviation
        normal = Normal(mean, std)                                      # Get Normal distribution curve
        xt = normal.rsample()                                           # Get parameterized sample from normal distribution
        action = torch.tanh(xt)                                         # Get action in bound of [-1, 1]
        log_prob = normal.log_prob(xt)                                  # Get log of policy probrability
        log_prob = log_prob - torch.log(1-action.pow(2) + epsilon)      # 
        log_prob = log_prob.sum(1, keepdim=True)                        # Get final log prob by summation

        return action, log_prob, mean, log_std

class CriticNetwork(nn.Module):
    """
    : Description : The network for giving Q-table
    """
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()

        # Q1
        self.l1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.l3_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.l4_q1 = nn.Linear(hidden_dim, 1)

        #Q2
        self.l1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.l4_q2 = nn.Linear(hidden_dim, 1)

        self.apply(init_weight)
    
    def forward(self, state, action):
        """
        : Description : Feedforward data to Critic network
        : Input :
            state  - List(float32)  =>  The list of [n] current sensor data, [1] previous v, [1] previous w, 
                                        [1] current goal angle, and [1] current goal distance
            action - List(float32)  =>  The list of [1] current v, and [1] current w which get from Actor Network
        : Output :
            x1     - float32        =>  Q-value that predicted by Critic network 1
            x2     - float32        =>  Q-value that predicted by Critic network 2
        """
        state_action = torch.cat([state, action], 1)        # Concat state and action to be input data

        x1 = F.relu(self.l1_q1(state_action))               # Duplicated network for more efficiency
        x1 = F.relu(self.l2_q1(x1))
        x1 = F.relu(self.l3_q1(x1))
        x1 = self.l4_q1(x1)


        x2 = F.relu(self.l1_q2(state_action))               # Duplicated network for more efficiency
        x2 = F.relu(self.l2_q2(x2))
        x2 = F.relu(self.l3_q2(x2))
        x2 = self.l4_q2(x2)

        return x1, x2

class SAC(object):
    """
    : Description : Soft Action-Critic model
    """
    def __init__(self, 
                 state_dim,
                 action_dim,
                 gamma = 0.99,
                 tau = 1e-2,
                 alpha = 0.2,
                 hidden_dim = 256,
                 lr = 3e-4
                 ):
        self.gamma = gamma      # Decay rate
        self.tau = tau
        self.alpha = alpha
        self.lr = lr

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')              # Get device for processing

        ### Actor-Critic model ###
        # Get Policy from Actor Network
        self.policy = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)           # Actor Network
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)                          # Set optimization to Actor Network

        # Get Q-value from Critic Netwokr
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)          # Critic Network
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)                          # Set optimization to Critic Network
        self.critic_target = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)   # Duplicated critic network
        hard_update(self.critic_target, self.critic)                                            # Hard copy weight to set duplicated network

        # Entropy
        self.target_entropy = torch.prod(torch.Tensor([action_dim]).to(self.device)).item()     # Get product of action dimension?

        # Alpha
        self.log_alpha = torch.zeros(1, requires_grad = True, device=self.device)               # Create 1x1 tensor to be log_alpha
        self.alpha_optim = Adam([self.log_alpha], lr=self.lr)                                   # Set optimization to log_alpha

    def select_action(self, state, eval=False):
        """
        : Description : Select action from policy distribution
        : Input : 
            state - List(float32)   =>  The current state of robot including 
                                        [n] Current LiDAR sensor, [1] previous V, [1] previous W, 
                                        [1] Current distance from target, and [1] Current angle from target
            eval  - Boolean         =>  select action for evaluation (True) / for training (False)
        : Output :
            action - List(float32)  =>  The list of [1] current v, and [1] current w
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if not eval:
            action, _, _, _ = self.policy.sample(state)     # Select real action if model is training
        else:
            _, _, action, _ = self.policy.sample(state)     # Select mean value of policy if model is evaluation

        action = action.detach().cpu().numpy()[0]
        return action
    
    def update_parameters(self, replay_bf, batch_size):
        """
        : Description : 
        : Input :
            replay_bf - List(Object)   =>  Experience data that contained by ReplayBuffer
                                           which includes state, action, reward, next_state, done
            batch_size - Integer       =>  Sample size for collecting from ReplayBuffer
        """
        # Get sample expereince from buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_bf.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)     # Add dimension
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)         # Add dimension

        # CALCULATE PREDICTION
        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _  = self.policy.sample(next_state_batch)                      # Get next state Policy
            qf1_next_target, qf2_next_target            = self.critic_target(next_state_batch, next_state_action)   # Get next state Q-value
            min_qf_next_target  = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi      # Set up minimum Q-value for next target
            next_q_value        = reward_batch + (1 - done_batch) * self.gamma * (min_qf_next_target)               # Get Target
        
        qf1, qf2 = self.critic(state_batch, action_batch)   # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)            # Calculate MSE between Q-value from Critric 1 and actual
        qf2_loss = F.mse_loss(qf2, next_q_value)            # Calculate MSE between Q-value from Critric 2 and actual
        qf_loss = qf1_loss + qf2_loss                       # Set summation of MSE be loss function

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, mean, log_std = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() 

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()

        soft_update(self.critic_target, self.critic, self.tau)

    # Save model parameters
    def save_models(self, episode_count):
        torch.save(self.policy.state_dict(), DIR_PATH + '/model/' + str(episode_count)+ '_policy_net.pth')
        torch.save(self.critic.state_dict(), DIR_PATH + '/model/' + str(episode_count)+ 'value_net.pth')
        print("====================================")
        print("...     Model has been saved     ...")
        print("====================================")
    
    # Load model parameters
    def load_models(self, episode, world):
        self.policy.load_state_dict(torch.load(DIR_PATH + '/model/' + str(episode)+ '_policy_net.pth'))
        self.critic.load_state_dict(torch.load(DIR_PATH + '/model/' + str(episode)+ 'value_net.pth'))
        hard_update(self.critic_target, self.critic)
        print('***Models load***')

class LearningNode(Node):
    def __init__(self):
        super().__init__('test2')
        print("Start train")
        self.timer_period = .1 # seconds
        self.timer = self.create_timer(self.timer_period, self.call_back)
        self.velPub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.setPosPub = self.create_publisher(ModelState, 'gazebo/set_model_state', 10)
        self.ep_time = self.get_clock().now() 
        self.start_ep_time = self.get_clock().now() 
        self.crash = 0
        self.win = False
        self.dummy_req = Empty_Request()
        self.reset = self.create_client(Empty, '/reset_simulation')
        self.reset.call_async(self.dummy_req)

        # parameter for running model
        self.max_episodes = 10001
        self.rewards = []
        self.batch_size = 256
        self.action_dim = 2
        self.state_dim = 24 # TEST
        self.hidden_dim = 500
        self.v_min = 0.0    #m/s
        self.v_max = 0.22   #m/s
        self.w_min = -2.0   #rad/s
        self.w_max = 2.0    #rad/s
        self.replay_buffer_size = 50000
        self.agent = SAC(self.state_dim, self.action_dim)
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.previous_action = np.array([0., 0.])
        self.warm_up = 4
        self.episode = 0
        self.tstep = 0
        self.done = False
        self.current_episode = 0.
        self.diff_angle = 0.
        self.previous_distance = 0.
        self.stop = 0
        self.position = Pose()
        self.prev_position = Pose()
        self.total_rewards = 0
        self.state = []
        self.max_tstep = 300

    def get_state(self, ranges_arr, past_action):
        """
        get current state
        """
        st_list = []
        diff_angle = self.diff_angle
        min_range = 0.136
        max_range = 3.5
        done = False

        # Handle value received from sensor
        for i in range(len(ranges_arr)):
            if ranges_arr[i] == float('Inf') or ranges_arr[i] == float('inf'):
                st_list.append(max_range)
            elif np.isnan(ranges_arr[i]) or ranges_arr[i] == float('nan'):
                st_list.append(0)
            else:
                st_list.append(ranges_arr[i])

        # If the robot close to target
        print('min(st_list)',min(st_list))
        if check_crash(st_list, MIN_RANGE):
            done = True

        # Add previous v and w
        for pa in past_action:
            st_list.append(pa)

        # Check Robot is on goal (win) or not
        self.win, current_distance = check_win(self.position.x, self.position.y, X_GOAL, Y_GOAL, 0.15)

        at_list = []
        at_list.append(self.diff_angle)     # current target angle
        at_list.append(current_distance)    # current distance

        # return st + at
        return st_list + at_list, done

    def get_reward(self, state, done):
        current_distance = state[-1]        # Get current distance from state
        diff_angle = state[-2]              # Get current different angle

        distance_change = self.previous_distance - current_distance

        # Set reward when distance_change
        if distance_change > 0:
            reward = 0.
        else:
            reward = 0.
        
        print("self.position.x",self.position.x)
        print("self.prev_position.x",self.prev_position.x)
        
        print("self.position.y",self.position.y)
        #print("self.prev_position.x",self.prev_position.x)
        print("self.prev_position.y",self.prev_position.y)
        # Prevent Robot is stop
        x_now = round(self.position.x, 3)
        y_now = round(self.position.y, 3)
        x_prev = round(self.prev_position.x, 3)
        y_prev = round(self.prev_position.y, 3)
        if x_now == x_prev and y_now == y_prev:
            self.stop = self.stop + 1
            print('=== ROBOT IS STOP ===')
            # Robot is stop too long --> End training
            print("self.stop",self.stop)
            if self.stop == 20:
                self.stop = 0
                # done = True
        else:
            self.stop = 0

        if done:
            # Reward for win
            if self.win:
                reward = 100
            # Penalty for stop
            else:
                reward = -5
        return reward, done

    def step(self, action, past_action, ranges_arr):
        # get v, and w
        v = action[0]
        w = action[1]

        # Publish v, w to robot
        self.publisher_vel(v,w)

        # get state
        
        state, done = self.get_state(ranges_arr, past_action)
        
        #print(state)
        # get reward from state
        reward, done = self.get_reward(state, done)
        print("reward------>",reward)
        return np.asarray(state), reward, done
        

    def call_back(self):
        # get current data
        x ,y , yaw = self.odom_receive()
        ranges_arr, range_max, angle_max, angle_increment = self.scan_receive()
        ranges_arr = minimum_segmented_lidar(ranges_arr)                            # Segmented lidar
        
        # learning RL function
        self.rl_func(x,y,yaw,ranges_arr, range_max, angle_max, angle_increment)

        self.ep_time = (self.get_clock().now() - self.start_ep_time).nanoseconds / 1e9
        self.crash = check_crash(ranges_arr, MIN_RANGE)
        self.win = check_win(x, y)
        # get_reward(x, y, yaw, self.crash, self.win)
        # print(self.ep_time) #if want to see time
        #print("self.ep_time",self.ep_time)
        #print("self.crash",self.crash)
        #print("self.win",self.win)
        # if self.ep_time >= 15:
        #     print('episode:',self.episode,end='')
        #     print(': time out, reset')
        #     self.reset_world()
        # elif self.crash:
        #     print('episode:',self.episode,end='')
        #     print(': crash, reset')
        #     self.reset_world()
        # elif self.win:
        #     print('episode:',self.episode,end='')
        #     print(': win, reset')
        #     print("-------------------------------------winnn++++++++++++++++++++++++")
        #     self.reset_world()

    def reset_world(self):
        self.crash = 0
        self.win = 0
        self.episode = self.episode + 1
        self.tstep = 0
        self.total_rewards = 0
        self.start_ep_time = self.get_clock().now() 
        self.reset.call_async(self.dummy_req)
        
    def odom_receive(self):
        _,msg_odom=self.wait_for_message('/odom', Odometry)
        
        # Get previous position
        if self.tstep == 0:
            self.prev_position = msg_odom.pose.pose.position
        else:
            self.prev_position = copy.deepcopy(self.position)
        # Get current position
        self.position = msg_odom.pose.pose.position

        x = msg_odom.pose.pose.position.x
        y = msg_odom.pose.pose.position.y
        # z = msg_odom.pose.pose.position.z
        
        orientation_q = msg_odom.pose.pose.orientation
        orientation_list = [ orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w ]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)

        ### ADD FOR CALCULATING GOAL ANGLE ###
        goal_angle = math.atan2(Y_GOAL - y, X_GOAL - x)
        diff_angle = goal_angle - yaw

        # Set angle in range of [-pi, pi]
        if diff_angle > math.pi:
            diff_angle = diff_angle - 2*math.pi
        elif diff_angle < -math.pi:
            diff_angle = diff_angle + 2*math.pi
        
        self.diff_angle = round(diff_angle, 3)
        
        
        return x, y, yaw
        
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
        
    
    def rl_func(self,x,y,yaw,ranges_arr, range_max, angle_max, angle_increment):
        """
        x = float : current position of robot compared to initial position in meter unit(x increase whun go forward, decrase when go backward)
        y = float : current position of robot compared to initial position in meter unit(y increase when go left, decrease when go right)
        yaw = float : rotation of robot in radian unit, increase when rotate counter-clock wise(up to 3.14 when flip 180 degree), switch to -3.14 when rotate greater than 180 degree and approach to 0 when rotate back to initial
        ranges_arr = numpy.ndarray : contain 450 float values of distance each angles in meter unit
        ranges_max = 15 meter, maximum measureable length of lidar sensor
        angle_max = around 6.28 in unit radian
        angle_increment = each steps of angle lidar beam in radian unit
        ------------------------
        v = linear velocity of robot ()
        """

        ### TRAINING IN EACH STEP ###
        if self.tstep == 0:
            self.state, _ = self.get_state(ranges_arr, [0]*self.action_dim)

        self.state = np.float32(self.state)

        # Evaluate model every 10 time steps
        if self.episode%10!=0:
            action = self.agent.select_action(self.state)
        else:
            action = self.agent.select_action(self.state, eval=True)
        
        # Unnormalizaed action
        unnorm_action = np.array([
            action_unnormalized(action[0], self.v_max, self.v_min),     # v
            action_unnormalized(action[1], self.w_max, self.w_min)      # w
            ])
        
        # Do a step
        #print("self.state",self.state)
        next_state, reward, done = self.step(unnorm_action, self.previous_action, ranges_arr)
        #print("next_state",next_state)
        next_state = np.float32(next_state)

        # update previous action
        self.previous_action = copy.deepcopy(action)

        # Summary reward
        self.total_rewards = self.total_rewards + reward
        
        # convert next_state to float32
        if self.episode%10!=0 or len(self.replay_buffer) > self.warm_up * self.batch_size:
            if reward == 100:
                print("MAXIMUM REWARD!")
                # Collect 3 times
                for _ in range(3):
                    self.replay_buffer.push(self.state, action, reward, next_state, done)
            else:
                self.replay_buffer.push(self.state, action, reward, next_state, done)
       
        if self.episode%10!=0 and len(self.replay_buffer) > self.warm_up * self.batch_size:
            self.agent.update_parameters(self.replay_buffer, self.batch_size)
        self.state = copy.deepcopy(next_state)

        # increase step
        print("self.tstep before",self.tstep)
        self.tstep = self.tstep + 1
        print("self.tstep after",self.tstep)

        # Check maximum time step
        if self.tstep >= self.max_tstep:
            done = True

        # update result
        if self.episode%10==0 and len(self.replay_buffer) > self.warm_up * self.batch_size:
            result = self.total_rewards
            self.rewards.append(result)
        
        # save model every 20 episode
        print("self.episode------------",self.episode)
        if self.episode%20==0 and  self.episode !=0 :
            self.agent.save_models(self.episode)

        # reset world if it's done
        print("value done",done)
        if done:
            self.reset_world()
        
        print("TEST: ReplayBuffer length:", len(self.replay_buffer))


    def publisher_vel(self,v,w):
        velMsg = Twist()
        velMsg.linear.x = v
        # velMsg.linear.y = 0.
        # velMsg.linear.z = 0.
        # velMsg.angular.x = 0.
        # velMsg.angular.y = 0.
        velMsg.angular.z = w
        self.velPub.publish(velMsg)


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


def main():
    rclpy.init()
    ln_obj=LearningNode()
    rclpy.spin(ln_obj)
    
    ln_obj.destroy_node()
    rclpy.shutdown()



if __name__=='__main__':
    main()
