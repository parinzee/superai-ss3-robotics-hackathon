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
from copy import deepcopy
import itertools

### ============= INITIAL RARAMETERS ============= ###

X_INIT          = 0.
Y_INIT          = 0.
THETA_INIT      = 0.
X_GOAL          = 0.542760
Y_GOAL          = 0.520140
SENSOR_SECTION  = 60
MIN_RANGE       = 0.136
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DIS_START_GOAL = np.sqrt((X_INIT - X_GOAL)**2 + (Y_INIT - Y_GOAL)**2)
SEED = 42

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

### Neural Network Class ###
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for agents.
    """

    def __init__(self, obs_dim, act_dim, max_size):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_size = max_size
        self.obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(max_size, dtype=np.float32)
        self.done_buf = np.zeros(max_size, dtype=np.float32)
        self.ptr, self.size = 0, 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.obs2_buf[self.ptr] = list(next_obs)
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

    def sample_batch_with_history(self, batch_size=32, max_hist_len=100):
        """
        :param batch_size:
        :param max_hist_len: the length of experiences before current experience
        :return:
        """
        idxs = np.random.randint(max_hist_len, self.size, size=batch_size)
        # History
        if max_hist_len == 0:
            hist_obs = np.zeros([batch_size, 1, self.obs_dim])
            hist_act = np.zeros([batch_size, 1, self.act_dim])
            hist_obs2 = np.zeros([batch_size, 1, self.obs_dim])
            hist_act2 = np.zeros([batch_size, 1, self.act_dim])
            hist_obs_len = np.zeros(batch_size)
            hist_obs2_len = np.zeros(batch_size)
        else:
            hist_obs = np.zeros([batch_size, max_hist_len, self.obs_dim])
            hist_act = np.zeros([batch_size, max_hist_len, self.act_dim])
            hist_obs_len = max_hist_len * np.ones(batch_size)
            hist_obs2 = np.zeros([batch_size, max_hist_len, self.obs_dim])
            hist_act2 = np.zeros([batch_size, max_hist_len, self.act_dim])
            hist_obs2_len = max_hist_len * np.ones(batch_size)

            # Extract history experiences before sampled index
            for i, id in enumerate(idxs):
                hist_start_id = id - max_hist_len
                if hist_start_id < 0:
                    hist_start_id = 0
                # If exist done before the last experience (not including the done in id), start from the index next to the done.
                if len(np.where(self.done_buf[hist_start_id:id] == 1)[0]) != 0:
                    hist_start_id = hist_start_id + (np.where(self.done_buf[hist_start_id:id] == 1)[0][-1]) + 1
                hist_seg_len = id - hist_start_id
                hist_obs_len[i] = hist_seg_len
                hist_obs[i, :hist_seg_len, :] = self.obs_buf[hist_start_id:id]
                hist_act[i, :hist_seg_len, :] = self.act_buf[hist_start_id:id]
                # If the first experience of an episode is sampled, the hist lengths are different for obs and obs2.
                if hist_seg_len == 0:
                    hist_obs2_len[i] = 1
                else:
                    hist_obs2_len[i] = hist_seg_len
                hist_obs2[i, :hist_seg_len, :] = self.obs2_buf[hist_start_id:id]
                hist_act2[i, :hist_seg_len, :] = self.act_buf[hist_start_id+1:id+1]

        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     hist_obs=hist_obs,
                     hist_act=hist_act,
                     hist_obs2=hist_obs2,
                     hist_act2=hist_act2,
                     hist_obs_len=hist_obs_len,
                     hist_obs2_len=hist_obs2_len)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, act_dim,
                 mem_pre_lstm_hid_sizes=(128,),
                 mem_lstm_hid_sizes=(128,),
                 mem_after_lstm_hid_size=(128,),
                 cur_feature_hid_sizes=(128,),
                 post_comb_hid_sizes=(128,),
                 hist_with_past_act=False):
        super(MLPCritic, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hist_with_past_act = hist_with_past_act
        #
        self.mem_pre_lstm_layers = nn.ModuleList()
        self.mem_lstm_layers = nn.ModuleList()
        self.mem_after_lstm_layers = nn.ModuleList()

        self.cur_feature_layers = nn.ModuleList()
        self.post_combined_layers = nn.ModuleList()
        # Memory
        #    Pre-LSTM
        if self.hist_with_past_act:
            mem_pre_lstm_layer_size = [obs_dim + act_dim] + list(mem_pre_lstm_hid_sizes)
        else:
            mem_pre_lstm_layer_size = [obs_dim] + list(mem_pre_lstm_hid_sizes)
        for h in range(len(mem_pre_lstm_layer_size) - 1):
            self.mem_pre_lstm_layers += [nn.Linear(mem_pre_lstm_layer_size[h],
                                                   mem_pre_lstm_layer_size[h + 1]),
                                         nn.ReLU()]
        #    LSTM
        self.mem_lstm_layer_sizes = [mem_pre_lstm_layer_size[-1]] + list(mem_lstm_hid_sizes)
        for h in range(len(self.mem_lstm_layer_sizes) - 1):
            self.mem_lstm_layers += [
                nn.LSTM(self.mem_lstm_layer_sizes[h], self.mem_lstm_layer_sizes[h + 1], batch_first=True)]

        #   After-LSTM
        self.mem_after_lstm_layer_size = [self.mem_lstm_layer_sizes[-1]] + list(mem_after_lstm_hid_size)
        for h in range(len(self.mem_after_lstm_layer_size)-1):
            self.mem_after_lstm_layers += [nn.Linear(self.mem_after_lstm_layer_size[h],
                                                     self.mem_after_lstm_layer_size[h+1]),
                                           nn.ReLU()]

        # Current Feature Extraction
        cur_feature_layer_size = [obs_dim + act_dim] + list(cur_feature_hid_sizes)
        for h in range(len(cur_feature_layer_size) - 1):
            self.cur_feature_layers += [nn.Linear(cur_feature_layer_size[h], cur_feature_layer_size[h + 1]),
                                        nn.ReLU()]

        # Post-Combination
        post_combined_layer_size = [self.mem_after_lstm_layer_size[-1] + cur_feature_layer_size[-1]] + list(
            post_comb_hid_sizes) + [1]
        for h in range(len(post_combined_layer_size) - 2):
            self.post_combined_layers += [nn.Linear(post_combined_layer_size[h], post_combined_layer_size[h + 1]),
                                          nn.ReLU()]
        self.post_combined_layers += [nn.Linear(post_combined_layer_size[-2], post_combined_layer_size[-1]),
                                      nn.Identity()]

    def forward(self, obs, act, hist_obs, hist_act, hist_seg_len):
        #
        tmp_hist_seg_len = deepcopy(hist_seg_len)
        tmp_hist_seg_len[hist_seg_len == 0] = 1
        if self.hist_with_past_act:
            x = torch.cat([hist_obs, hist_act], dim=-1)
        else:
            x = hist_obs
        # Memory
        #    Pre-LSTM
        for layer in self.mem_pre_lstm_layers:
            x = layer(x)
        #    LSTM
        for layer in self.mem_lstm_layers:
            x, (lstm_hidden_state, lstm_cell_state) = layer(x)
        #    After-LSTM
        for layer in self.mem_after_lstm_layers:
            x = layer(x)
        #    History output mask to reduce disturbance cased by none history memory
        hist_out = torch.gather(x, 1,
                                (tmp_hist_seg_len - 1).view(-1, 1).repeat(1, self.mem_after_lstm_layer_size[-1]).unsqueeze(
                                    1).long()).squeeze(1)

        # Current Feature Extraction
        x = torch.cat([obs, act], dim=-1)
        for layer in self.cur_feature_layers:
            x = layer(x)

        # Post-Combination
        extracted_memory = hist_out
        x = torch.cat([extracted_memory, x], dim=-1)

        for layer in self.post_combined_layers:
            x = layer(x)
        # squeeze(x, -1) : critical to ensure q has right shape.
        return torch.squeeze(x, -1), extracted_memory


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit,
                 mem_pre_lstm_hid_sizes=(128,),
                 mem_lstm_hid_sizes=(128,),
                 mem_after_lstm_hid_size=(128,),
                 cur_feature_hid_sizes=(128,),
                 post_comb_hid_sizes=(128,),
                 hist_with_past_act=False):
        super(MLPActor, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.hist_with_past_act = hist_with_past_act
        #
        self.mem_pre_lstm_layers = nn.ModuleList()
        self.mem_lstm_layers = nn.ModuleList()
        self.mem_after_lstm_layers = nn.ModuleList()

        self.cur_feature_layers = nn.ModuleList()
        self.post_combined_layers = nn.ModuleList()

        # Memory
        #    Pre-LSTM
        if self.hist_with_past_act:
            mem_pre_lstm_layer_size = [obs_dim + act_dim] + list(mem_pre_lstm_hid_sizes)
        else:
            mem_pre_lstm_layer_size = [obs_dim] + list(mem_pre_lstm_hid_sizes)
        for h in range(len(mem_pre_lstm_layer_size) - 1):
            self.mem_pre_lstm_layers += [nn.Linear(mem_pre_lstm_layer_size[h],
                                                   mem_pre_lstm_layer_size[h + 1]),
                                         nn.ReLU()]
        #    LSTM
        self.mem_lstm_layer_sizes = [mem_pre_lstm_layer_size[-1]] + list(mem_lstm_hid_sizes)
        for h in range(len(self.mem_lstm_layer_sizes) - 1):
            self.mem_lstm_layers += [
                nn.LSTM(self.mem_lstm_layer_sizes[h], self.mem_lstm_layer_sizes[h + 1], batch_first=True)]
        #   After-LSTM
        self.mem_after_lstm_layer_size = [self.mem_lstm_layer_sizes[-1]] + list(mem_after_lstm_hid_size)
        for h in range(len(self.mem_after_lstm_layer_size) - 1):
            self.mem_after_lstm_layers += [nn.Linear(self.mem_after_lstm_layer_size[h],
                                                     self.mem_after_lstm_layer_size[h + 1]),
                                           nn.ReLU()]

        # Current Feature Extraction
        cur_feature_layer_size = [obs_dim] + list(cur_feature_hid_sizes)
        for h in range(len(cur_feature_layer_size) - 1):
            self.cur_feature_layers += [nn.Linear(cur_feature_layer_size[h], cur_feature_layer_size[h + 1]),
                                        nn.ReLU()]

        # Post-Combination
        post_combined_layer_size = [self.mem_after_lstm_layer_size[-1] + cur_feature_layer_size[-1]] + list(
            post_comb_hid_sizes) + [act_dim]
        for h in range(len(post_combined_layer_size) - 2):
            self.post_combined_layers += [nn.Linear(post_combined_layer_size[h], post_combined_layer_size[h + 1]),
                                          nn.ReLU()]
        self.post_combined_layers += [nn.Linear(post_combined_layer_size[-2], post_combined_layer_size[-1]), nn.Tanh()]

    def forward(self, obs, hist_obs, hist_act, hist_seg_len):
        #
        tmp_hist_seg_len = deepcopy(hist_seg_len)
        tmp_hist_seg_len[hist_seg_len == 0] = 1
        if self.hist_with_past_act:
            x = torch.cat([hist_obs, hist_act], dim=-1)
        else:
            x = hist_obs
        # Memory
        #    Pre-LSTM
        for layer in self.mem_pre_lstm_layers:
            x = layer(x)
        #    LSTM
        for layer in self.mem_lstm_layers:
            x, (lstm_hidden_state, lstm_cell_state) = layer(x)
        #    After-LSTM
        for layer in self.mem_after_lstm_layers:
            x = layer(x)
        hist_out = torch.gather(x, 1,
                                (tmp_hist_seg_len - 1).view(-1, 1).repeat(1, self.mem_after_lstm_layer_size[-1]).unsqueeze(
                                    1).long()).squeeze(1)

        # Current Feature Extraction
        x = obs
        for layer in self.cur_feature_layers:
            x = layer(x)

        # Post-Combination
        extracted_memory = hist_out
        x = torch.cat([extracted_memory, x], dim=-1)

        for layer in self.post_combined_layers:
            x = layer(x)
        return self.act_limit * x, extracted_memory


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit=1,
                 critic_mem_pre_lstm_hid_sizes=(128,),
                 critic_mem_lstm_hid_sizes=(128,),
                 critic_mem_after_lstm_hid_size=(128,),
                 critic_cur_feature_hid_sizes=(128,),
                 critic_post_comb_hid_sizes=(128,),
                 critic_hist_with_past_act=False,
                 actor_mem_pre_lstm_hid_sizes=(128,),
                 actor_mem_lstm_hid_sizes=(128,),
                 actor_mem_after_lstm_hid_size=(128,),
                 actor_cur_feature_hid_sizes=(128,),
                 actor_post_comb_hid_sizes=(128,),
                 actor_hist_with_past_act=False):
        super(MLPActorCritic, self).__init__()
        self.q1 = MLPCritic(obs_dim, act_dim,
                            mem_pre_lstm_hid_sizes=critic_mem_pre_lstm_hid_sizes,
                            mem_lstm_hid_sizes=critic_mem_lstm_hid_sizes,
                            mem_after_lstm_hid_size=critic_mem_after_lstm_hid_size,
                            cur_feature_hid_sizes=critic_cur_feature_hid_sizes,
                            post_comb_hid_sizes=critic_post_comb_hid_sizes,
                            hist_with_past_act=critic_hist_with_past_act)
        self.q2 = MLPCritic(obs_dim, act_dim,
                            mem_pre_lstm_hid_sizes=critic_mem_pre_lstm_hid_sizes,
                            mem_lstm_hid_sizes=critic_mem_lstm_hid_sizes,
                            mem_after_lstm_hid_size=critic_mem_after_lstm_hid_size,
                            cur_feature_hid_sizes=critic_cur_feature_hid_sizes,
                            post_comb_hid_sizes=critic_post_comb_hid_sizes,
                            hist_with_past_act=critic_hist_with_past_act)
        self.pi = MLPActor(obs_dim, act_dim, act_limit,
                           mem_pre_lstm_hid_sizes=actor_mem_pre_lstm_hid_sizes,
                           mem_lstm_hid_sizes=actor_mem_lstm_hid_sizes,
                           mem_after_lstm_hid_size=actor_mem_after_lstm_hid_size,
                           cur_feature_hid_sizes=actor_cur_feature_hid_sizes,
                           post_comb_hid_sizes=actor_post_comb_hid_sizes,
                           hist_with_past_act=actor_hist_with_past_act)

    def act(self, obs, hist_obs=None, hist_act=None, hist_seg_len=None, device=None):
        if (hist_obs is None) or (hist_act is None) or (hist_seg_len is None):
            hist_obs = torch.zeros(1, 1, self.obs_dim).to(device)
            hist_act = torch.zeros(1, 1, self.act_dim).to(device)
            hist_seg_len = torch.zeros(1).to(device)
        with torch.no_grad():
            act, _, = self.pi(obs, hist_obs, hist_act, hist_seg_len)
            return act.cpu().numpy()


### ============================================== ###


class LearningNode(Node):

    def __init__(self):
        super().__init__('test2')
        print("=== Creating Model ===")
        self.timer_period = .25 # seconds
        self.timer = self.create_timer(self.timer_period, self.call_back)
        self.velPub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.setPosPub = self.create_publisher(ModelState, 'gazebo/set_model_state', 10)
        self.ep_time = self.get_clock().now() 
        self.start_ep_time = self.get_clock().now() 
        self.dummy_req = Empty_Request()
        self.reset = self.create_client(Empty, '/reset_simulation')
        self.reset.call_async(self.dummy_req)

        # Set Seed
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        print("Running with seed: ", SEED)

        # Env Settings
        self.obs_dim = SENSOR_SECTION + 2 # LIDAR + distance + angle
        self.act_dim = 2 # v, w
        self.act_limit = 1
        self.v_max = 0.6
        self.w_max = 8
        print("obs_dim: ", self.obs_dim)
        print("act_dim: ", self.act_dim)
        print("act_limit: ", self.act_limit)

        # Model Hyper Parameters
        self.steps_per_epoch=1000
        self.epochs=100 
        self.replay_size=int(5e6)
        self.gamma=0.99
        self.polyak=0.995
        self.pi_lr=1e-3
        self.q_lr=1e-3
        self.start_steps=10000
        self.update_after=1000
        self.update_every=50
        self.act_noise=0.1
        self.target_noise=0.2
        self.noise_clip=0.5
        self.policy_delay=2
        self.num_test_episodes=10
        self.max_ep_len=1000
        self.batch_size=100
        self.max_hist_len=100
        self.flicker_prob=0.2
        self.random_noise_sigma=0.1
        self.random_sensor_missing_prob=0.1
        self.use_double_critic = True
        self.use_target_policy_smooth = True
        self.critic_mem_pre_lstm_hid_sizes=(128,)
        self.critic_mem_lstm_hid_sizes=(128,)
        self.critic_mem_after_lstm_hid_size=(128,)
        self.critic_cur_feature_hid_sizes=(128,)
        self.critic_post_comb_hid_sizes=(128,)
        self.critic_hist_with_past_act=False
        self.actor_mem_pre_lstm_hid_sizes=(128,)
        self.actor_mem_lstm_hid_sizes=(128,)
        self.actor_mem_after_lstm_hid_size=(128,)
        self.actor_cur_feature_hid_sizes=(128,)
        self.actor_post_comb_hid_sizes=(128,)
        self.actor_hist_with_past_act=False
        self.save_freq=1
        self.total_steps = self.steps_per_epoch * self.epochs

        # Data Variables
        self.all_n_steps = 0
        self.tstep = 0
        self.curr_epoch = 0
        self.done = False
        self.crash = False
        self.win = False
        self.diff_angle = 0.


        # Initalize Model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ac = MLPActorCritic(self.obs_dim, self.act_dim, self.act_limit,
                        critic_mem_pre_lstm_hid_sizes=self.critic_mem_pre_lstm_hid_sizes,
                        critic_mem_lstm_hid_sizes=self.critic_mem_lstm_hid_sizes,
                        critic_mem_after_lstm_hid_size=self.critic_mem_after_lstm_hid_size,
                        critic_cur_feature_hid_sizes=self.critic_cur_feature_hid_sizes,
                        critic_post_comb_hid_sizes=self.critic_post_comb_hid_sizes,
                        critic_hist_with_past_act=self.critic_hist_with_past_act,
                        actor_mem_pre_lstm_hid_sizes=self.actor_mem_pre_lstm_hid_sizes,
                        actor_mem_lstm_hid_sizes=self.actor_mem_lstm_hid_sizes,
                        actor_mem_after_lstm_hid_size=self.actor_mem_after_lstm_hid_size,
                        actor_cur_feature_hid_sizes=self.actor_cur_feature_hid_sizes,
                        actor_post_comb_hid_sizes=self.actor_post_comb_hid_sizes,
                        actor_hist_with_past_act=self.actor_hist_with_past_act
        )
        self.ac_targ = deepcopy(self.ac)
        self.ac.to(self.device)
        self.ac_targ.to(self.device)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, max_size=self.replay_size)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.q_optimizer = Adam(self.q_params, lr=self.q_lr)


    def get_state(self, ranges_arr, past_action):
        """
        get current state
        """
        st_list = []
        # diff_angle = self.diff_angle
        # min_range = 0.136
        max_range = 3.5

        # Handle value received from sensor
        for i in range(len(ranges_arr)):
            if ranges_arr[i] == float('Inf') or ranges_arr[i] == float('inf'):
                st_list.append(max_range)
            elif np.isnan(ranges_arr[i]) or ranges_arr[i] == float('nan'):
                st_list.append(0)
            else:
                st_list.append(ranges_arr[i])

        # If the robot close to target
        print('min(st_list)', min(st_list))
        if check_crash(st_list, MIN_RANGE):
            self.done = True

        # Add previous v and w
        for pa in past_action:
            st_list.append(pa)

        # Check Robot is on goal (win) or not
        self.win, current_distance = check_win(self.position.x, self.position.y, X_GOAL, Y_GOAL, 0.15)

        at_list = []
        at_list.append(self.diff_angle)     # current target angle
        at_list.append(current_distance)    # current distance

        # return st + at
        return st_list + at_list

    def get_reward(self, state):
        current_distance = state[-1]        # Get current distance from state
        diff_angle = state[-2]              # Get current different angle

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

        if self.done:
            # Reward for win
            if self.win and not self.crash:
                print('===!! ROBOT WON !!===')
                reward = 100
            
            elif self.crash:
                print('===!! ROBOT CRASH !!===')
                reward = -100

            # Penalty for stop
            else:
                reward = -5

        return reward

    def step(self, action, past_action, ranges_arr):
        # get v, and w
        v = action[0]
        w = action[1]

        # Publish v, w to robot
        self.publisher_vel(v,w)

        # get state
        state = self.get_state(ranges_arr, past_action)
        
        #print(state)
        # get reward from state
        reward = self.get_reward(state)
        print("reward------>",reward)
        return np.asarray(state), reward 
        

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
    
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        h_o, h_a, h_o2, h_a2, h_o_len, h_o2_len = data['hist_obs'], data['hist_act'], data['hist_obs2'], data['hist_act2'], data['hist_obs_len'], data['hist_obs2_len']

        q1, q1_extracted_memory = self.ac.q1(o, a, h_o, h_a, h_o_len)
        q2, q2_extracted_memory = self.ac.q2(o, a, h_o, h_a, h_o_len)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ, _ = self.ac_targ.pi(o2, h_o2, h_a2, h_o2_len)

            # Target policy smoothing
            if self.use_target_policy_smooth:
                epsilon = torch.randn_like(pi_targ) * self.target_noise
                epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
                a2 = pi_targ + epsilon
                a2 = torch.clamp(a2, -self.act_limit, self.act_limit)
            else:
                a2 = pi_targ

            # Target Q-values
            q1_pi_targ, _ = self.ac_targ.q1(o2, a2, h_o2, h_a2, h_o2_len)
            q2_pi_targ, _ = self.ac_targ.q2(o2, a2, h_o2, h_a2, h_o2_len)

            if self.use_double_critic:
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            else:
                q_pi_targ = q1_pi_targ
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()

        if self.use_double_critic:
            loss_q = loss_q1 + loss_q2
        else:
            loss_q = loss_q1

        # Useful info for logging
        # import pdb; pdb.set_trace()
        loss_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                         Q2Vals=q2.detach().cpu().numpy(),
                         Q1ExtractedMemory=q1_extracted_memory.mean(dim=1).detach().cpu().numpy(),
                         Q2ExtractedMemory=q2_extracted_memory.mean(dim=1).detach().cpu().numpy())

        return loss_q, loss_info


        # Set up function for computing TD3 pi loss
    def compute_loss_pi(self, data):
        o, h_o, h_a, h_o_len = data['obs'], data['hist_obs'], data['hist_act'], data['hist_obs_len']
        a, a_extracted_memory = self.ac.pi(o, h_o, h_a, h_o_len)
        q1_pi, _ = self.ac.q1(o, a, h_o, h_a, h_o_len)
        loss_info = dict(ActExtractedMemory=a_extracted_memory.mean(dim=1).detach().cpu().numpy())
        return -q1_pi.mean(), loss_info


    def update(self, data, timer):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Possibly update pi and target networks
        if timer % self.policy_delay == 0:
            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in self.q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            self.pi_optimizer.zero_grad()
            loss_pi, loss_info_pi = self.compute_loss_pi(data)
            loss_pi.backward()
            self.pi_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in self.q_params:
                p.requires_grad = True

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)
    
    def get_action(self, o, o_buff, a_buff, o_buff_len, noise_scale, device=None):
        h_o = torch.tensor(o_buff).view(1, o_buff.shape[0], o_buff.shape[1]).float().to(device)
        h_a = torch.tensor(a_buff).view(1, a_buff.shape[0], a_buff.shape[1]).float().to(device)
        h_l = torch.tensor([o_buff_len]).float().to(device)
        with torch.no_grad():
            a = self.ac.act(torch.as_tensor(o, dtype=torch.float32).view(1, -1).to(device),
                       h_o, h_a, h_l).reshape(self.act_dim)
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)
    
    # TODO: FIX ENVIRONMENT
    # def test_agent(self.):
    #     for j in range(self.num_test_episodes):
    #         o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0

    #         if max_hist_len > 0:
    #             o_buff = np.zeros([max_hist_len, obs_dim])
    #             a_buff = np.zeros([max_hist_len, act_dim])
    #             o_buff[0, :] = o
    #             o_buff_len = 0
    #         else:
    #             o_buff = np.zeros([1, obs_dim])
    #             a_buff = np.zeros([1, act_dim])
    #             o_buff_len = 0

    #         while not (d or (ep_len == max_ep_len)):
    #             # Take deterministic actions at test time (noise_scale=0)
    #             a = get_action(o, o_buff, a_buff, o_buff_len, 0, device)
    #             o2, r, d, _ = test_env.step(a)

    #             ep_ret += r
    #             ep_len += 1
    #             # Add short history
    #             if max_hist_len != 0:
    #                 if o_buff_len == max_hist_len:
    #                     o_buff[:max_hist_len - 1] = o_buff[1:]
    #                     a_buff[:max_hist_len - 1] = a_buff[1:]
    #                     o_buff[max_hist_len - 1] = list(o)
    #                     a_buff[max_hist_len - 1] = list(a)
    #                 else:
    #                     o_buff[o_buff_len + 1 - 1] = list(o)
    #                     a_buff[o_buff_len + 1 - 1] = list(a)
    #                     o_buff_len += 1
    #             o = o2



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
            self.state, _ = self.get_state(ranges_arr, [0] * self.action_dim)
            self.ep_ret = 0
            self.ep_len = 0
            self.previous_action = np.float32([0, 0])
            if self.max_hist_len > 0:
                o_buff = np.zeros([self.max_hist_len, self.obs_dim])
                a_buff = np.zeros([self.max_hist_len, self.act_dim])
                o_buff[0, :] = self.state
                o_buff_len = 0
            else:
                o_buff = np.zeros([1, self.obs_dim])
                a_buff = np.zeros([1, self.act_dim])
                o_buff_len = 0
        
        # Randomly draw actions until we get over start steps
        if self.all_n_steps > self.start_steps:
            print("Getting Action From Model...")
            a = self.get_action(self.state, o_buff, a_buff, o_buff_len, self.act_noise, self.device)

        else:
            # v, w
            print("Randomly Sampling Actions...")
            a = np.array([np.random(-1, 1), np.random(-1, 1)])

        self.state = np.float32(self.state)

        # Unnormalizaed action
        unnorm_action = np.array([
            action_unnormalized(a[0], self.v_max, self.v_min),     # v
            action_unnormalized(a[1], self.w_max, self.w_min)      # w
        ])
        
        next_state, reward = self.step(unnorm_action, self.previous_action, ranges_arr)

        self.ep_ret += reward
        self.ep_len += 1
        self.all_n_steps += 1

        next_state = np.float32(next_state)

        # update previous action
        self.previous_action = copy.deepcopy(a)

        # Summary reward
        self.total_rewards = self.total_rewards + reward
        
        self.replay_buffer.store(self.state, a, reward, next_state, self.done)

        # Add short history
        if self.max_hist_len != 0:
            if o_buff_len == self.max_hist_len:
                o_buff[:self.max_hist_len - 1] = o_buff[1:]
                a_buff[:self.max_hist_len - 1] = a_buff[1:]
                o_buff[self.max_hist_len - 1] = list(self.state)
                a_buff[self.max_hist_len - 1] = list(a)

            else:
                o_buff[o_buff_len + 1 - 1] = list(self.state)
                a_buff[o_buff_len + 1 - 1] = list(a)
                o_buff_len += 1

        self.state = copy.deepcopy(next_state)

        # increase step
        self.tstep = self.tstep + 1

        print(f"Step: {self.tstep}, Reward: {reward}, Total Reward: {self.total_rewards}, Episode Length: {self.ep_len}, Episode Return: {self.ep_ret}")

        # Check maximum time step
        if self.tstep > self.steps_per_epoch:
            self.done = True

        if self.tstep >= self.update_after and self.t % self.update_every == 0:
            print("Updating Model...")
            for j in range(self.update_every):
                batch = self.replay_buffer.sample_batch_with_history(self.batch_size, self.max_hist_len)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.update(data=batch, timer=j)
        
        if self.done:
            print(f"Done — Episode: {self.episode}, Total Reward: {self.total_rewards}, Episode Length: {self.ep_len}, Episode Return: {self.ep_ret}")
            print(f"Crash: {self.crash}")
            self.reset_world()
            self.episode += 1


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
