# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:51:36 2020

@author: liangyu

Create the agent for a UE
"""

import copy
import numpy as np
from numpy import pi
from collections import namedtuple
from random import random, uniform, choice, randrange, sample
import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from scenario import Scenario, BS

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))  # Define a transition tuple

class ReplayMemory(object):    # Define a replay memory

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def Push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def Sample(self, batch_size):
        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DNN(nn.Module):  # Define a deep neural network

    def __init__(self, opt, sce, scenario):  # Define the layers of the fully-connected hidden network
        super(DNN, self).__init__()
        self.input_layer = nn.Linear(opt.nagents, 64)
        self.middle1_layer = nn.Linear(64, 32)
        self.middle2_layer = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, scenario.BS_Number() * sce.nChannel)
		
    def forward(self, state):  # Define the neural network forward function
        x1 = F.relu(self.input_layer(state))
        x2 = F.relu(self.middle1_layer(x1))
        x3 = F.relu(self.middle2_layer(x2))
        out = self.output_layer(x3)
        return out
        

class Agent:  # Define the agent (UE)
    
    def __init__(self, opt, sce, scenario, index, device):  # Initialize the agent (UE)
        self.opt = opt
        self.sce = sce
        self.id = index
        self.device = device
        self.location = self.Set_Location(scenario)
        self.memory = ReplayMemory(opt.capacity)
        self.model_policy = DNN(opt, sce, scenario)
        self.model_target = DNN(opt, sce, scenario)
        self.model_target.load_state_dict(self.model_policy.state_dict())
        self.model_target.eval()
        self.optimizer = optim.RMSprop(params=self.model_policy.parameters(), lr=opt.learningrate, momentum=opt.momentum)

    def Set_Location(self, scenario):  # Initialize the location of the agent
        Loc_RRU, _ = scenario.BS_Location()
        Loc_agent = np.zeros(2)
        LocM = choice(Loc_RRU)
        r = self.sce.rRRU*random()
        theta = uniform(-pi,pi)
        Loc_agent[0] = LocM[0] + r*np.cos(theta)
        Loc_agent[1] = LocM[1] + r*np.sin(theta) 
        return Loc_agent
    
    def Get_Location(self):
        return self.location
     
    def Select_Action(self, state, scenario, eps_threshold):   # Select action for a user based on the network state
        L = scenario.BS_Number()  # The total number of BSs
        K = self.sce.nChannel  # The total number of channels
        #P = self.sce.nP # The total num of Power
        sample = random()       
        if sample < eps_threshold:  # epsilon-greeedy policy
            with torch.no_grad():
                Q_value = self.model_policy(state)   # Get the Q_value from DNN
                action = Q_value.max(0)[1].view(1,1)
        else:           
            action = torch.tensor([[randrange(L*K)]], dtype=torch.long)
        return action      


    def Get_Gain(self, action, action_i, scenario):
        BS = scenario.Get_BaseStations()
        L = scenario.BS_Number()  # The total number of BSs
        K = self.sce.nChannel  # The total number of channels

        BS_selected = action_i // K
        Ch_selected = action_i % K  # Translate to the selected BS and channel based on the selected action index
        Loc_diff = BS[BS_selected].Get_Location() - self.location
        distance = np.sqrt((Loc_diff[0] ** 2 + Loc_diff[1] ** 2))  # Calculate the distance between BS and UE
        temp_Rx_power, h = BS[BS_selected].temp_Receive_Power(distance)  # Calculate the received power
        InterInterference = 0  # Initialize the Inter Interference
        for i in range(self.opt.nagents):  # Calculate InterInterference
            BS_select_i = action[i] // K
            Ch_select_i = action[i] % K  # The choice of other users
            if BS_select_i != BS_selected:
                if Ch_select_i == Ch_selected:
                    Loc_diff_i = BS[BS_select_i].Get_Location() - self.location
                    distance_i = np.sqrt((Loc_diff_i[0] ** 2 + Loc_diff_i[1] ** 2))
                    temp_Rx_power_i, h_i = BS[BS_select_i].temp_Receive_Power(distance_i)
                    InterInterference += temp_Rx_power_i  # Record all the InterInterference
        Noise = 10 ** ((self.sce.N0) / 10) * self.sce.BW  # Calculate the noise
        g = h / (InterInterference + Noise)  # Calculate the equivalent channel gain
        return g

    def Get_Proportion(self, action, action_i, state, scenario, ue, g):  # Get reward for the state-action pair
        BS = scenario.Get_BaseStations()
        L = scenario.BS_Number()  # The total number of BSs
        K = self.sce.nChannel  # The total number of channels

        BS_selected = action_i // K
        Ch_selected = action_i % K  # Translate to the selected BS and channel based on the selected action index
        IntraNum = 0 # Initialize the number of Intra Interference users
        sumg = 0    # Initialize the sum of channel gain
        proportion = 1
        for i in range(self.opt.nagents):   # Calculate the number of Intra Interference users
            BS_select_i = action[i] // K
            Ch_select_i = action[i] % K   # The choice of other users
            if BS_select_i == BS_selected:
                if Ch_select_i == Ch_selected:
                    IntraNum += 1
                    sumg += g[i]
        for i in range(self.opt.nagents):   # Calculate the Power proportion of Intra Interference users
            BS_select_i = action[i] // K
            Ch_select_i = action[i] % K   # The choice of other users
            if BS_select_i == BS_selected:
                if Ch_select_i == Ch_selected:
                    if i!=ue:
                        proportion = (1 / (IntraNum - 1)) * ((sumg - g[ue]) / sumg)  # Calculate the proportion of Tranmit_Power of each user
        return proportion   #  已经知道了该用户的功率比例关系


    def Get_Reward(self, action, action_i, state, scenario, ue, g, proportion, nstep):  # Get reward for the state-action pair
        BS = scenario.Get_BaseStations()
        L = scenario.BS_Number()  # The total number of BSs
        K = self.sce.nChannel  # The total number of channels

        BS_selected = action_i // K
        Ch_selected = action_i % K  # Translate to the selected BS and channel based on the selected action index
        Loc_diff = BS[BS_selected].Get_Location() - self.location
        distance = np.sqrt((Loc_diff[0]**2 + Loc_diff[1]**2))  # Calculate the distance between BS and UE
        Rx_power, h = BS[BS_selected].temp_Receive_Power(distance)  # Calculate the received power
        IntraInterference = 0  # Initialize the Intra Interference
        InterInterference = 0  # Initialize the Inter Interference

        if Rx_power == 0.0:
            reward = self.sce.negative_cost  # Out of range of the selected BS, thus obtain a negative reward
            Rate = 0
            QoS = 0  # Definitely, QoS cannot be satisfied
        else:                    # If inside the coverage, then we will calculate the reward value
            for i in range(self.opt.nagents):  # Calculate InterInterference
                BS_select_i = action[i] // K
                Ch_select_i = action[i] % K  # The choice of other users
                if BS_select_i != BS_selected:
                    if Ch_select_i == Ch_selected:
                        Loc_diff_i = BS[BS_select_i].Get_Location() - self.location
                        distance_i = np.sqrt((Loc_diff_i[0] ** 2 + Loc_diff_i[1] ** 2))
                        Rx_power_i, h_i = BS[BS_select_i].Receive_Power(distance_i, proportion[i])
                        InterInterference += Rx_power_i  # Record all the InterInterference
                else:
                    if Ch_select_i == Ch_selected:
                        if g[i] > g[ue]:   # 遍历所有用户时判断g是否大于所分析用户，只取大于所分析用户的g值用户的功率作为IntraInterference
                            Loc_diff_i = BS[BS_select_i].Get_Location() - self.location
                            distance_i = np.sqrt((Loc_diff_i[0] ** 2 + Loc_diff_i[1] ** 2))
                            Rx_power_i, h_i = BS[BS_select_i].Receive_Power(distance_i, proportion[i])  # 得到信道增益更大的用户实际分配发送功率比例作为该用户的干扰功率大小之一
                            IntraInterference += Rx_power_i

            Noise = 10 ** ((self.sce.N0) / 10) * self.sce.BW  # Calculate the noise
            SINR = Rx_power / (IntraInterference + InterInterference + Noise)  # Calculate the SINR
            Rate = self.sce.BW * np.log2(1 + SINR) / (10**6)      # Calculate the rate of UE

            if state[0, ue] == 0:  # Calculate the reward of user in slice1
                TargetRate1 = self.sce.Target_Rate1 # Calculate the target rate of UE
                # if SINR >= 10**(self.sce.QoS_thr1/10):
                if Rate >= TargetRate1:
                    QoS = 1
                    reward = 1
                else:
                    QoS = 0
                    """reward = self.sce.negative_cost"""
                    profit = Rate / TargetRate1
                    """profit = self.sce.profit * Rate"""
                    Tx_power_dBm = BS[BS_selected].temp_Transmit_Power_dBm()   # Calculate the transmit power of the selected BS
                    #cost = self.sce.power_cost * Tx_power_dBm + self.sce.action_cost * nstep  # Calculate the total cost
                    reward = profit #- cost
            else:   #  Calculate the reward of user in slice2
                TargetRate2 = self.sce.Target_Rate2
                # if SINR >= 10**(self.sce.QoS_thr2/10):
                if Rate >= TargetRate2:
                    QoS = 1
                    reward = 1
                else:
                    QoS = 0
                    """reward = self.sce.negative_cost"""
                    profit = Rate / TargetRate2
                    """profit = self.sce.profit * Rate"""
                    Tx_power_dBm = BS[BS_selected].temp_Transmit_Power_dBm()   # Calculate the transmit power of the selected BS
                    #cost = self.sce.power_cost * Tx_power_dBm + self.sce.action_cost * nstep  # Calculate the total cost
                    reward = profit #- cost

        # reward = torch.tensor([reward])
        # Rate = torch.tensor([Rate])
        return QoS, reward, Rate
    
    def Save_Transition(self, state, action, next_state, reward, scenario):  # Store a transition
        L = scenario.BS_Number()     # The total number of BSs
        K = self.sce.nChannel        # The total number of channels
        action = torch.tensor([[action]])
        reward = torch.tensor([reward])
        state = state.unsqueeze(0)
        next_state = next_state.unsqueeze(0)
        self.memory.Push(state, action, next_state, reward)
    
    def Target_Update(self):  # Update the parameters of the target network
        self.model_target.load_state_dict(self.model_policy.state_dict())
            
    def Optimize_Model(self):
        if len(self.memory) < self.opt.batch_size:
            return
        transitions = self.memory.Sample(self.opt.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.model_policy(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.opt.batch_size)
        
        """
        next_action_batch = torch.unsqueeze(self.model_policy(non_final_next_states).max(1)[1], 1)
        next_state_values = self.model_target(non_final_next_states).gather(1, next_action_batch) 
        expected_state_action_values = (next_state_values * self.opt.gamma) + reward_batch.unsqueeze(1) 
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)  # Double DQN
        """

        next_state_values[non_final_mask] = self.model_target(non_final_next_states).max(1)[0].detach()  # DQN
        expected_state_action_values = (next_state_values * self.opt.gamma) + reward_batch
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model_policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss






















