# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:49:45 2020

@author: liangyu

Running simuluation
"""

import copy, json, argparse
import torch
from copy import deepcopy
from pylab import * #作图用
from matplotlib import rcParams
config = {
            "font.family": 'serif',
            "font.size": 15,
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun'],
         }
rcParams.update(config) #图例显示汉字宋体和新罗马

from scenario import Scenario
from agent import Agent
from dotdic import DotDic
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('D:/Research/UARA-DRL-master/UARA-DRL-master/Log/test3')

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")
#model.to(device)

def create_agents(opt, sce, scenario, device):
	agents = []   # Vector of agents
	for i in range(opt.nagents):
		agents.append(Agent(opt, sce, scenario, index=i, device=device)) # Initialization, create a CNet for each agent
	return agents
    
def run_episodes(opt, sce, agents, scenario): 
    nepisode = 0
    episode_num = torch.zeros(opt.nepisodes)
    totalreward = torch.zeros(opt.nepisodes)
    Throughput = torch.zeros(opt.nepisodes)
    totalLoss = torch.zeros(opt.nepisodes)
    action = torch.zeros(opt.nagents,dtype=int)
    proportion = torch.zeros(opt.nagents)
    g = torch.zeros(opt.nagents)
    reward = torch.zeros(opt.nagents)
    Rate = torch.zeros(opt.nagents)
    QoS = torch.zeros(opt.nagents)
    state_target = torch.ones(opt.nagents)  # The QoS requirement
    f = open("DDQN.csv","w+")
    f.write("This includes the running steps:\n")
    # totalstep = 0
    while nepisode < opt.nepisodes:
        state = torch.zeros([2, opt.nagents])  # Reset the state
        for i in range(opt.user_type1): # Set the User Type in state
            state[0, i] = 0
        for i in range(opt.user_type2):
            state[0, i + opt.user_type1] = 1
        next_state = torch.zeros([2, opt.nagents])  # Reset the next_state
        next_state[0, :] = state[0, :]  # Keep the User Type in next_state
        episode_num[nepisode] = nepisode    # Update episode number
        nstep = 0
        while nstep < opt.nsteps:
            eps_threshold = opt.eps_min + opt.eps_increment * nstep * (nepisode + 1)
            if eps_threshold > opt.eps_max:
                eps_threshold = opt.eps_max  # Linear increasing epsilon
                # eps_threshold = opt.eps_min + (opt.eps_max - opt.eps_min) * np.exp(-1. * nstep * (nepisode + 1)/opt.eps_decay) 
                # Exponential decay epsilon
            for i in range(opt.nagents):
                action[i] = agents[i].Select_Action(state[1, :], scenario, eps_threshold)  # Select action
            for i in range(opt.nagents):
                g[i] = agents[i].Get_Gain(action, action[i], scenario) # Get channel gain
                proportion[i] = agents[i].Get_Proportion(action, action[i], state, scenario, i, g) # Get Transmit Power Proportion
            for i in range(opt.nagents):
                QoS[i], reward[i], Rate[i] = agents[i].Get_Reward(action, action[i], state, scenario, i, g, proportion, nstep)  # Obtain reward and next state
                next_state[1, i] = QoS[i]
            for i in range(opt.nagents):
                agents[i].Save_Transition(state[1, :], action[i], next_state[1, :], reward[i], scenario)  # Save the state transition
                loss = agents[i].Optimize_Model()  # Train the model
                if nstep % opt.nupdate == 0:  # Update the target network for a period
                    agents[i].Target_Update()
            state[1, :] = deepcopy(next_state[1, :])  # State transits
            if torch.all(state.eq(state_target)):  # If QoS is satisified, break
                break
            nstep += 1
        # totalstep += nstep
        totalreward[nepisode] = torch.sum(reward)
        Throughput[nepisode] = torch.sum(Rate)
        totalLoss[nepisode] = loss
        print('Episode Number:', nepisode, 'Reward:', totalreward[nepisode], 'Throughput:', Throughput[nepisode], 'loss', totalLoss[nepisode])
        writer.add_scalar('Reward:', totalreward[nepisode], nepisode)
        f.write("%i \n" % nstep)
        nepisode += 1
    f.close()
    writer.close()

    # 创建数组并保存
    Reward = totalreward.numpy().reshape((opt.nepisodes, 1))
    SysThroughput = Throughput.numpy().reshape((opt.nepisodes, 1))
    Loss = totalLoss.numpy().reshape((opt.nepisodes, 1))
    Data = np.hstack((Reward, SysThroughput, Loss))
    # 存储
    np.savetxt(fname="DQN_20UE_1M0.5MQoS_10BS_5RB_720kHz.csv", X=Data, fmt="%d", delimiter=",")

    # # 读取
    # b = np.loadtxt(fname="test.csv", dtype=np.int, delimiter=",")

    figure(figsize=(8, 6), dpi=80)
    subplot(1, 2, 1)
    plot(episode_num, totalreward, color="green", marker='s', linewidth=1.5, linestyle="-", label="用户数：，速率阈值：")
    xlim(0, 500)
    xlabel("episodes")
    ylim(0, 50)
    ylabel("reward")

    subplot(1, 2, 2)
    plot(episode_num, Throughput, color="blue", marker='s', linewidth=1.5, linestyle="-", label="用户数：，速率阈值：")
    xlim(0, 500)
    xlabel("episodes")
    ylabel("Throughput / Mbps")

    savefig("test1.png", dpi=1080)

    show()


def run_trial(opt, sce):
    scenario = Scenario(sce)
    agents = create_agents(opt, sce, scenario, device)  # Initialization
    run_episodes(opt, sce, agents, scenario)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c1', '--config_path1', type=str, help='path to existing scenarios file')
    parser.add_argument('-c2', '--config_path2', type=str, help='path to existing options file')
    parser.add_argument('-n', '--ntrials', type=int, default=1, help='number of trials to run')
    args = parser.parse_args()
    sce = DotDic(json.loads(open('D:/Research/RAN-DRL/RAN-DRL/Config/config_1.json', 'r').read()))
    opt = DotDic(json.loads(open('D:/Research/RAN-DRL/RAN-DRL/Config/config_2.json', 'r').read()))  # Load the configuration file as arguments
    # sce = DotDic(json.loads(open('config_1.json', 'r').read()))
    # opt = DotDic(json.loads(open('config_2.json', 'r').read()))  # Load the configuration file as arguments
    for i in range(args.ntrials):
        trial_result_path = None
        trial_opt = deepcopy(opt)
        trial_sce = deepcopy(sce)
        run_trial(trial_opt, trial_sce)
