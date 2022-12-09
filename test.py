import copy, json, argparse
import torch
import numpy
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
import numpy as np
from scenario import Scenario
from agent import Agent
from dotdic import DotDic
from torch.utils.tensorboard import SummaryWriter
# A=torch.zeros(1,10)
# for i in range(10):  # Set the User Type in state
#     A[0,i] = i
# B=torch.zeros(1,10)
# for i in range(10):  # Set the User Type in state
#     B[0,i] = 10-i
#
# # 创建数组（3维）
# a = A.numpy().reshape((10,1))
# b = B.numpy().reshape((10,1))
# c=np.hstack((a,b))

# # 存储
# np.savetxt(fname="test.csv", X=c, fmt="%d",delimiter=",")
#
# # 读取
# r = np.loadtxt(fname="test.csv", dtype=np.int, delimiter=",")
a=720000*np.log10(1 + (10**4)) / (10**6)


print(a)
