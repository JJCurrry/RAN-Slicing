# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:51:49 2020

@author: liangyu

Create the network simulation scenario
"""

import numpy as np
from numpy import pi
from random import random, uniform, choice

class BS:  # Define the base station
    
    def __init__(self, sce, BS_index, BS_type, BS_Loc, BS_Radius):
        self.sce = sce
        self.id = BS_index
        self.BStype = BS_type
        self.BS_Loc = BS_Loc
        self.BS_Radius = BS_Radius
        
    def reset(self):  # Reset the channel status
        self.Ch_State = np.zeros(self.sce.nChannel)    
        
    def Get_Location(self):
        return self.BS_Loc
    
    def temp_Transmit_Power_dBm(self):  # Calculate the transmit power of a BS
        if self.BStype == "RRU":
            temp_Tx_Power_dBm = 50
        elif self.BStype == "FAP":
            temp_Tx_Power_dBm = 46
        """
        elif self.BStype == "FBS":
            Tx_Power_dBm = 20
        """
        return temp_Tx_Power_dBm  # Transmit power in dBm, no consideration of power allocation now

    def temp_Receive_Power(self, d):  # Calculate the received power by transmit power and path loss of a certain BS
        temp_Tx_Power_dBm = self.temp_Transmit_Power_dBm() / self.sce.nChannel
        temp_Tx_Power = 10 ** (temp_Tx_Power_dBm / 10)  # Transmit power in mW
        if self.BStype == "RRU" :
            loss = 128.1 + 37.6 * np.log10(d/1000)
            """loss = 34 + 40 * np.log10(d)"""
        elif self.BStype == "FAP":
            loss = 140.7 + 36.7 * np.log10(d/1000)
            """loss = 37 + 30 * np.log10(d)"""
        """
        elif self.BStype == "FBS":
            loss = 37 + 30 * np.log10(d)
        """
        if d <= self.BS_Radius:
            temp_Rx_power_dBm = temp_Tx_Power_dBm - loss  # Received power in dBm
            temp_Rx_power = 10**(temp_Rx_power_dBm/10)  # Received power in mW
            h = temp_Rx_power / temp_Tx_Power
        else:
            temp_Rx_power = 0.0
            h = temp_Rx_power / temp_Tx_Power
        return temp_Rx_power, h

    def Transmit_Power_dBm(self):  # Calculate the transmit power of a BS
        if self.BStype == "RRU":
            Tx_Power_dBm = 50
        elif self.BStype == "FAP":
            Tx_Power_dBm = 46
        """
        elif self.BStype == "FBS":
            Tx_Power_dBm = 20
        """
        return Tx_Power_dBm  # Transmit power in dBm, no consideration of power allocation now

    def Receive_Power(self, d, proportion):  # Calculate the received power by transmit power and path loss of a certain BS
        Tx_Power_dBm = (self.Transmit_Power_dBm() / self.sce.nChannel) * proportion
        Tx_Power = 10 ** (Tx_Power_dBm / 10)  # Transmit power in mW
        if self.BStype == "RRU" :
            loss = 128.1 + 37.6 * np.log10(d/1000)
            """loss = 34 + 40 * np.log10(d)"""
        elif self.BStype == "FAP":
            loss = 140.7 + 36.7 * np.log10(d/1000)
            """loss = 37 + 30 * np.log10(d)"""
        """
        elif self.BStype == "FBS":
            loss = 37 + 30 * np.log10(d)
        """
        if d <= self.BS_Radius:
            Rx_power_dBm = Tx_Power_dBm - loss  # Received power in dBm
            Rx_power = 10**(Rx_power_dBm/10)  # Received power in mW
            h = Rx_power / Tx_Power
        else:
            Rx_power = 0.0
            h = Rx_power / Tx_Power
        return Rx_power, h

class Scenario:  # Define the network scenario

    def __init__(self, sce):  # Initialize the scenario we simulate
        self.sce = sce
        self.BaseStations = self.BS_Init()

    def reset(self):   # Reset the scenario we simulate
        for i in range(len(self.BaseStations)):
            self.BaseStations[i].reset()

    def BS_Number(self):
        nBS = self.sce.nRRU + self.sce.nFAP  # The number of base stations
        """nBS = self.sce.nRRU + self.sce.nFAP + self.sce.nFBS  # The number of base stations"""
        return nBS
    
    def BS_Location(self):
        Loc_RRU = np.zeros((self.sce.nRRU,2))  # Initialize the locations of BSs
        Loc_FAP = np.zeros((self.sce.nFAP,2))
        """Loc_FBS = np.zeros((self.sce.nFBS,2))"""
        
        for i in range(self.sce.nRRU):
            # Loc_RRU[i,0] = 500 + 900*i  # x-coordinate
            # Loc_RRU[i,1] = 500  # y-coordinate
            Loc_RRU[i, 0] = 200 + 350 * i  # x-coordinate
            Loc_RRU[i, 1] = 200  # y-coordinate
        #要把RRU改成随机分布！
        for i in range(self.sce.nFAP):
            """
            Loc_FAP[i,0] = Loc_RRU[int(i/4),0] + 250*np.cos(pi/2*(i%4))
            Loc_FAP[i,1] = Loc_RRU[int(i/4),1] + 250*np.sin(pi/2*(i%4))
            """
            LocM = choice(Loc_RRU)
            r = self.sce.rRRU*random()
            theta = uniform(-pi,pi)
            Loc_FAP[i,0] = LocM[0] + r*np.cos(theta)
            Loc_FAP[i,1] = LocM[1] + r*np.sin(theta)
        # FAP随机分布！

        """
        for i in range(self.sce.nFBS):
            LocM = choice(Loc_RRU)
            r = self.sce.rRRU*random()
            theta = uniform(-pi,pi)
            Loc_FBS[i,0] = LocM[0] + r*np.cos(theta)
            Loc_FBS[i,1] = LocM[1] + r*np.sin(theta)
        """

        return Loc_RRU, Loc_FAP# return Loc_RRU, Loc_FAP, Loc_FBS
    
    def BS_Init(self):   # Initialize all the base stations 
        BaseStations = []  # The vector of base stations
        Loc_RRU, Loc_FAP= self.BS_Location()# Loc_RRU, Loc_FAP, Loc_FBS = self.BS_Location()
        
        for i in range(self.sce.nRRU):  # Initialize the RRUs
            BS_index = i
            BS_type = "RRU"
            BS_Loc = Loc_RRU[i]
            BS_Radius = self.sce.rRRU            
            BaseStations.append(BS(self.sce, BS_index, BS_type, BS_Loc, BS_Radius))
            
        for i in range(self.sce.nFAP):
            BS_index = self.sce.nRRU + i
            BS_type = "FAP"
            BS_Loc = Loc_FAP[i]
            BS_Radius = self.sce.rFAP
            BaseStations.append(BS(self.sce, BS_index, BS_type, BS_Loc, BS_Radius))

        """
        for i in range(self.sce.nFBS):
            BS_index = self.sce.nRRU + self.sce.nFAP + i
            BS_type = "FBS"
            BS_Loc = Loc_FBS[i]
            BS_Radius = self.sce.rFBS
            BaseStations.append(BS(self.sce, BS_index, BS_type, BS_Loc, BS_Radius))
        """
        return BaseStations
            
    def Get_BaseStations(self):
        return self.BaseStations


        
            
    

