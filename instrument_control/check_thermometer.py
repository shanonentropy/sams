# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:51:10 2023

@author: zahmed

this code defines a class that is used to read out the resistance
from thermistor using Agilent 34410A DMM

this differs from the file in experiment control folder in that it is class
used to access a thermometer's state in the course of a program

the program in experimental fodler is used to log temperature response of the 
thermistor independent of the NV experiment underway
"""

import pyvisa 
import numpy as np




class Thermometer:
    def __init__(self, rm='', dmm=''):
        ###### declare the instrument 
        ## Connect to the multimeter
        self.rm = pyvisa.ResourceManager()
        self.dmm = self.rm.open_resource("GPIB0::21::INSTR") # check to see if this is true
    def device_configure(self):
        
        ## Set the measurement      
        #rest the device
        #self.dmm.write("*RST")
        #configure to measure resistnace/voltage
        self.dmm.write(":CONF:RES")  #dmm.write(":CONF:VOLT:DC")
        # set range to auto
        self.dmm.write(":RES:RANG:AUTO:ON")
        #set the integration time to 1 sec for resistance/ for voltage 10 cycles
        #self.dmm.write(":RES:APER 1") #dmm.write(":CONF:VOLT:DC")
        # set source trig to immediate
        self.dmm.write(":TRIG:SOUR IMM")
        #set num of readings to 5
        self.dmm.write(":SAMP:COUN 5")
        # take the readings 
        self.dmm.write(":SAMP:COUNT:AUTO ONCE")
        self.dmm.write(":FORM:ELEM READ")
    def read_resistance(self):
        # put readings into a container
        self.a = np.fromstring((self.dmm.query(":READ?")).replace('\n',','), sep=',').mean()
        return self.a
    def print_resistance(self):
        print('the mean resistance value is {}'.format(self.a))
              

# =============================================================================
# 
# therm = Thermometer()
# 
# therm.device_configure()
# therm.print_resistance()
# 
# 
# =============================================================================
