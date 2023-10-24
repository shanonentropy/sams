# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:51:10 2023

@author: zahmed

this is simple script to log temperature readings from a 4-wire thermistor 
using pyvisa

this differs from the file in experiment control folder in that it doesn't ask 
for time_duration everytime it is called.

Once started it just logs data 



"""

import pyvisa 
import time
import numpy as np
from datetime import datetime, date
from pathlib import Path

import time
import csv
import sys



####### declare the instrument 

## Connect to the multimeter
rm = pyvisa.ResourceManager()
dmm = rm.open_resource("GPIB0::21::INSTR") # check to see if this is true

## Set the measurement

#rest the device
dmm.write("*RST")
#configure to measure resistnace/voltage
dmm.write("*RST")  #dmm.write(":CONF:VOLT:DC")
# set range to auto
dmm.write(":RES:RANG:AUTO:ON")
#set the integration time to 1 sec for resistance/ for voltage 10 cycles
dmm.write(":RES:APER 1") #dmm.write(":CONF:VOLT:DC")
# set source trig to immediate
dmm.write(":TRIG:SOUR IMM")
#set num of readings to 5
dmm.write(":SAMP:COUN 5")
# take the readings 
dmm.write(":SAMP:COUNT:AUTO ONCE")
dmm.write(":FORM:ELEM READ")
# put readings into a container
a = np.fromstring((dmm.query(":READ?")).replace('\n',','), sep=',').mean()
print('the mean resistance value is {}'.format(a))
              

# path to where file with given name is stored
folder = Path("c:/sams/saved_data")
date =  datetime.now().strftime("%Y_%m_%d_%H_%M_%S") #str(date.today()).replace('-','')
fn = 'datalog_'+date+'_check_thermistor.txt'
file_open = folder / fn

# function for recording temperature to file_open

# num of hours
hr = 2
time_duration = hr*3600 # update this number accordingly

# write a loop to measure over time frame covering the temperature experiment
# log machine time, monotnic time and temperatuer and/or resistance 


# open file in write module

def writer(time_duration, time_spacing=60):
    ''' this code'''
    with open(file_open, mode ='w', newline='') as data:
        fieldnames = ['index', 'elapsed_time', 'time', 'temp']
        data_writer = csv.DictWriter(data, fieldnames=fieldnames)
        data_writer.writeheader()
        print(" please note that the program as configured samples at 0.5 Hz")
        to = time.monotonic() #time at the start of the measurement
        #temp_recorder(time_duration)
        for x in range(time_duration//2):
            t_e = np.round((time.monotonic()-to), 2)
            time_stp = time.monotonic()
            a = np.fromstring((dmm.query(":READ?")).replace('\n',','), sep=',').mean()
            data_writer.writerow({'index':x,'elapsed_time':t_e,'time':time_stp, 'resistance':a})
            data.flush()# forces python to write to disk rather than writing to file in memory
            time.sleep(time_spacing) # wait 10 secon

writer(time_duration)










'''
# Set the measurement mode to resistance
dmm.mode = Agilent34401A.Mode.resistance

# Set the number of readings to take
dmm.trigger_count = 10

# Take the readings and print them
readings = dmm.take_reading()
print(readings)
'''

