# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:51:10 2023

@author: zahmed

this is simple script to log temperature readings from a 2-wire thermistor 
using pyvisa. this is stand-alone version of the code that predates the 
Thermometer class (c"/sams/instrument_control/check_thermometer.py")

to do: rename the file 11/7


"""


import pyvisa 
import time, datetime
import numpy as np
from datetime import datetime, date
from pathlib import Path

import time
import csv
import sys



####### declare the instrument 
# Connect to the multimeter
rm = pyvisa.ResourceManager()
dmm = rm.open_resource("GPIB0::21::INSTR") # check to see if this is true
# Set the measurement

#rest the device
dmm.write("*RST")
#configure to measure resistnace/voltage
dmm.write('CONF:RES')  #dmm.write(":CONF:VOLT:DC")
# set range to auto
dmm.write(":RES:RANG:AUTO:ON")
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
dates =  str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) 
fnt = 'datalog_'+dates+'_check_thermistor.txt'
file_open_temp = folder / fnt



# open file in write module
# write a loop to measure over time frame covering the temperature experiment
# log machine time, monotnic time and temperatuer and/or resistance 

elapsed_time_temp, time_step_temp, unix_time, resistance = [],[],[], []
time_spacing = 10

print(" please note that the program as configured samples at 0.5 Hz")
# main code: ask for user input on when to start data collection and for how long
while True:
    recorder =  input("to start recording temperature data press S, to quit the loop press Q ")
    if recorder.lower() == 's':
        time_duration =  input("how many hours of data do you want? enter a number ")
        try:
            time_duration = int(time_duration)*3600 #convert hours into seconds
            print('lab data collection has began')
            to = time.monotonic()
            while (time.monotonic()-to) < time_duration:
                t_e = np.round((time.monotonic()-to), 2)
                time_stp = time.monotonic()
                a = np.fromstring((dmm.query(":READ?")).replace('\n',','), sep=',').mean()
                elapsed_time_temp.append(t_e),time_step_temp.append(time_stp),unix_time.append(time.time()), resistance.append(a)
                time.sleep(time_spacing) # wait 10 secon
                print(time.monotonic()-to)
            break
        except ValueError:
            print('not an integer!')
            break
    elif recorder.lower() == 'q':
        print('exiting program')
        break
    else:
        print("please hit the s key")
        #break
data = list(zip(elapsed_time_temp, time_step_temp, unix_time, resistance))
print('data is being saved')   
with open(file_open_temp, mode ='w', newline='') as f:
    fieldnames = [ 'elapsed_time', 'time_step', 'unix_time', 'resistance']
    data_writer = csv.DictWriter(f, fieldnames=fieldnames)
    data_writer.writeheader()
    for r in data:
        data_writer.writerow({'elapsed_time':r[0], 'time_step':r[1],'unix_time':r[2] ,'resistance':r[3]})
        

