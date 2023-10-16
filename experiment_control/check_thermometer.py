# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:51:10 2023

@author: zahmed

this is simple script to log temperature readings from a 4-wire thermistor 
using pymeausre 
"""

from pymeasure.instruments.agilent import Agilent34401A
import time
import numpy as np
from datetime import datetime, date
from pathlib import Path

import time
import csv
import sys



####### declare the instrument 
# Connect to the multimeter
dmm = Agilent34401A("GPIB::1") # check to see if this is true
# Set the measurement mode to temperature
dmm.mode = Agilent34401A.Mode.temperature
# Set the number of readings to take
dmm.trigger_count = 10
# Test run of the readout utility; Take readings and print them
readings = dmm.take_reading()
print(readings)


# path to where file with given name is stored
folder = Path("c:/sams/saved_data")
date = str(date.today()).replace('-','')
fn = 'datalog_'+date+'_check_thermistor.txt'
file_open = folder / fn

# function for recording temperature to file_open

def temp_recorder(time_duration, time_spacing = 10):
    ''' time_duration: in secs, the total length of the measurement
    time_spacing  is the sleep time between readings'''
    for x in range(time_duration//2):
        t_e = np.round((time.monotonic()-to), 2)
        time_stp = time.monotonic()
        a = dmm.take_reading()
        #b = a.split('\r'); c = a#b[0]
        data_writer.writerow({'index':x,'elapsed_time':t_e,'time':time_stp, 'temp':a})
        data.flush()# forces python to write to disk rather than writing to file in memory
        time.sleep(time_spacing) # wait 30 secon


# write a loop to measure over time frame covering the temperature experiment
# log machine time, monotnic time and temperatuer and/or resistance 


# open file in write module
with open(file_open, mode ='w', newline='') as data:
    fieldnames = ['index', 'elapsed_time', 'time', 'temp']
    data_writer = csv.DictWriter(data, fieldnames=fieldnames)
    data_writer.writeheader()
    print(" please note that the program as configured samples at 0.5 Hz")
    # main code: ask for user input on when to start data collection and for how long
    while True:
        recorder =  input("to start recording temperature data press S, to quit the loop press Q ")
        if recorder.lower() == 's':
            time_duration =  input("how many hours of data do you want? enter a number ")
            try:
                time_duration = int(time_duration)*3600 #convert hours into seconds
                print('lab data collection has began')
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
    to = time.monotonic() #time at the start of the measurement
    temp_recorder(time_duration)














'''
# Set the measurement mode to resistance
dmm.mode = Agilent34401A.Mode.resistance

# Set the number of readings to take
dmm.trigger_count = 10

# Take the readings and print them
readings = dmm.take_reading()
print(readings)
'''

