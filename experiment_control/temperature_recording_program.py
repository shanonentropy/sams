# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 09:39:08 2023

@author: zahmed

this code will allow me to use the Arroyo API to record the lab temperature
over a week long time period.

the idea is to see how the lab temperature behaves over some time period,
7 days, with the lights on and off

the motivation is to use this data to understand the short (1 hr) and long term
(1-2 days) temperature drift that may be occuring in the background of NV strain
measurement.

09112023: currently the code is used to monitor room temperature when bech top
ESR measurements are done

the code has been modified to run from the prompt, taking in user input for
starting and stopping the program. It also

"""

import numpy as np
from datetime import datetime, date
from pathlib import Path

import time
import csv
import sys

# import insturment control module
sys.path.append('c:\\sams\\instrument_control')
from serial_interface import arroyo
'''declare instruments'''
# create an instance of arroyo
tec = arroyo()

# set the number of time samples to be taken
#### for test run, 5 min is 5 steps and for 7 days it is 10,080 with sleep(30)

#time_duration = 21080 # desired_time_length_in_seconds/time_between_measurements

# path to where file with given name is stored
folder = Path("c:/sams/saved_data")
date = str(date.today()).replace('-','')
fn = 'datalog_'+date+'_lab_monitoring.txt'
file_open = folder / fn

# function for recording temperature to file_open

def temp_recorder(time_duration):
    for x in range(time_duration//2):
        t_e = np.round((time.monotonic()-to), 2)
        time_stp = time.monotonic()
        a = tec.read_temp()
        #b = a.split('\r'); c = a#b[0]
        data_writer.writerow({'index':x,'elapsed_time':t_e,'time':time_stp, 'temp':a})
        data.flush()# forces python to write to disk rather than writing to file in memory
        time.sleep(30) # wait 30 secon

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
