# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:13:53 2023

@author: zahmed

drywell testing under SAMS logic
"""

#import modules
from time import sleep
import time
from waiting import wait
import numpy as np
from pathlib import Path
import os # Import os module
import pandas as pd
import datetime, csv

import sys # Import python sys module
sys.path.append('c:\\sams\instrument_control')
from drywell_interface import Dry_well # drywell control module
from wait_for import wait_for_x, wait_for_drywell
from temperature_generator import Cycling
#from check_thermometer import temp_recorder, writer
####### set drywell #######

drywell = Dry_well()
print(drywell.read_stability_status())
#initial conditions
current_temp = drywell.read_temp()
current_ramp_rate = drywell.read_rate()
current_units = drywell.read_unit()
ramp_rate = drywell.read_rate()
print(current_temp, current_ramp_rate, current_units)
set_point = 25
drywell.set_output(1)
drywell.set_temp(set_point)
sleep(60)
### testing wait_for_x, a generic wait module
#wait_for_x(drywell.read_stability_status(), drywell.read_temp(),sleep_seconds =30, timeout_seconds=2000)
wait_for_x(drywell)
drywell.beep()
### testing wait_for_drywell: explicitly calls drywell.read_stability_status() in the code 
print(drywell.read_temp())

########## drywell loop 
''' loop the drywell back and forth and see if the wait modules
are working properly

to do list:
    *  connect check thermometer to the computer
    * test the code for check thermometer
    * if nec., modify the code for check thermometer
    * proceed with the loop below
        it should yield a bunch of files 
'''






temps =  Cycling(start= -30, stop=70, step = 5, cycles= 1).temperatures()

set_temp, tiempo = [],[]

for setpoint in temps:
    drywell.set_temp(setpoint); to = time.monotonic()
    sleep(60)
    wait_for_x(drywell,timeout_seconds=4000)
    #wait_for_drywell(drywell)
    drywell.beep()
    #waiting only 10 sec so as to catch as much of the 
    print(drywell.read_temp())
    set_temp.append(setpoint)
    tiempo.append(time.monotonic()-to)
    sleep(60)
    
    
drywell.set_temp(25)    
wait_for_x(drywell, timeout_seconds=4500)  
data =  zip( set_temp, tiempo)

# path to where file with given name is stored
folder = Path("c:/sams/saved_data")
date_today =  datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") #str(date.today()).replace('-','')
fn = 'drywell_validation_test'+date_today+'_time_to_settle_as_rep_by_drywell.txt'
file_open = folder / fn

# write to file
with open(file_open, 'w', newline='') as f:
    fieldnames = ['set_temp', 'time'] #['index', 'elapsed_time', 'time', 'temp']
    data_writer = csv.DictWriter(f, fieldnames)
    data_writer.writeheader()
    
    for row in data:
        data_writer.writerow({'set_temp':row[0], 'time': row[1] })

 
drywell.close()
    

  
    
    
    
#############

# =============================================================================
# def wait_for_drywell(sleep_seconds = 30, timeout_seconds= 3000):
#     count = 1; 
#     print('starting counter:', count); 
#     to = time.monotonic() 
#     while count < timeout_seconds//sleep_seconds:
#         if drywell.read_stability_status()== 0:
#             sleep(sleep_seconds)
#             count+= 1
#             print(count, drywell.read_temp(), drywell.read_stability_status())
#         elif drywell.read_stability_status() ==1:
#                 print('stable'); print(time.monotonic()-to)
#                 break
#         elif drywell.read_stability_status() !=1:
#             sleep(sleep_seconds)
#             print('went to bad place')
#             count+=1
#         else:
#             print('timed out')
#             break
# 
# 
# def wait_for_x(funk, funk2,sleep_seconds = 30, timeout_seconds= 3000, ):
#     ''' funk is the function whose binary output you wait on,
#     sleep_seconds = refresh period between queries
#     timeout_seconds= total time to wait before the function breaks out of the loop'''
#     count = 0;
#     print('setting the start counter at:{}'.format(count))
#     #to= time.time()
#     while count < timeout_seconds//sleep_seconds:
#         if funk== 0:
#             sleep(sleep_seconds)
#             count +=1 
#             print(count,funk,  funk2)
#         elif funk ==1:
#                 print('stable', funk); #print(time.monotonic()-to)
#                 break
#         else:
#             sleep(sleep_seconds)
#             count = count+1
#             print('unstable output', count)
#     else:
#         print('timed out')
# =============================================================================




