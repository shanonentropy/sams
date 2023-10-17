# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:13:53 2023

@author: zahmed

drywell testing under SAMS logic


rewrite this code to include logging capability. I need to do a 
long-term valuation of the wait_for module to make sure it is working 
as intended and catching the non-sensical out of 



"""

#import modules
import time
from time import sleep
#from waiting import wait
import numpy as np
from pathlib import Path
import os # Import os module
import pandas as pd


import sys # Import python sys module
sys.path.append('c:\\sams\instrument_control')
from drywell_interface import Dry_well # drywell control module
from temperature_generator import Cycling
from wait_for import wait_for_x


########## wait module 
def wait_for_drywell(sleep_seconds = 30, timeout_seconds= 3000):
    count = 1; 
    print('starting counter:', count); 
    to = time.monotonic() 
    while count < timeout_seconds//sleep_seconds:
        if drywell.read_stability_status()== 0:
            sleep(sleep_seconds)
            count+= 1
            print(count)
        elif drywell.read_stability_status() ==1:
                print('stable'); print(time.monotonic()-to)
                break
        elif drywell.read_stability_status() !=1:
            sleep(sleep_seconds)
            print('went to bad place')
            count+=1
        else:
            print('timed out')
            break

####### set drywell 

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
wait_for_drywell(sleep_seconds =30, timeout_seconds=2000)
#wait(drywell.read_stability_status, sleep_seconds =30, timeout_seconds=2000)
drywell.beep()
print(drywell.read_temp())
print(drywell.read_rate())




''' note to self: does the drywell.read_rate() output require further processing
before being inserted into a file name, if so, address it in ramp_test function 
of the data_acq_pl file'''

# set up a temp index
temp_index = Cycling(start=25.0, stop=30.0, step=5, cycles=1).temperatures()

for _, temperature in enumerate(temp_index):
    print(temperature)
    drywell.set_temp(temperature)
    wait_for_drywell(sleep_seconds =30, timeout_seconds=2000)
    drywell.beep()
    print(drywell.read_stability_status())
    sleep(60)
    print(drywell.read_stability_status())
    print('temperature is now stable at {}, moving to next temp'.format(drywell.read_temp()))
    
    
print('testing_imported_module')
drywell.set_temp(25)
wait_for_x(drywell.read_stability_status(), sleep_seconds=30, timeout_seconds=3000)




    
drywell.close()









