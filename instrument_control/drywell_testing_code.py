# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:13:53 2023

@author: zahmed

drywell testing under SAMS logic
"""

#import modules
from time import sleep
from waiting import wait
import numpy as np
from pathlib import Path
import os # Import os module
import pandas as pd


import sys # Import python sys module
sys.path.append('c:\\sams\instrument_control')
from drywell_interface import dry_well # drywell control module
from wait_for import wait_for_x, wait_for_drywell
from temperature_generator import Cycling
#from check_thermometer import temp_recorder, writer
####### set drywell #######

drywell = dry_well()
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
### testing wait_for_x, a generic wait module
wait_for_x(drywell.read_stability_status(), sleep_seconds =30, timeout_seconds=2000)
drywell.beep()
### testing wait_for_drywell: explicitly calls drywell.read_stability_status() in the code 
wait_for_drywell(drywell.read_stability_status(), sleep_seconds =30, timeout_seconds=2000)
drywell.beep()
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



temps =  Cycling(start= 25, stop=45, step = 5, cycles= 3).temperatures()

for setpoint in temps:
    drywell.set_temp(setpoint)
    wait_for_x(drywell.read_stability_status(), sleep_seconds =30, timeout_seconds=2000)
    #waiting only 10 sec so as to catch as much of the 
    sleep(1800)
    
    

    
    
    
    


