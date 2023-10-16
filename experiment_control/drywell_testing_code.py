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
from temperature_generator import cycling
####### set drywell 

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
wait(drywell.read_stability_status, sleep_seconds =30, timeout_seconds=2000)
drywell.beep()
print(drywell.read_temp())
print(drywell.read_rate())

''' note to self: does the drywell.read_rate() output require further processing
before being inserted into a file name, if so, address it in ramp_test function 
of the data_acq_pl file'''

# set up a temp index
temp_index = cycling(start=25, stop=30, step=5, cycles=1).temperatures()

for _, temperature in enumerate(temp_index):
    print(temperature)
    drywell.set_temp(temperature)
    wait(drywell.read_stability_status, sleep_seconds =30, timeout_seconds=2000)
    drywell.beep()
    print('temperature is now stable at {}, moving to next temp'.format(drywell.read_temp()))
    
    
    
drywell.close()