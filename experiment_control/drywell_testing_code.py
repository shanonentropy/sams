# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:13:53 2023

@author: zahmed

drywell testing under SAMS logic: better code is in the 
instrumental control folder

Delete this file from the repo 11/7/2023


rewrite this code to include logging capability. I need to do a 
long-term valuation of the wait_for module to make sure it is working 
as intended and catching the non-sensical out of 


Te



"""

#import modules
import time
from time import sleep
from datetime import datetime, date
#from waiting import wait
import numpy as np
from pathlib import Path
import os # Import os module
import pandas as pd
import csv
import pyvisa
import sys # Import python sys module
sys.path.append('c:\\sams\instrument_control')
from drywell_interface import Dry_well # drywell control module
from temperature_generator import Cycling
from wait_for import wait_for_x, wait_for_drywell


########## wait module -  commented out
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
# =============================================================================

# =============================================================================
# ####### instantiate the drywell 
# =============================================================================
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
wait_for_drywell(drywell,sleep_seconds =30, timeout_seconds=2000)
drywell.beep()
print(drywell.read_temp())
print(drywell.read_rate())



# =============================================================================
# set up a temp index
# =============================================================================

print(Cycling(start=25.0, stop=35.0, step=5, cycles=1).params())
temp_index = Cycling(start=25.0, stop=35.0, step=5, cycles=1).temperatures()


# =============================================================================
# enumerate over temperature profile
# note: data is not recorded here, this is final tune-up before data recording 
# =============================================================================

for _, temperature in enumerate(temp_index):
    print(temperature)
    drywell.set_temp(temperature)
    wait_for_drywell(sleep_seconds =30, timeout_seconds=2000)
    drywell.beep()
    print(drywell.read_stability_status())
    sleep(60)
    print(drywell.read_stability_status())
    print('temperature is now stable at {}, moving to next temp'.format(drywell.read_temp()))
    
    
print('testing_imported_module, setting temp to 27 C')
drywell.set_temp(27)
wait_for_x(drywell, sleep_seconds=30, timeout_seconds=3000)
print('testing_imported_module, setting temp to 25 C')
drywell.set_temp(25)
wait_for_x(drywell, sleep_seconds=30, timeout_seconds=3000)



# =============================================================================
# =============================================================================
# # Thermal cycling
# =============================================================================
# =============================================================================
''' here we will cycle the drywell 3 times over the temp range of interest
while recording the temperature in the well using a check thermometer

We will measure the cycle over the same range and conditions that we intend to 
do most of our measurements on: -30 C to 70 C

'''


# instatiate the grid
print(Cycling(start=-30.0, stop=70.0, step=5, cycles=3).params())
temp_index = Cycling(start=-30.0, stop=70.0, step=5, cycles=3).temperatures()

#instantiate the check thermometer


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



#### let's get cycling


''' note: there is no extra settling time here; we are going to log
thermometer readings immediately and keep logging them for 60 mins.
the idea is to see how long it takes to get to equilibrium at each temperature
over the range of interest and once at equilibrium does it stay stable?'''

# lists to log data in
time_of_measurement,elapsed_time, set_temp, resistance = [],[],[] 

for _, temperature in enumerate(temp_index):
    print(temperature)
    drywell.set_temp(temperature)
    wait_for_x(drywell, sleep_seconds =30, timeout_seconds=2000)
    drywell.beep()
    t_start = time.monotonic()
    while time.monotonic()-t_start<3600:
        a = np.fromstring((dmm.query(":READ?")).replace('\n',','), sep=',').mean()
        time_of_measurement.append(time.monotonic(), (time.monotonic()-t_start),set_temp.append(temperature),resistance.append(a) )
        sleep(10)
        print(temperature, a)
        
        
### save file

data =  zip(time_of_measurement, elapsed_time, set_temp, resistance)

# path to where file with given name is stored
folder = Path("c:/sams/saved_data")
date_today =  datetime.now().strftime("%Y_%m_%d_%H_%M_%S") #str(date.today()).replace('-','')
fn = 'datalog_'+date_today+'_check_thermistor.txt'
file_open = folder / fn

# write to file
with open(file_open, 'w', newline='') as f:
    fieldnames = ['time', 'elapsed_time', 'set temp', 'resistance']
    data_writer = csv.DictWriter(f, fieldnames=fieldnames)
    data_writer.writeheader()
    for row in data:
        data_writer.writerow({'time_of_measurement': data[0], 'elapsed_time':data[1], 'set_temp':data[2], 'resistance':data[3] })

   
drywell.close()









