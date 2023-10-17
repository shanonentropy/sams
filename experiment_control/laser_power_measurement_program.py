# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 12:43:10 2023

@author: zahmed
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 09:39:08 2023

@author: zahmed

this code will allow me to use the thorlabs PM100D to measure the power profile
of the LABS and newport laser the idea is to see how the laser(s) behave(s) over some time period,
5  hours.

the motivation is to use this data to understand the short (1 min) and long term
(1-2 days) drift that may be occuring in the background of bench top NV
measurements.

validation of this code will also validate the csv.DictWriter and time.monotonic
approach

pow replaced with laser.get_power() call

"""

# import modules

import numpy as np
from pathlib import Path
import time
import csv
import sys
from datetime import date

# add path to sys for custom modules
sys.path.append('c:\\sams\\instrument_control')
# import insturment control modules
from thorlabs_powermeter_API import *
from dlnsec import DLnsec #, *  ### test to see if * is necessary

'''declare instruments'''
# create an instance of arroyo
power_meter=ThorlabsPM100D()
# create an instance of the laser'''
laser= DLnsec('COM7')

'''set a min laser power'''
laser.set_power(5)  # run 5, 10, 15, 20, 25, and 30
'''set laser to lasing mode'''
laser.set_mode('LAS')
laser.on()

# set the number of time samples to be taken
time_duration = 3600 # seconds

folder = Path("c:\\sams\saved_data")

def power_record(time_duration):
    for x in range(time_duration):
        t_e = np.round((time.monotonic()-to), 2)
        data_writer.writerow({'index':x,'elapsed_time':t_e,'time':time.monotonic(),'set_power':laser.get_power() ,'power':power_meter.measure_power()})   # record index, elapsed time, machine time and power meter output
        data.flush()# forces python to write to disk rather than writing to file in memory
        time.sleep(1) # wait 1 seconds



# loop over laser power and record data
def laser_power_loop(time_duration):
    for pow in [1,10,30,50,80,90]:
        print(pow) #delete after debugging
        laser.set_power(pow)
        print('laser power set to {} percent'.format(pow))
        power_record(time_duration)

# set file name
folder = Path("c:/sams/saved_data")
date = str(date.today()).replace('-','')
fn = 'datalog_'+date+'_laser_power_mw.txt'
file_open = folder / fn

with open(file_open, mode ='w', newline='') as data:
    fieldnames = ['index', 'elapsed_time', 'time', 'set_power','power']
    data_writer = csv.DictWriter(data, fieldnames=fieldnames)
    data_writer.writeheader()
    to = time.monotonic()
    laser_power_loop(time_duration)

laser.shutdown()
laser.close()

print('''when done for the day''')
