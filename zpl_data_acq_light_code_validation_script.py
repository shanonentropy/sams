# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 13:43:30 2023

@author: zahmed
 this is a test code for validating zpl data acquisiton routine with out turning
 the laser on

"""

#import modules
from time import sleep
from waiting import wait
import numpy as np
from pathlib import Path
import clr # Import the .NET class library
import os # Import os module
import pandas as pd
#import pyvisa


import sys # Import python sys module
sys.path.append('c:\\sams\instrument_control')
from drywell_interface import dry_well # drywell control module
from dlnsec import DLnsec#, * # laser control modue
from spectrocop_control import Spectroscopy
from temperature_generator import Cycling
from data_acq_pl import Data_Acq_PL 
# Import System.IO for saving and opening files
from System.IO import *
from System.Threading import AutoResetEvent   # this is for thread mangement
# Import c compatible List and String
from System.Collections.Generic import List
from System import String, IntPtr, Int64, Double # this is because python reqs explicit call from .NET
from System.Runtime.InteropServices import Marshal
from System.IO import FileAccess
clr.AddReference('System.Windows.Forms')

# Add needed dll references
sys.path.append(os.environ['LIGHTFIELD_ROOT'])
sys.path.append(os.environ['LIGHTFIELD_ROOT']+"\\AddInViews")
clr.AddReference('PrincetonInstruments.LightFieldViewV5')
clr.AddReference('PrincetonInstruments.LightField.AutomationV5')
clr.AddReference('PrincetonInstruments.LightFieldAddInSupportServices')

# PI imports
from PrincetonInstruments.LightField.Automation import Automation
from PrincetonInstruments.LightField.AddIns import CameraSettings
from PrincetonInstruments.LightField.AddIns import SensorTemperatureStatus
from PrincetonInstruments.LightField.AddIns import DeviceType
from PrincetonInstruments.LightField.AddIns import ExperimentSettings
from PrincetonInstruments.LightField.AddIns import SpectrometerSettings



### top code will ensure that laser, heat bath and camera are in communication
### and properly set
####### set laser but don't turn it on

laser = DLnsec('com7')  # check if this is the correct port
print(laser.get_power())  # check initial condition
#set initial laser power
power = 10
laser.set_power(power)
# set mode
laser.set_mode('LAS')
# laser action
#laser.on()
#turn laser off
#laser.off()

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
wait(drywell.read_stability_status, sleep_seconds =30, timeout_seconds=2000)
drywell.beep()
print(drywell.read_temp())

####### setup camera and spectrometer

spectroscopy = Spectroscopy()
data_acq = Data_Acq_PL()


################ activate  camera activation

# Create the LightField Application (true for visible)
# The 2nd parameter forces LF to load with no experiment
auto = Automation(True, List[String]())

# Get experiment object
experiment = auto.LightFieldApplication.Experiment

acquireCompleted = AutoResetEvent(False)

# Load experiment i.e. pre-configured settings
exp = 'dummy_code' # dummy camera 'xxxx'
experiment.ExperimentCompleted += experiment_completed


spectrocopy.AcquireAndLock('dummy_test')

##################################################################
##################################################################
##################################################################
###################### Experiment ################################
##################################################################
##################################################################
##################################################################


#''' we turn the laser on and set power to an initial value'''

#laser.set_power(90)

''' set drywell to room temp or some chosen value'''
drywell.set_temp(25); drywell.beep()


''' First we want to see if the dummy camera 
will let us test the get_status() loop with with laser power loop '''

# first we want to ensure that temp is locked
while True:
    if Spectroscopy().get_status()== 1:  #'note: locked value in spectroscopy appears to be 1'
        Data_Acq_PL().loop_laser_power()
    else:
        print('temperature lock lost, terminate experiment')
        break
''' if not, just test the line below, this should set the laser power to desired value
- with laser off, and then prompt it to acquire a spectra with the correct file name
as designated in my naming scheme
 '''

#Data_Acq_PL.loop_laser_power()



#laser.set_power(10); sleep(60); laser.off()
print('starting temp cycling')


''' single cycle temp'''
# print the proposed profile
Cycling(start=25, stop=26, step=1, cycles=1).params()

# setup the profile
temp_index =  Cycling(start=25, stop=26, step=1, cycles=1).temperatures()
# this list will log meta deta for the thermal profile'''
#meta_data =[]

#execute the temperature profile
Data_Acq_PL().temperature_cycling(temp_index, meta_data=[])

###################



''' acquire stability data '''


# first we want to ensure that temp is locked
while True:
    if Spectroscopy().get_status()== 1: #'note: enter appropriate return for locked':
        Data_Acq_PL().stability_analysis()
    else:
        print('temperature lock lost, terminate experiment')
        break
#

''' acquire ramp data '''

data_acq.ramp_test(low_temp = 25, high_temp=30, sleep_time=10, acqs=1000 )

#### sep test the ramp_test code with new exp loading, if that fails, 
#### build in provision in the code to wait for the manual adjustment

''' turn the laser off'''
#laser.set_power(10); sleep(30); laser.off(); laser.set_mode('STOP')
laser.close()

''' turn the drywell off'''
drywell.close()
