# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 13:43:30 2023

@author: zahmed
 this is a test code for validating zpl data acquisiton routine with out turning
 the laser on
 
 
 
import of spectroscopy class did not work  
 

import Data_Acq_Pl fails because it needs to inherent drywell which it doesn't recognize'

"""

#import modules
from time import sleep
import time
import numpy as np
from pathlib import Path
import clr # Import the .NET class library
import os # Import os module
import pandas as pd
#import pyvisa


import sys # Import python sys module
sys.path.append('c:\\sams\instrument_control')
from drywell_interface import Dry_well # drywell control module
from dlnsec import DLnsec#, * # laser control modue
from spectroscopy_control import experiment_completed,AcquireAndLock,get_status,set_value
from temperature_generator import Cycling
from data_acq_pl import Data_Acq_PL 
from wait_for import wait_for_drywell, wait_for_x
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


# =============================================================================
# top code will ensure that laser, heat bath and camera are in communication
# and properly configured
# =============================================================================


# =============================================================================
# set laser but don't turn it on
# =============================================================================

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

# =============================================================================
# instantiate the drywell
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
#wait(drywell.read_stability_status, sleep_seconds =30, timeout_seconds=2000)
wait_for_drywell(drywell)
drywell.beep()
print(drywell.read_temp())

####### setup camera and spectrometer

####### spectrometer specific functions

def set_center_wavelength(center_wave_length): 
    # Set the spectrometer center wavelength   
    experiment.SetValue(
        SpectrometerSettings.GratingCenterWavelength,
        center_wave_length)    


def get_spectrometer_info():   
    print(String.Format("{0} {1}","Center Wave Length:" ,
                  str(experiment.GetValue(
                      SpectrometerSettings.GratingCenterWavelength))))     
       
    print(String.Format("{0} {1}","Grating:" ,
                  str(experiment.GetValue(
                      SpectrometerSettings.Grating))))
    
def set_temperature(temperature):
    # Set temperature when LightField is ready to run and
    # when not acquiring data.     
    if (experiment.IsReadyToRun & experiment.IsRunning==False):
        experiment.SetValue(
            CameraSettings.SensorTemperatureSetPoint,
            temperature)        

def get_current_temperature():
    print(String.Format(
        "{0} {1}", "Current Temperature:",
        experiment.GetValue(CameraSettings.SensorTemperatureReading)))

def get_current_setpoint():
    print(String.Format(
        "{0} {1}", "Current Temperature Set Point:",
        experiment.GetValue(CameraSettings.SensorTemperatureSetPoint)))        

def get_status():    
    current = experiment.GetValue(CameraSettings.SensorTemperatureStatus)
    
    print(String.Format(
        "{0} {1}", "Current Status:",
        "UnLocked" if current == SensorTemperatureStatus.Unlocked 
        else "Locked"))
    return current

def set_value(setting, value):    
    # Check for existence before setting
    # gain, adc rate, or adc quality
    if experiment.Exists(setting):
        experiment.SetValue(setting, value )

def device_found():
    # Find connected device
    for device in experiment.ExperimentDevices:
        if (device.Type == DeviceType.Camera):
            return True
            print("Camera not found. Please add a camera and try again.")
    return False  

def experiment_completed(sender, evernt_args):
    print('..acq completed')
    acquireCompleted.Set()
    

def AcquireAndLock(name):
    print('acq...', end='')
    name += '{0:06.2f}ms.CWL{1:07.2f}nm'.format(\
                                                experiment.GetValue(CameraSettings.ShutterTimingExposureTime)\
                                             ,   700.0)
    experiment.SetValue(ExperimentSettings.FileNameGenerationBaseFileName, name)
    experiment.Acquire()
    acquireCompleted.WaitOne()  

    
    
# Create the LightField Application (true for visible)
# The 2nd parameter forces LF to load with no experiment
auto = Automation(True, List[String]())

# Get experiment object
experiment = auto.LightFieldApplication.Experiment

acquireCompleted = AutoResetEvent(False)

# Load experiment i.e. pre-configured settings
exp = 'demo_experiment' # dummy camera 'xxxx'
experiment.ExperimentCompleted += experiment_completed

AcquireAndLock('dummy_test3')
    

# =============================================================================
# instantiate data acqusition class
# =============================================================================


dq = Data_Acq_PL()

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
drywell.set_temp(25); wait_for_x(drywell); drywell.beep()

''' First we want to see if the dummy camera 
will let us test the get_status() loop with with laser power loop '''

# first we want to ensure that temp is locked; #### not working
while True:
    if get_status()== 1:  #'note: locked value in spectroscopy appears to be 1'
        dq.loop_laser_power(drywell, wait_for_x, laser, AcquireAndLock)  #'write out stuff to declared in the func'
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
Cycling(start=25, stop=27, step=1, cycles=1).params()

# setup the profile
temp_index =  Cycling(start=25, stop=26, step=1, cycles=1).temperatures()
# this list will log meta deta for the thermal profile'''
#execute the temperature profile
dq.temperature_cycling(temp_index, wait_for_x, AcquireAndLock, get_status, laser, drywell) #'write out stuff to declared in the func'

###################


''' acquire stability data '''



# first we want to ensure that temp is locked
while True:
    if get_status()== 1: #'note: enter appropriate return for locked':
        dq.stability_analysis(wait_for_x, drywell, laser, AcquireAndLock)  #'wait_for_x, drywell_cls, laser_cls,AcquireAndLock,
    else:
        print('temperature lock lost, terminate experiment')
        break

''' acquire ramp data '''
dq.ramp_test(wait_for_x, drywell, get_status, laser, AcquireAndLock, low_temp = 25, high_temp=30, sleep_time=10, acqs=1000 ) #'write out stuff to declared in the func'

#### sep test the ramp_test code with new exp loading, if that fails, 
#### build in provision in the code to wait for the manual adjustment

''' turn the laser off'''
#laser.set_power(10); sleep(30); laser.off(); laser.set_mode('STOP')
laser.close()

''' turn the drywell off'''
drywell.close()




