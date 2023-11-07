# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:52:01 2023

@author: zahmed
this code will be be used to control the PL spectrocopy experiment
it controls the camera, spectrometer, laser and drywell.

this hopefully will be an intermediary code with Data_Acq_PL class
eventually incorporating most of the routine functions in the next iteration.

Note: as part of the class inheretance resolution in Data_Acq_PL class 
the Dry_well class functions including close(), write_command(),   

"""

##import modules
from time import sleep
#import numpy as np
#from pathlib import Path
#import clr # Import the .NET class library
#import os # Import os module
#import pandas as pd
#import pyvisa


import sys # Import python sys module
sys.path.append('c:\\sams\instrument_control')
#from drywell_interface import dry_well # drywell control module
#from dlnsec import DLnsec#, * # laser control modue
#from spectrocop_control import Spectroscopy
from temperature_generator import Cycling
''' are these two import lines necessary'''
from data_acq_pl import Data_Acq_PL,  experiment, acquireCompleted#, auto 
from data_acq_pl import get_status_temp, experiment_completed, AcquireAndLock #,set_center_wavelength, 

## Import System.IO for saving and opening files
#from System.IO import *
#from System.Threading import AutoResetEvent   # this is for thread mangement
## Import c compatible List and String
#from System.Collections.Generic import List
#from System import String, IntPtr, Int64, Double # this is because python reqs explicit call from .NET
#from System.Runtime.InteropServices import Marshal
#from System.IO import FileAccess
#clr.AddReference('System.Windows.Forms')

## Add needed dll references
#sys.path.append(os.environ['LIGHTFIELD_ROOT'])
#sys.path.append(os.environ['LIGHTFIELD_ROOT']+"\\AddInViews")
#clr.AddReference('PrincetonInstruments.LightFieldViewV5')
#clr.AddReference('PrincetonInstruments.LightField.AutomationV5')
#clr.AddReference('PrincetonInstruments.LightFieldAddInSupportServices')

## PI imports
#from PrincetonInstruments.LightField.Automation import Automation
#from PrincetonInstruments.LightField.AddIns import CameraSettings
#from PrincetonInstruments.LightField.AddIns import SensorTemperatureStatus
#from PrincetonInstruments.LightField.AddIns import DeviceType
#from PrincetonInstruments.LightField.AddIns import ExperimentSettings
#from PrincetonInstruments.LightField.AddIns import SpectrometerSettings




dq =  Data_Acq_PL()


# =============================================================================
# ################ activate  camera activation
''' do I need these calls here and if so do I need to put the import window up top''' 
## Create the LightField Application (true for visible)
## The 2nd parameter forces LF to load with no experiment
#auto = Automation(True, List[String]())
 
## Get experiment object
#experiment = auto.LightFieldApplication.Experiment
# 
#acquireCompleted = AutoResetEvent(False)
# 
# Load experiment i.e. pre-configured settings
exp = 'automated_pl_exp_mod' # dummy camera 'xxxx'
experiment.ExperimentCompleted += experiment_completed
# 
# 
AcquireAndLock('test_2')


# =============================================================================

##################################################################
##################################################################
##################################################################
###################### Experiment ################################
##################################################################
##################################################################
##################################################################


''' we turn the laser on and set power to an initial value'''

dq.set_power(90)

''' set drywell to room temp or some chosen value'''
dq.set_temp(25); dq.beep()


''' We start with power loop, using default settings '''
dq.loop_laser_power()
    

''' acquire stability data '''

# first we want to ensure that temp is locked
dq.stability_analysis()




''' turn laser off, while you get the temp loop ready'''
dq.set_power(10); sleep(60); dq.off()
print('starting temp cycling')


''' loop over a defined temp'''
# print the proposed profile
Cycling(start=-30, stop=70, step=5, cycles=3).params()

# setup the profile
temp_index =  Cycling(start=-30, stop=70, step=5, cycles=3).temperatures()
# this list will log meta deta for the thermal profile'''
#meta_data =[]

''' set the drywell to the lowest temperature to get it ready for the 
temperature cycling experiment'''

dq.set_temp(-30)
dq.wait_for_x(timeout_seconds=4000)

#execute the temperature profile
dq.temperature_cycling(temp_index, meta_data=[])

###################



''' acquire stability data '''

dq.stability_analysis()


''' acquire ramp data '''
# ramp_test()



''' turn the laser off'''
#laser.set_power(10); sleep(30); laser.off(); laser.set_mode('STOP')
dq.shutdown()
dq.close()

''' turn the drywell off'''
dq.close_drywell()
