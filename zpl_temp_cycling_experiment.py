# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:52:01 2023

@author: zahmed
this code will be be used to control the PL spectrocopy experiment
it control the camera, spectrometer, laser and drywell.
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
from spectrocop_control import spectroscopy
from temperature_generator import cycling
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
####### set laser

laser = DLnsec('com7')  # check if this is the correct port
print(laser.get_power())  # check initial condition
#set initial laser power
power = 10
laser.set_power(power)
# set mode
laser.set_mode('LAS')
# laser action
laser.on()
#turn laser off
#laser.off()

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

####### setup camera and spectrometer

################ activate  camera activation

# Create the LightField Application (true for visible)
# The 2nd parameter forces LF to load with no experiment
auto = Automation(True, List[String]())

# Get experiment object
experiment = auto.LightFieldApplication.Experiment

acquireCompleted = AutoResetEvent(False)

# Load experiment i.e. pre-configured settings
exp = 'automated_pl_exp_mod' # dummy camera 'xxxx'
experiment.ExperimentCompleted += experiment_completed


spectrocopy.AcquireAndLock('test_2')


############################ acqusition routines ###########
### looping over laser power at set temp


def loop_laser_power(set_point = 25, power_level= [30, 50, 90, 30]):
    #drywell = dry_well()
    #set_point = 25.0
    drywell.set_temp(set_point); #drywell.set_output(1);
    wait(drywell.read_stability_status, sleep_seconds =20, timeout_seconds=3600)
    drywell.beep()
    for p in power_level:
        laser.set_power(p); print('now in laser loop with power at {} percent'.format(p))
        #set filename
        fn = 'laser_power_'+str(p)+'_temp_'+str(str(drywell.read_temp()).replace('.',','))+'_'
        #call camera fuction, pass fn a base filename
        spectroscopy.AcquireAndLock(fn)


###################### temperature scanning loop

''' from cycling recall temperature generator and create a temp profile'''


def temperature_cycling(temp_index, meta_data,settling_time=900):
    for i in range(len(temp_index)):
        print('index', i ); #sleep(1)
        current_temp = drywell.read_temp()
        drywell.set_temp((temp_index[i]))
        print('set temp is:',drywell.read_set_temp()); print('current temp is:',drywell.read_temp());
        wait(drywell.read_stability_status, sleep_seconds =20, timeout_seconds=2000)
        drywell.beep()
        print(drywell.read_stability_status()); sleep(settling_time)
        print('now stable at ', drywell.read_temp()); print(drywell.read_stability_status());
        ### turn on laser
        while True:
            if spectroscopy.get_status()== 'note: enter appropriate return for locked':
                laser.on(); sleep(5)
                laser.set_power(90); sleep(60)
                p = laser.get_power()
                #### call camera func, use fn to set filename base
                fn = 'laser_power_'+str(p)+'_temp_'+str(str(drywell.read_temp()).replace('.',','))+'_'
                ''' this line exists so I will have a history of drywell's behaviour over the experiment
                need to add keysight thermometer readout (temp and resistance) to this file and replace drywell temp with
                check thermometer temp in the file'''
                meta_data.append([time.time(), i, drywell.read_temp(), drywell.read_stability_status()])
                #### call camera func, use fn to set filename base
                fn = 'laser_power_'+str(p)+'_drywelltemp_'+str(str(drywell.read_temp()).replace('.',','))+'_'
                spectroscopy.AcquireAndLock(fn)
                laser.set_power(10); sleep(30)
                laser.off()
            else:
                print('temperature lock has been lost, terminating experiment')
                break
    folder = Path("c:/nv_ensemble/")
    date = str(date.today()).replace('-','')
    fnm = 'meta_data_'+date+'_nv_exp_ESR_temp.txt'
    file_open = folder / fnm
    df = pd.DataFrame(meta_data)
    df.columns=['time', 'index', 'temp', 'stability_index']
    df.to_csv(path_or_buf=file_open, sep=',')



##### Ramp testing captures heating profile as drywell ramps from e.g.-30 C t0 25 C
''' note that this function will require the user to switch to single frame
acqusition mode. I need to implement loading of a different experiment at this
stage '''

def ramp_test(low_temp = -30,high_temp = 25, sleep_time = 900, acqs=10000 ):
    ''' low_temp sets the lower temp where the ramp starts with, default -30C
        high_temp set the upper bond on temp where the ramp ends, default 25 C
        sleep_time is the equilibrition time before the data acqsition starts, defualt is 900s
        acqs is the number of acqustion to acquired during the ramp
        note: pre-ramp is fixed at 100
    '''
    set_point = low_temp; #print(drywell.read_output());
    wait(drywell.read_stability_status, sleep_seconds =20, timeout_seconds=6000)
    drywell.beep()
    sleep(sleep_time)
    #loop_laser_power()
    ####### call camera to record 15 min worth of data at set temp
    ''' put in call to load a different camera setting'''
    exp = 'automated_pl_exp_mod' # dummy camera 'xxxx'
    experiment.ExperimentCompleted += experiment_completed
    spectroscopy.AcquireAndLock('test_loading')
    while True:
        if spectroscopy.get_status()== 'note: enter appropriate return for locked':
            p = laser.get_power()
            for x in range(100):
                fn = 'heat_ramp_'+drywell.read_rate()+'deg_per_min_'+'laser_power_'+str(p)+'_temp_'+str(str(drywell.read_temp()).replace('.',','))+'_'
                spectroscopy.AcquireAndLock(fn)
        else:
            print('camera temperature lock is lost')
            break
            #sleep(1)
    # set new temp targe; note that default is 15 frames each of 1 sec
    while True:
        if spectroscopy.get_status()== 'note: enter appropriate return for locked':
            drywell.set_temp(25);
            p = laser.get_power()
            for x in range(acqs):
                '''check what the output of the drywell.read_rate looks like and if it needs to be reformatted.'''
                fn = 'heat_ramp_'+drywell.read_rate()+'_per_min_laser_power_'+str(p)+'_temp_'+str(str(drywell.read_temp()).replace('.',','))+'_'
                spectroscopy.AcquireAndLock(fn)
        else:
            print('lock has been lost, terminating experiment')
            break
        #sleep(1)


def stability_analysis(n=100, t=25, delta_time=1):
    '''this function acquires N number of spectra that will be used to anaylze
    ADEV profile over long time scales

    n= number of spectra acquired
    t = temperature defalt 25 C
    delta_time =  time in between spectra
    '''
    exp = 'automated_pl_exp_mod' # dummy camera 'xxxx'
    drywell.set_temp(t)
    wait(drywell.read_stability_status, sleep_seconds =20, timeout_seconds=2000)
    print(drywell.read_stability_status()); sleep(settling_time)
    print('now stable at ', drywell.read_temp()); print(drywell.read_stability_status());
    for i in range(n):
        print('at {} C stability run'.format(t), i)
        fn = 'laser_power_'+str(p)+'_temp_'+str(str(drywell.read_temp()).replace('.',','))+'_'
        ''' put in call to load a different camera setting'''
        spectroscopy.AcquireAndLock(fn)
        sleep(delta_time)



##################################################################
##################################################################
##################################################################
###################### Experiment ################################
##################################################################
##################################################################
##################################################################


''' we turn the laser on and set power to an initial value'''

laser.set_power(90)

''' set drywell to room temp or some chosen value'''
drywell.set_temp(25); drywell.beep()


''' We start with power loop, using default settings '''

# first we want to ensure that temp is locked
while True:
    if spectroscopy.get_status()== 1:  #'note: locked value in spectroscopy appears to be 1'
        loop_laser_power()
    else:
        print('temperature lock lost, terminate experiment')
        break

laser.set_power(10); sleep(60); laser.off()
print('starting temp cycling')


''' loop over a defined temp'''
# print the proposed profile
cycling(start=-30, stop=70, step=3, cycles=1).params()

# setup the profile
temp_index =  cycling(start=-30, stop=70, step=3, cycles=1).temperatures()
# this list will log meta deta for the thermal profile'''
meta_data =[]

#execute the temperature profile
temperature_cycling(temp_index, meta_data)

###################



''' acquire stability data '''


# first we want to ensure that temp is locked
while True:
    if spectroscopy.get_status()== 'note: enter appropriate return for locked':
        stability_analysis()
    else:
        print('temperature lock lost, terminate experiment')
        break
#

''' acquire ramp data '''

# ramp_test()



''' turn the laser off'''
laser.set_power(10); sleep(30); laser.off(); laser.set_mode('STOP')
laser.close()

''' turn the drywell off'''
drywell.close()
