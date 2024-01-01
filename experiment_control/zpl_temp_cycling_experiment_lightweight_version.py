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

Note: 11/08 Therm class is deactivated as the workstation does not recognize the device
will try the pass-through config with SRS cable to see if that works

"""

##import modules
from time import sleep
import sys # Import python sys module
sys.path.append('c:\\sams\instrument_control')
from data_acq_pl import Data_Acq_PL ,  experiment, acquireCompleted#, auto 
from data_acq_pl import get_status_temp, experiment_completed, AcquireAndLock #,set_center_wavelength, 


# =============================================================================
# instantiate the PL data acquisition class
#   this will declare the instruments and load the acqusition functions
#   note: be mindful of default settings for the functions
#   note: currently check_thermometer cls Therm() is deactivated. 
# =============================================================================
dq =  Data_Acq_PL()


# =============================================================================
# turn the laser on //default is off for safety reasons
# =============================================================================

''' we turn the laser on and set power to an initial value'''
dq.set_power(10)

''' set drywell to room temp or some chosen value'''
dq.set_temp(25); dq.beep()


''' We start with power loop, using default settings [10,30,50, 90, 30]'''
dq.set_mode('LAS')
dq.on()  #turn laser on
dq.loop_laser_power()


''' set laser to a power appropriate for the experiment based on results above'''    
dq.set_power(50)

''' acquire stability data '''
# the default settling_time is 900 sec, n = 100, t is 25 C
dq.stability_analysis()

''' turn laser off, while you get the temp loop ready'''
dq.set_power(10); sleep(60); dq.off()
print('starting temp cycling')

''' loop over a defined temp//print the array'''
# print the proposed profile
dq.params()

''' set the drywell to the lowest temperature to get it ready for the 
temperature cycling experiment'''
dq.set_temp(-30) 
dq.wait_for_x(timeout_seconds=4000)

# =============================================================================
# #execute the temperature cycling profile
# =============================================================================
dq.temperature_cycling()

''' turn the laser back on 'cause temp cycling turns it off'''
dq.on()

# =============================================================================
# power dependence and stability analysis at low temp, followed by ramp testing
# =============================================================================
''' acquire stability data at the lowest temp of the cycle'''
dq.stability_analysis()

''' acquire power dependence at the lowest temperature'''
dq.loop_laser_power()

''' acquire ramp data '''
# default settings low_temp=-30, high_temp=25, sleep_time=900, acqs=10000, baseline_acqs=10
dq.ramp_test()

# =============================================================================
# shutdown 
# =============================================================================
''' turn the laser off'''
dq.shutdown()
dq.close()

''' turn the drywell off'''
dq.close_drywell()
