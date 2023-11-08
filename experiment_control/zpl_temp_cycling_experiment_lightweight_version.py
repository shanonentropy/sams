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
#import time


import sys # Import python sys module
sys.path.append('c:\\sams\instrument_control')
from data_acq_pl import Data_Acq_PL ,  experiment, acquireCompleted#, auto 
from data_acq_pl import get_status_temp, experiment_completed, AcquireAndLock #,set_center_wavelength, 





# note: set cycling params here when calling the class
### note to self: figure out how to inherit default settings of parent class
dq =  Data_Acq_PL(25,30, 5, 1)



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

dq.set_mode('LAS')
dq.on()  #turn laser on
dq.loop_laser_power()
    

''' acquire stability data '''
# remove the n=3 after validation testing
# the default settling_time is 900 sec
dq.stability_analysis(n=3, settling_time=10)




''' turn laser off, while you get the temp loop ready'''
dq.set_power(10); sleep(60); dq.off()
print('starting temp cycling')


''' loop over a defined temp//print the array'''
# print the proposed profile
dq.params()

''' set the drywell to the lowest temperature to get it ready for the 
temperature cycling experiment'''

#dq.set_temp(-30) 
#dq.wait_for_x(timeout_seconds=4000)

#execute the temperature profile
dq.temperature_cycling(settling_time=10)

###################
''' turn the laser back on 'cause temp cycling turns it off'''

dq.on()

''' acquire stability data at the lowest temp of the cycle'''
dq.stability_analysis(n=3, settling_time=1)

''' acquire power dependence at the lowest temperature'''
dq.loop_laser_power()

''' acquire ramp data '''
# revert to default setting after validation run
dq.ramp_test(low_temp=25, high_temp=30, sleep_time=10, acqs=10, baseline_acqs=3)



''' turn the laser off'''
#laser.set_power(10); sleep(30); laser.off(); laser.set_mode('STOP')
dq.shutdown()
dq.close()

''' turn the drywell off'''
dq.close_drywell()
