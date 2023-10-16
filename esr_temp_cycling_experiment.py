# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 08:13:43 2023

@author: zahmed

this code will be be used to control the ESR spectrocopy experiment.
for now the program will rely on qdspectro library to call upon the esr
equipment while temperature is controlled using the drywell code.


goal in oct-dec timeframe is to rewrite the qdspectro library casting it into
OOP language which will make available the experimental parameters as 
__init__ variables that a future RL alogrithm can access and modify as needed

"""

#import modules
from time import sleep
from waiting import wait
import numpy as np
from pathlib import Path
import clr # Import the .NET class library
import os # Import os module
import pandas as pd
import sys # Import python sys module

sys.path.append('c:\\sams\instrument_control')
from drywell_interface import dry_well # drywell control module
from dlnsec import DLnsec#, * # laser control modue
from temperature_generator import cycling
from mainControl import*
from ESRconfig import*
#from esr_control import cw_esr


####### set laser 

laser = DLnsec('com7')  # check if this is the correct port
print(laser.get_power())  # check initial condition 
#set initial laser power
power = 10
laser.set_power(power)
# set mode 
laser.set_mode('LAS')
# laser action
laser.on(); sleep(60)
#turn laser off
laser.off()

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
drywell.set_output(1); drywell.set_temp(set_point)
wait(drywell.read_stability_status, sleep_seconds =30, timeout_seconds=2000)
sleep(900)
print(drywell.read_temp())

print(drywell.read_stability_status())


############################ acqusition routines ###########
### looping over laser power at set temp  
''' add functionality to change filename'''

def loop_laser_power(set_point = 25, power_level= [30, 50, 90, 30]):
    #drywell = dry_well()
    #set_point = 25.0
    drywell.set_temp(set_point); #drywell.set_output(1); 
    wait(drywell.read_stability_status, sleep_seconds =20, timeout_seconds=3600)  
    drywell.beep()
    for p in power_level:
        laser.set_power(p); print('now in laser loop with power at {} percent'.format(p))
        #set filename
        #fn = 'laser_power_'+str(p)+'_temp_'+str(str(drywell.read_temp()).replace('.',','))+'_'  
        '''replace this command'''
        runfile('mainControl.py', args='ESRconfig')
        #sleep(1)#sleep func is there to help regularize the time acqusition
 

###########  loop over mw power

''' write a code that will modify the input to initial 
settings for the SG384'''



###########  loop over acq parameters

''' write code(s) to loop over sample times, number of samples, and averages'''



###################### temperature scanning loop 

''' from cycling recall temperature generator and create a temp profile'''

''' add functionality to change filename'''

def temperature_cycling(temp_index, meta_data,settling_time=900):
    for i in range(len(temp_index)):
        print('index', i ); #sleep(1)
        drywell.set_temp((temp_index[i]))
        print('set temp is:',drywell.read_set_temp()); print('current temp is:',drywell.read_temp());
        wait(drywell.read_stability_status, sleep_seconds =20, timeout_seconds=2000)
        drywell.beep()
        print(drywell.read_stability_status()); sleep(settling_time)
        print('now stable at ', drywell.read_temp()); print(drywell.read_stability_status());
        ### turn on laser
        laser.on(); sleep(5)
        laser.set_power(90); sleep(60)
        p = laser.get_power()
        #### call esr func, use fn to set filename base
        fn = 'laser_power_'+str(p)+'_temp_'+str(str(drywell.read_temp()).replace('.',','))+'_' 
        #### to be deleted
        ''' this line exists so I will have a history of drywell's behaviour over the experiment
        need to add keysight thermometer readout (temp and resistance) to this file and replace drywell temp with
        check thermometer temp in the file'''
        meta_data.append([time.time(), i, drywell.read_temp(), drywell.read_stability_status()])
        '''replace this func'''
        runfile('mainControl.py', args='ESRconfig')
        laser.set_power(10); sleep(30)
        laser.off()

    '''this won't be needed in the future, could be utilized to log thermometer'''
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
    ''' specify input parameters'''
   # write code here
    for x in range(100):
        fn = 'heat_ramp_'+drywell.read_rate()+'deg_per_min_'+'laser_power_'+str(p)+'_temp_'+str(str(drywell.read_temp()).replace('.',','))+'_' 
        '''replace this func'''
        runfile('mainControl.py', args='ESRconfig')
    
    # set new temp targe; note that default is 15 frames each of 1 sec
    drywell.set_temp(25); 
    p = laser.get_power()
    for x in range(acqs):
        '''check what the output of the drywell.read_rate looks like and if it needs to be reformatted.'''
        fn = 'heat_ramp_'+drywell.read_rate()+'_per_min_laser_power_'+str(p)+'_temp_'+str(str(drywell.read_temp()).replace('.',','))+'_' 
        '''replace this func'''
        runfile('mainControl.py', args='ESRconfig')  

        
        
def stability_analysis(n=100, t=25, delta_time=1, settling_time = 900):
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
        '''replace this func'''
        runfile('mainControl.py', args='ESRconfig')
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
loop_laser_power() 
   
laser.set_power(10); sleep(60); laser.off()

################
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


################
''' acquire stability data '''
stability_analysis()



################

''' acquire ramp data '''

# ramp_test()    

################
################

''' turn the laser off'''
laser.set_power(10); sleep(30); laser.off(); laser.set_mode('STOP')
laser.close()

''' turn the drywell off'''
drywell.close()