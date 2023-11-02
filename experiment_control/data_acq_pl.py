# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:26:21 2023

@author: zahmed

Data acqustion class for PL

This is a collection of functions that call upon the laser, drywell and camera
to enable automated collection of PL data under temperature cycling and 
temperature ramp testing conditions

At the present this class will be duplicated to enable ESR measurements. In the 
future the class could be expanded to give a keyword-based ability to choose
between PL and ESR.


note: I am using the class as container to make it easier to call these functions
I could make import the spectroscopy class into this and pass all of its 
variables into it but that seems a bit unnecssary since the way I use it, 
the spectroscopy class is just a set of function and most of the instance vairables
are stored in the *experiment*. 

"""
import sys # Import python sys module
sys.path.append('c:\\sams\instrument_control')
from drywell_interface import Dry_well # drywell control module
from dlnsec import DLnsec#, * # laser control modue
from wait_for import wait_for_drywell, wait_for_x
from time import sleep, date, datetime, time
from pathlib import Path
import pandas as pd


class Data_Acq_PL:
    ''' data acqusition class for acquring temperature dependent
    PL data using the drywell'''
    
    ############################ acqusition routines ###########
    ### looping over laser power at set temp


    def loop_laser_power(drywell_cls, wait_for_x, laser_cls,AcquireAndLock,set_point = 25, power_level= [30, 50, 90, 30]):
        '''
        drywell_cls = an instance of the Dry_well class 
        wait_for_x = either wait_for_x or wait_for_drywell funcs are passed 
        laser_cls = an instance of the laser class 
        set_point default is 25C, it is the temp over which power dependence is measured
        power_level is list containing the percent power level of the laser used in the measurement '''
        drywell_cls.set_temp(set_point); #drywell.set_output(1);
        #wait(drywell.read_stability_status, sleep_seconds =20, timeout_seconds=3600)
        wait_for_x(drywell_cls, sleep_seconds=30, timeout_seconds=3000)
        drywell_cls.beep()
        for p in power_level:
            laser_cls.set_power(p); print('now in laser loop with power at {} percent'.format(p))
            #set filename
            fn = 'laser_power_'+str(p)+'_temp_'+str(str(drywell_cls.read_temp()).replace('.',','))+'_'
            #call camera fuction, pass fn a base filename
            AcquireAndLock(fn)
            
    ###################### temperature scanning loop

    ''' from cycling recall temperature generator and create a temp profile'''


    def temperature_cycling( temp_index_list,wait_for_x, AcquireAndLock, get_status, laser_cls, drywell_cls, settling_time=900,meta_data=[]):
        for i, temp in enumerate(temp_index_list):
            print('index', i ); #sleep(1)
            current_temp = drywell_cls.read_temp()
            drywell_cls.set_temp(temp)
            print('set temp is:',drywell_cls.read_set_temp()); print('current temp is:',drywell_cls.read_temp());
            wait_for_x(drywell_cls, sleep_seconds =20, timeout_seconds=2000)
            drywell_cls.beep()
            print(drywell_cls.read_stability_status()); sleep(settling_time)
            print('now stable at ', drywell_cls.read_temp()); print(drywell_cls.read_stability_status());
            ### turn on laser
            while True:
                if get_status()== 'note: enter appropriate return for locked':
                    laser_cls.on(); sleep(5)
                    laser_cls.set_power(90); sleep(60)
                    p = laser_cls.get_power()
                    #### call camera func, use fn to set filename base
                    fn = 'laser_power_'+str(p)+'_temp_'+str(str(drywell_cls.read_temp()).replace('.',','))+'_'
                    ''' this line exists so I will have a history of drywell's behaviour over the experiment
                    need to add keysight thermometer readout (temp and resistance) to this file and replace drywell temp with
                    check thermometer temp in the file'''
                    meta_data.append([time.time(), i, drywell_cls.read_temp(), drywell_cls.read_stability_status()])
                    #### call camera func, use fn to set filename base
                    fn = 'laser_power_'+str(p)+'_drywelltemp_'+str(str(drywell_cls.read_temp()).replace('.',','))+'_'
                    AcquireAndLock(fn)
                    laser_cls.set_power(10); sleep(30)
                    laser_cls.off()
                else:
                    print('temperature lock has been lost, terminating experiment')
                    break
                
        folder = Path("c:/sams/saved_data")
        dates = str(date.today()).replace('-','')
        fnm = 'meta_data_'+dates+'_nv_exp_ESR_temp.txt'
        file_open = folder / fnm
        df = pd.DataFrame(meta_data)
        df.columns=['time', 'index', 'temp', 'stability_index']
        df.to_csv(path_or_buf=file_open, sep=',')
            
    
    ##### Ramp testing captures heating profile as drywell ramps from e.g.-30 C t0 25 C
    

    def ramp_test(wait_for_x, drywell_cls, get_status,laser_cls,AcquireAndLock, low_temp = -30,high_temp = 25, sleep_time = 900, acqs=10000 ):
        ''' low_temp sets the lower temp where the ramp starts with, default -30C
            high_temp set the upper bond on temp where the ramp ends, default 25 C
            sleep_time is the equilibrition time before the data acqsition starts, defualt is 900s
            acqs is the number of acqustion to acquired during the ramp
            note: pre-ramp is fixed at 100
        '''
        set_point = low_temp; #print(drywell.read_output());
        #wait(drywell.read_stability_status, sleep_seconds =20, timeout_seconds=6000)
        wait_for_x(drywell_cls.read_stability_status(), sleep_seconds=30, timeout_seconds=3000)
        drywell_cls.beep()
        sleep(sleep_time)
        #loop_laser_power()
        ####### call camera to record 15 min worth of data at set temp
        ''' put in call to load a different camera setting'''
        #exp = 'automated_pl_exp_mod' # dummy camera 'xxxx'
        #experiment.ExperimentCompleted += experiment_completed
        AcquireAndLock('test_loading')
        while True:
            if get_status()== 1:#'note: enter appropriate return for locked':
                p = laser_cls.get_power()
                for x in range(100):
                    fn = 'heat_ramp_'+drywell_cls.read_rate()+'deg_per_min_'+'laser_power_'+str(p)+'_temp_'+str(str(drywell_cls.read_temp()).replace('.',','))+'_'
                    AcquireAndLock(fn)
            else:
                print('camera temperature lock is lost')
                break
                #sleep(1)
        # set new temp targe; note that default is 15 frames each of 1 sec
        while True:
            if get_status()== 'note: enter appropriate return for locked':
                drywell_cls.set_temp(25);
                p = laser_cls.get_power()
                for x in range(acqs):
                    '''check what the output of the drywell.read_rate looks like and if it needs to be reformatted.'''
                    fn = 'heat_ramp_'+drywell_cls.read_rate()+'_per_min_laser_power_'+str(p)+'_temp_'+str(str(drywell_cls.read_temp()).replace('.',','))+'_'
                    AcquireAndLock(fn)
            else:
                print('lock has been lost, terminating experiment')
                break
            #sleep(1)
            
            
    def stability_analysis(wait_for_x, drywell_cls, laser_cls,AcquireAndLock, n=100, t=25, delta_time=1, settling_time=900):
        '''this function acquires N number of spectra that will be used to anaylze
        ADEV profile over long time scales
    
        n= number of spectra acquired
        t = temperature defalt 25 C
        delta_time =  time in between spectra
        '''
        #exp = 'automated_pl_exp_mod' # dummy camera 'xxxx'
        drywell_cls.set_temp(t)
        wait_for_x(drywell_cls.read_stability_status(), laser_cls, sleep_seconds=30, timeout_seconds=3000)
        #wait(drywell.read_stability_status, sleep_seconds =20, timeout_seconds=2000)
        print(drywell_cls.read_stability_status()); sleep(settling_time)
        print('now stable at ', drywell_cls.read_temp()); print(drywell_cls.read_stability_status());
        for i in range(n):
            print('at {} C stability run'.format(t), i)
            fn = 'laser_power_'+str(laser_cls.get_power())+'_temp_'+str(str(drywell_cls.read_temp()).replace('.',','))+'_'
            ''' put in call to load a different camera setting'''
            AcquireAndLock(fn)
            sleep(delta_time)


