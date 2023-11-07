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

note: def AcquireandLock has a new variable added: cw=700.0; this is so
that I can change the filename to reflect a change in CWL for NV_zero measurements


"""
import clr # Import the .NET class library
import sys # Import python sys module
import os # Import os module

sys.path.appen('c:/sams/instrument_control')
from drywell_interface import Dry_well
from dlnsec import DLnsec
from check_thermometer import Thermometer
from time import time, sleep
from temperature_generator import Cycling
from pathlib import Path
from datetime import date,datetime
import pandas as pd
import numpy as np


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

def get_status_temp():
    current = experiment.GetValue(CameraSettings.SensorTemperatureStatus)

    t_status  = (String.Format(
        "{0}",
        "UnLocked" if current == SensorTemperatureStatus.Unlocked
        else "Locked"))

    return t_status


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


def AcquireAndLock(name, cw=700.0):
    print('acq...', end='')
    name += '{0:06.2f}ms.CWL{1:07.2f}nm'.format(\
                                                experiment.GetValue(CameraSettings.ShutterTimingExposureTime)\
                                             ,   cw)
    experiment.SetValue(ExperimentSettings.FileNameGenerationBaseFileName, name)
    experiment.Acquire()
    acquireCompleted.WaitOne()


################ activate dummy camera activation

# Create the LightField Application (true for visible)
# The 2nd parameter forces LF to load with no experiment
auto = Automation(True, List[String]())

# Get experiment object
experiment = auto.LightFieldApplication.Experiment

acquireCompleted = AutoResetEvent(False)

# Load experiment i.e. pre-configured settings
exp = 'automated_pl_exp_mod' # 'demo_experiment'
experiment.Load(exp)
experiment.ExperimentCompleted += experiment_completed


AcquireAndLock('test_2')





class Data_Acq_PL(Cycling, Dry_well, DLnsec, Thermometer):
    ''' data acqusition class for acquring temperature dependent
    PL data using the drywell'''

    def __init__(self, start, end, step, cycles, port):
        super().__init__(self, start, end, step, cycles)
        Dry_well.__init__(self)
        DLnsec.__init__(self,port='COM7')
        Thermometer.__init(self)
        self.temp_index = self.temp_generator()


    def wait_for_x(self, sleep_seconds = 30, timeout_seconds= 3000):
        ''' funk is the instance of a clas whose component func's binary output you wait on,
        sleep_seconds = refresh period between queries
        timeout_seconds= total time to wait before the function breaks out of the loop'''
        count = 0;
        print('setting the start counter at:{}'.format(count))
        #to= time.time()
        while count < timeout_seconds//sleep_seconds:
            if self.read_stability_status()== 0:
                sleep(sleep_seconds)
                count +=1
                print(count)
            elif self.read_stability_status() ==1:
                    print('stable'); #print(time.monotonic()-to)
                    break
            else:
                sleep(sleep_seconds)
                count = count+1
                print('unstable output', count)
        else:
            print('timed out')




    ############################ acqusition routines ###########
    ### looping over laser power at set temp


    def loop_laser_power(self, set_point = 25, power_level= [10, 30, 50, 90, 30]):
        '''set_point default is 25C, it is the temp over which power dependence is measured
        power_level is list containing the percent power level of the laser used in the measurement '''
        self.set_temp(self.set_point); #drywell.set_output(1);
        self.wait_for_x(sleep_seconds =20, timeout_seconds=3600)
        self.beep()
        if get_status_temp()=='Locked':
            for p in self.power_level:
                self.set_power(p); print('now in laser loop with power at {} percent'.format(p))
                #set filename
                fn = 'laser_power_'+str(p)+'_temp_'+str(str(self.read_temp()).replace('.',','))+'_'
                #call camera fuction, pass fn a base filename
                AcquireAndLock(fn)

    ###################### temperature scanning loop

    ''' from cycling recall temperature generator and create a temp profile'''


    def temperature_cycling(self,temp_index, meta_data=[],settling_time=900, laser_pow = 90):
        for i, t in enumerate(self.temp_index):
            print('index {}, temp {}'.format( i,t )); #sleep(1)
            #current_temp = self.read_temp()
            self.set_temp(t)
            print('set temp is:',self.set_temp()); print('current temp is:', self.read_temp());
            self.wait_for_x(self, sleep_seconds =20, timeout_seconds=2000)
            self.beep()
            print(self.read_stability_status()); sleep(settling_time)
            print('now stable at ', self.read_temp()); print(self.read_stability_status());
            ### turn on laser
            if get_status_temp()== 'Locked':
                self.on(); sleep(5)
                self.set_power(laser_pow); sleep(60)
                p = self.get_power()
                #### call camera func, use fn to set filename base
                fn = 'laser_power_'+str(p)+'_temp_'+str(str(self.read_temp()).replace('.',','))+'_'
                ''' this line exists so I will have a history of drywell's behaviour over the experiment
                need to add keysight thermometer readout (temp and resistance) to this file and replace drywell temp with
                check thermometer temp in the file'''
                meta_data.append([time.time(), time.monotonic(), i, t, self.read_temp(), self.read_stability_status()])
                #### call camera func, use fn to set filename base
                fn = 'laser_power_'+str(p)+'_drywelltemp_'+str(str(self.read_temp()).replace('.',','))+'_'
                AcquireAndLock(fn)
                self.set_power(10); sleep(30)
                self.off()
            else:
                print('temperature lock has been lost, terminating experiment')


        folder = Path("c:/nv_ensemble/")
        dates = str(date.today()).replace('-','')
        fnm = 'meta_data_'+dates+'_nv_exp_ESR_temp.txt'
        file_open = folder / fnm
        df = pd.DataFrame(meta_data)
        df.columns=['time', 'monotonic_time','index', 'set_temp','temp', 'stability_index']
        df.to_csv(path_or_buf=file_open, sep=',')


    ##### Ramp testing captures heating profile as drywell ramps from e.g.-30 C t0 25 C
    def ramp_test(self,low_temp = -30,high_temp = 25, sleep_time = 900, acqs=10000 ):
        ''' low_temp sets the lower temp where the ramp starts with, default -30C
            high_temp set the upper bond on temp where the ramp ends, default 25 C
            sleep_time is the equilibrition time before the data acqsition starts, defualt is 900s
            acqs is the number of acqustion to acquired during the ramp
            note: pre-ramp is fixed at 100
        '''
        self.set_temp(low_temp)
        self.wait_for_x(self, sleep_seconds =20, timeout_seconds=6000)
        self.beep()
        sleep(sleep_time)
        #loop_laser_power()
        ####### call camera to record 100 scans at the low temp for baseline
        ''' put in call to load a different camera setting'''
        #exp = 'automated_pl_exp_mod' # dummy camera 'xxxx'
        #experiment.ExperimentCompleted += experiment_completed
        #AcquireAndLock('test_loading')
        if get_status_temp()== 'Locked': #'note: enter appropriate return for locked':
            p = self.get_power()
            for x in range(100):
                fn = 'heat_ramp_'+self.read_rate()+'deg_per_min_'+'laser_power_'+str(p)+'_temp_'+str(str(self.read_temp()).replace('.',','))+'_'
                AcquireAndLock(fn)
        else:
            print('camera temperature lock is lost')

        # set new temp target; 
        if get_status()== 1:#'note: enter appropriate return for locked':
            self.set_temp(25);
            p = self.get_power()
            for x in range(acqs):
                '''check what the output of the drywell.read_rate looks like and if it needs to be reformatted.'''
                fn = 'heat_ramp_'+self.read_rate()+'_per_min_laser_power_'+str(p)+'_temp_'+str(str(self.read_temp()).replace('.',','))+'_'
                AcquireAndLock(fn)
        else:
            print('lock has been lost, terminating experiment')

            #sleep(1)


    def stability_analysis(self,n=100, t=25, delta_time=1, settling_time=900):
        '''this function acquires N number of spectra that will be used to anaylze
        ADEV profile over long time scales

        n= number of spectra acquired
        t = temperature defalt 25 C
        delta_time =  time in between spectra
        '''
        #exp = 'automated_pl_exp_mod' # dummy camera 'xxxx'
        self.set_temp(t)
        self.wait_for_x(self, sleep_seconds =20, timeout_seconds=3000)
        print(self.read_stability_status()); sleep(settling_time)
        print('now stable at ', self.read_temp()); print(self.read_stability_status());
        if get_status_temp()=='Locked':
            for i in range(n):
                print('at {} C stability run'.format(t), i)
                fn = 'laser_power_'+str(self.get_power())+'_temp_'+str(str(self.read_temp()).replace('.',','))+'_'
                ''' put in call to load a different camera setting'''
                AcquireAndLock(fn)
                sleep(delta_time)
        else:
            print('ccd out of temp lock')

# =============================================================================
#
#
# class Data_Acq_PL:
#     ''' data acqusition class for acquring temperature dependent
#     PL data using the drywell'''
#
#     ############################ acqusition routines ###########
#     ### looping over laser power at set temp
#
#
#     def loop_laser_power(set_point = 25, power_level= [10, 30, 50, 90, 30]):
#         '''set_point default is 25C, it is the temp over which power dependence is measured
#         power_level is list containing the percent power level of the laser used in the measurement '''
#         drywell.set_temp(set_point); #drywell.set_output(1);
#         wait(drywell.read_stability_status, sleep_seconds =20, timeout_seconds=3600)
#         drywell.beep()
#         for p in power_level:
#             laser.set_power(p); print('now in laser loop with power at {} percent'.format(p))
#             #set filename
#             fn = 'laser_power_'+str(p)+'_temp_'+str(str(drywell.read_temp()).replace('.',','))+'_'
#             #call camera fuction, pass fn a base filename
#             spectroscopy.AcquireAndLock(fn)
#
#     ###################### temperature scanning loop
#
#     ''' from cycling recall temperature generator and create a temp profile'''
#
#
#     def temperature_cycling(temp_index, meta_data=[],settling_time=900, laser_pow = 90):
#         for i, t in enumerate(temp_index):
#             print('index', i ); #sleep(1)
#             current_temp = drywell.read_temp()
#             drywell.set_temp(t)
#             print('set temp is:',drywell.read_set_temp()); print('current temp is:',drywell.read_temp());
#             wait(drywell.read_stability_status, sleep_seconds =20, timeout_seconds=2000)
#             drywell.beep()
#             print(drywell.read_stability_status()); sleep(settling_time)
#             print('now stable at ', drywell.read_temp()); print(drywell.read_stability_status());
#             ### turn on laser
#             while True:
#                 if spectroscopy.get_status()== 1:
#                     laser.on(); sleep(5)
#                     laser.set_power(laser_pow); sleep(60)
#                     p = laser.get_power()
#                     #### call camera func, use fn to set filename base
#                     fn = 'laser_power_'+str(p)+'_temp_'+str(str(drywell.read_temp()).replace('.',','))+'_'
#                     ''' this line exists so I will have a history of drywell's behaviour over the experiment
#                     need to add keysight thermometer readout (temp and resistance) to this file and replace drywell temp with
#                     check thermometer temp in the file'''
#                     meta_data.append([time.time(), time.monotonic(), i, t, drywell.read_temp(), drywell.read_stability_status()])
#                     #### call camera func, use fn to set filename base
#                     fn = 'laser_power_'+str(p)+'_drywelltemp_'+str(str(drywell.read_temp()).replace('.',','))+'_'
#                     spectroscopy.AcquireAndLock(fn)
#                     laser.set_power(10); sleep(30)
#                     laser.off()
#                 else:
#                     print('temperature lock has been lost, terminating experiment')
#                     break
#
#         folder = Path("c:/nv_ensemble/")
#         date = str(date.today()).replace('-','')
#         fnm = 'meta_data_'+date+'_nv_exp_ESR_temp.txt'
#         file_open = folder / fnm
#         df = pd.DataFrame(meta_data)
#         df.columns=['time', 'monotonic_time','index', 'set_temp','temp', 'stability_index']
#         df.to_csv(path_or_buf=file_open, sep=',')
#
#
#     ##### Ramp testing captures heating profile as drywell ramps from e.g.-30 C t0 25 C
#     def ramp_test(low_temp = -30,high_temp = 25, sleep_time = 900, acqs=10000 ):
#         ''' low_temp sets the lower temp where the ramp starts with, default -30C
#             high_temp set the upper bond on temp where the ramp ends, default 25 C
#             sleep_time is the equilibrition time before the data acqsition starts, defualt is 900s
#             acqs is the number of acqustion to acquired during the ramp
#             note: pre-ramp is fixed at 100
#         '''
#         set_point = low_temp; #print(drywell.read_output());
#         wait(drywell.read_stability_status, sleep_seconds =20, timeout_seconds=6000)
#         drywell.beep()
#         sleep(sleep_time)
#         #loop_laser_power()
#         ####### call camera to record 15 min worth of data at set temp
#         ''' put in call to load a different camera setting'''
#         #exp = 'automated_pl_exp_mod' # dummy camera 'xxxx'
#         #experiment.ExperimentCompleted += experiment_completed
#         spectroscopy.AcquireAndLock('test_loading')
#         while True:
#             if spectroscopy.get_status()== 1: #'note: enter appropriate return for locked':
#                 p = laser.get_power()
#                 for x in range(100):
#                     fn = 'heat_ramp_'+drywell.read_rate()+'deg_per_min_'+'laser_power_'+str(p)+'_temp_'+str(str(drywell.read_temp()).replace('.',','))+'_'
#                     spectroscopy.AcquireAndLock(fn)
#             else:
#                 print('camera temperature lock is lost')
#                 break
#                 #sleep(1)
#         # set new temp targe; note that default is 15 frames each of 1 sec
#         while True:
#             if spectroscopy.get_status()== 1:#'note: enter appropriate return for locked':
#                 drywell.set_temp(25);
#                 p = laser.get_power()
#                 for x in range(acqs):
#                     '''check what the output of the drywell.read_rate looks like and if it needs to be reformatted.'''
#                     fn = 'heat_ramp_'+drywell.read_rate()+'_per_min_laser_power_'+str(p)+'_temp_'+str(str(drywell.read_temp()).replace('.',','))+'_'
#                     spectroscopy.AcquireAndLock(fn)
#             else:
#                 print('lock has been lost, terminating experiment')
#                 break
#             #sleep(1)
#
#
#     def stability_analysis(n=100, t=25, delta_time=1):
#         '''this function acquires N number of spectra that will be used to anaylze
#         ADEV profile over long time scales
#
#         n= number of spectra acquired
#         t = temperature defalt 25 C
#         delta_time =  time in between spectra
#         '''
#         exp = 'automated_pl_exp_mod' # dummy camera 'xxxx'
#         drywell.set_temp(t)
#         wait(drywell.read_stability_status, sleep_seconds =20, timeout_seconds=2000)
#         print(drywell.read_stability_status()); sleep(settling_time)
#         print('now stable at ', drywell.read_temp()); print(drywell.read_stability_status());
#         for i in range(n):
#             print('at {} C stability run'.format(t), i)
#             fn = 'laser_power_'+str(p)+'_temp_'+str(str(drywell.read_temp()).replace('.',','))+'_'
#             ''' put in call to load a different camera setting'''
#             spectroscopy.AcquireAndLock(fn)
#             sleep(delta_time)
#
#
#
# =============================================================================
