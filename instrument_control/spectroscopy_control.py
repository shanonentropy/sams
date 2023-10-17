# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 16:05:33 2023

@author: zahmed
set of functions to control spectrometer and camera functions
"""
from time import sleep
import time
import numpy as np
from pathlib import Path
import clr # Import the .NET class library
import os # Import os module
import pandas as pd

import sys
import clr # Import the .NET class library
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
####### spectrometer specific functions
class Spectroscopy:
    '''set of functions to control spectrometer (set and inquire center wavelength
    and camera functions (temperatue: set_temperature/get_current_temp/
                          get_current_set_point
                          
                          get_status i.e if camera is locked or not
                          
                          other functions also available to control device settings,
check device status and thread-lock-data-acquisition )'''
    
    # Create the LightField Application (true for visible)
    # The 2nd parameter forces LF to load with no experiment
    def __init__(self):
        self.auto = Automation(True, List[String]())

        # Get experiment object
        self.experiment = self.auto.LightFieldApplication.Experiment

        self.acquireCompleted = AutoResetEvent(False)

        # Load experiment i.e. pre-configured settings
        self.exp = 'demo_experiment' # dummy camera 'xxxx'
        self.experiment.ExperimentCompleted += self.experiment_completed
    
    
    
    def set_center_wavelength(self, center_wave_length): 
        # Set the spectrometer center wavelength   
        experiment.SetValue(
            SpectrometerSettings.GratingCenterWavelength,
            center_wave_length)    
    
    
    def get_spectrometer_info(self,):   
        print(String.Format("{0} {1}","Center Wave Length:" ,
                      str(experiment.GetValue(
                          SpectrometerSettings.GratingCenterWavelength))))     
           
        print(String.Format("{0} {1}","Grating:" ,
                      str(experiment.GetValue(
                          SpectrometerSettings.Grating))))
        
    ####### camera specific functions    
    
    def set_temperature(self,temperature):
        # Set temperature when LightField is ready to run and
        # when not acquiring data.     
        if (experiment.IsReadyToRun & experiment.IsRunning==False):
            experiment.SetValue(
                CameraSettings.SensorTemperatureSetPoint,
                temperature)        
    
    def get_current_temperature(self):
        print(String.Format(
            "{0} {1}", "Current Temperature:",
            experiment.GetValue(CameraSettings.SensorTemperatureReading)))
    
    def get_current_setpoint(self):
        print(String.Format(
            "{0} {1}", "Current Temperature Set Point:",
            experiment.GetValue(CameraSettings.SensorTemperatureSetPoint)))        
    
    def get_status(self):    
        current = experiment.GetValue(CameraSettings.SensorTemperatureStatus)
        
        print(String.Format(
            "{0} {1}", "Current Status:",
            "UnLocked" if current == SensorTemperatureStatus.Unlocked 
            else "Locked"))
        
        return current
    
    def set_value(self,setting, value):    
        # Check for existence before setting
        # gain, adc rate, or adc quality
        if experiment.Exists(setting):
            experiment.SetValue(setting, value )
    
    def device_found(self):
        # Find connected device
        for device in experiment.ExperimentDevices:
            if (device.Type == DeviceType.Camera):
                return True
         
        # If connected device is not a camera inform the user
        print("Camera not found. Please add a camera and try again.")
        return False  
    
    def experiment_completed(self,sender, evernt_args):
        print('..acq completed')
        acquireCompleted.Set()
        
    
    def AcquireAndLock(self,name):
        print('acq...', end='')
        name += '{0:06.2f}ms.CWL{1:07.2f}nm'.format(\
                                                    experiment.GetValue(CameraSettings.ShutterTimingExposureTime)\
                                                 ,   700.0)
        experiment.SetValue(ExperimentSettings.FileNameGenerationBaseFileName, name)
        experiment.Acquire()
        acquireCompleted.WaitOne()  
    
    
    #### 
