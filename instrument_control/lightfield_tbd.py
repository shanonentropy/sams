# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 19:47:28 2023

@author: zahmed   spectroscopy class attempt
"""

from time import sleep
import clr
import sys
import os
from System.IO import *
from System.Threading import AutoResetEvent
from System.Collections.Generic import List
from System import String, IntPtr, Int64, Double
from System.Runtime.InteropServices import Marshal
from System.IO import FileAccess

class LightField:
    def __init__(self):
        clr.AddReference('System.Windows.Forms')
        sys.path.append(os.environ['LIGHTFIELD_ROOT'])
        sys.path.append(os.environ['LIGHTFIELD_ROOT']+"\\AddInViews")
        clr.AddReference('PrincetonInstruments.LightFieldViewV5')
        clr.AddReference('PrincetonInstruments.LightField.AutomationV5')
        clr.AddReference('PrincetonInstruments.LightFieldAddInSupportServices')
        
        from PrincetonInstruments.LightField.Automation import Automation
        from PrincetonInstruments.LightField.AddIns import CameraSettings
        from PrincetonInstruments.LightField.AddIns import SensorTemperatureStatus
        from PrincetonInstruments.LightField.AddIns import DeviceType
        from PrincetonInstruments.LightField.AddIns import ExperimentSettings
        from PrincetonInstruments.LightField.AddIns import SpectrometerSettings
        
        self.experiment = Automation(True, ListString)
    
    def set_center_wavelength(self, center_wave_length):
        self.experiment.SetValue(
            SpectrometerSettings.GratingCenterWavelength,
            center_wave_length)
        
        print(String.Format("{0} {1}","Grating:" ,
                      str(self.experiment.GetValue(
                          SpectrometerSettings.Grating))))     
    
    def get_current_temperature(self):
        print(String.Format(
            "{0} {1}", "Current Temperature:",
            self.experiment.GetValue(CameraSettings.SensorTemperatureReading)))
    
    def get_status(self):
        current = self.experiment.GetValue(CameraSettings.SensorTemperatureStatus)
        
        print(String.Format(
            "{0} {1}", "Current Status:",
            "UnLocked" if current == SensorTemperatureStatus.Unlocked 
            else "Locked"))
        
        return current
    
    def set_value(self, setting, value):
        if self.experiment.Exists(setting):
            self.experiment.SetValue(setting, value)
    
    def device_found(self):
        for device in self.experiment.ExperimentDevices:
            if (device.Type == DeviceType.Camera):
                return True
                
                print("Camera not found. Please add a camera and try again.")
                
        return False
    
    def experiment_completed(self, sender, event_args):
        print('..acq completed')
        
        acquireCompleted.Set()
    
    def AcquireAndLock(self, name):
        print('acq...', end='')
        
        name += '{0:06.2f}ms.CWL{1:07.2f}nm'.format(
            self.experiment.GetValue(CameraSettings.ShutterTimingExposureTime),
            700.0)
        
        self.experiment.SetValue(ExperimentSettings.FileNameGenerationBaseFileName, name)
        
        self.experiment.Acquire()
        
        acquireCompleted.WaitOne()
        
        
        
        
lf = LightField()
#lf.set_center_wavelength(500)
lf.get_current_temperature()
lf.get_status()
#lf.set_value("gain", 10)
#lf.device_found()
lf.AcquireAndLock("test")        