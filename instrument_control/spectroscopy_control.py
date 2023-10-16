# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 16:05:33 2023

@author: zahmed
set of functions to control spectrometer and camera functions
"""

####### spectrometer specific functions
class Spectroscopy:
    '''set of functions to control spectrometer (set and inquire center wavelength
    and camera functions (temperatue: set_temperature/get_current_temp/
                          get_current_set_point
                          
                          get_status i.e if camera is locked or not
                          
                          other functions also available to control device settings,
check device status and thread-lock-data-acquisition )'''
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
        
    ####### camera specific functions    
    
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
         
        # If connected device is not a camera inform the user
        print("Camera not found. Please add a camera and try again.")
        return False  
    
    def experiment_completed(sender, evernt_args):
        print('..acq completed')
        acquireCompleted.Set()
        
    
    def AcquireAndLock(name, self):
        print('acq...', end='')
        name += '{0:06.2f}ms.CWL{1:07.2f}nm'.format(\
                                                    experiment.GetValue(CameraSettings.ShutterTimingExposureTime)\
                                                 ,   700.0)
        experiment.SetValue(ExperimentSettings.FileNameGenerationBaseFileName, name)
        experiment.Acquire()
        acquireCompleted.WaitOne(self)  
    
    
    #### 
