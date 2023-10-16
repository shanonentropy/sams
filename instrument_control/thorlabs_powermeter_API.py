# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 14:39:57 2022

@author: zahmed

the main code was from github (not Claude's fork').
It was then modified to work with the latest version of 
pyvisa using usb as opposed to GPIP connection

In addition, documentation has been added to explain each of the functions and 
in some cases changes made to the code to fit my needs

note that this code obiviates the need for ThorlabsPM100D module from pip, which 
didn't work well on atleast one of the computers'

"""

from __future__ import division, print_function
import pyvisa#, usb.core
import time
import logging

TRIES_BEFORE_FAILURE = 10
RETRY_SLEEP_TIME = 0.010  # in seconds

logger = logging.getLogger(__name__)


class ThorlabsPM100D(object):
    """
    Define and communicate with Thorlabs PM100D power meter    
    using the PyVISA 1.12 library over USB. 
    """

    
    def __init__(self, port="USB0::0x1313::0x8078::P0031649::INSTR", debug=False):
    
        self.port = port #define communication port
        self.debug = debug #debug set to false

        rm = pyvisa.ResourceManager() # instatiate a resource manager
    
        if debug: rm.list_resources() #kept from the old code will test later
    
        self.pm = rm.open_resource(port) #open communication with the device 
    
        self.idn = self.pm.query("*IDN?") # establish identity of device/basic test
        
        self.sensor_idn = self.pm.query("SYST:SENS:IDN?") #sensor identity query see page 49 of manual
        
        self.pm.write("CONF:POW") # configures device to measure power

        self.wavelength_min = float(self.pm.query("SENS:CORR:WAV? MIN")) #queries min wavelength
        self.wavelength_max = float(self.pm.query("SENS:CORR:WAV? MAX")) #queried max wavelength
        self.get_wavelength()  # function to fetch wavelength information
        
        self.get_attenuation_dB() #
        
        self.write("SENS:POW:UNIT W") # set to Watts
        self.power_unit = self.pm.query("SENS:POW:UNIT?") # query units being used

        #most commond commands
        self.get_auto_range()
                
        self.get_average_count()
        
        self.get_power_range()        
        self.measure_power()
        self.measure_frequency()
        
        
    
    def ask(self, cmd):
        '''function for querying the device
        kept from old code, may delete later but that's extra work for no gain
        not the use of repr- it turns any object into a string representation
        making it suitable for printing'''
        
        if self.debug: logger.debug( "PM100D ask " + repr(cmd) )
        resp = self.pm.query(cmd)
        if self.debug: logger.debug( "PM100D resp ---> " + repr(resp) )
        return resp
    
    def write(self, cmd):
        '''write command, used when configuring the device post initialization'''
        if self.debug: logger.debug( "PM100D write" + repr(cmd) )
        resp = self.pm.write(cmd)
        if self.debug: logger.debug( "PM100D written --->" + repr(resp))
        
    def get_wavelength(self):
        '''as the name says, gets you the wavelength sensor is set for 
        see page 52 of the manual'''
        try_count = 0
        while True:
            try:
                self.wl = float(self.ask("SENS:CORR:WAV?"))
                if self.debug: logger.debug( "wl:" + repr(self.wl) )
                break
            except:
                if try_count > 9:
                    logger.warning( "Failed to get wavelength." )
                    break
                else:
                    time.sleep(RETRY_SLEEP_TIME)  #take a rest..
                    try_count = try_count + 1
                    logger.debug( "trying to get the wavelength again.." )
        return self.wl
    
    def set_wavelength(self, wl):
        '''see page 52 of manual. func here sets the wavelength'''
        try_count = 0
        while True:
            try:
                self.write("SENS:CORR:WAV %f" % wl)
                time.sleep(0.005) # Sleep for 5 ms before rereading the wl.
                break
            except:
                if try_count > 9:
                    logger.warning( "Failed to set wavelength." )
                    time.sleep(0.005) # Sleep for 5 ms before rereading the wl.
                    break
                else:
                    time.sleep(RETRY_SLEEP_TIME)  #take a rest..
                    try_count = try_count + 1
                    logger.warning( "trying to set wavelength again.." )

        return self.get_wavelength()
    
    def get_attenuation_dB(self):
        '''sets user specified beam attenuation factor
        value specifed inin dB (range for 60db to -60db) gain or attenuation,
        default 0 dB'''
        
        self.attenuation_dB = float( self.ask("SENS:CORR:LOSS:INP:MAGN?") )
        if self.debug: logger.debug( "attenuation_dB " + repr(self.attenuation_dB))
        return self.attenuation_dB

    
    def get_average_count(self):
        """see page 52. Each measurement is approximately 3 ms.
        returns the number of measurements the result is averaged over"""
        self.average_count = int( self.ask("SENS:AVER:COUNt?") )
        if self.debug: logger.debug( "average count:" +  repr(self.average_count))
        return self.average_count
    
    def set_average_count(self, cnt):
        """see page 52. Sets the number of measurements over which the result 
        is averaged"""
        self.write("SENS:AVER:COUNT %i" % cnt)
        return self.get_average_count()
            
    
    def measure_power(self):
        '''the reason we are here, returns the power meausrement'''
        self.power = float(self.ask("MEAS:POW?"))
        if self.debug: logger.debug( "power: " + repr( self.power))
        return self.power
        
    def get_power_range(self):
        '''see page 54. queries the upper limit to power range'''
        self.power_range = self.ask("SENS:POW:RANG:UPP?") 
        if self.debug: logger.debug( "power_range " + repr( self.power_range ))
        return self.power_range


    def set_power_range(self, range):
        #un tested
        self.write("SENS:POW:RANG:UPP {}".format(range))

   
    def get_auto_range(self):
        '''see page 54. queries autro-range function's state'''
        resp = self.ask("SENS:POW:RANG:AUTO?")
        if True:
            logger.debug( 'get_auto_range ' + repr(resp) )
        self.auto_range = bool(int(resp))
        return self.auto_range
    
    def set_auto_range(self, auto = True):
        logger.debug( "set_auto_range " + repr( auto))
        if auto:
            self.write("SENS:POW:RANG:AUTO ON") # turn on auto range
        else:
            self.write("SENS:POW:RANG:AUTO OFF") # turn off auto range
    
    
    def measure_frequency(self):
        '''need to test this function. not sure if it is valid anymore
        I don't see meas:freq? in the manual'''
        self.frequency = self.ask("MEAS:FREQ?")
        if self.debug: logger.debug( "frequency " + repr( self.frequency))
        return self.frequency


    def get_zero_magnitude(self):
        '''command not found in manual
        need to test or del'''
        resp = self.ask("SENS:CORR:COLL:ZERO:MAGN?")
        if self.debug:
            logger.debug( "zero_magnitude " + repr(resp) )
        self.zero_magnitude = float(resp)
        return self.zero_magnitude
        
    def get_zero_state(self): 
        '''test or del; command not in manual'''
        resp = self.ask("SENS:CORR:COLL:ZERO:STAT?")
        if self.debug:
            logger.debug( "zero_state" + repr(resp))
        self.zero_state = bool(int(resp))
        if self.debug:
            logger.debug( "zero_state" + repr(resp) + '--> ' + repr(self.zero_state))
        return self.zero_state
    
    def run_zero(self):
        '''see above'''
        resp = self.ask("SENS:CORR:COLL:ZERO:INIT")
        return resp
    
    def get_photodiode_response(self):
        '''queries the photodiode response value. see page 53 '''
        resp = self.ask("SENS:CORR:POW:PDIOde:RESP?")
        #resp = self.ask("SENS:CORR:VOLT:RANG?")
        #resp = self.ask("SENS:CURR:RANG?")
        if self.debug:
            logger.debug( "photodiode_response (A/W)" + repr(resp) )
        
        self.photodiode_response = float(resp) # A/W
        return self.photodiode_response 
    
    def measure_current(self):
        '''measures current. command changed from ("MEAS:CURR?") 
        to sense. need to test'''
        resp = self.ask("SENSe:CURR?")  #("MEAS:CURR?")
        if self.debug:
            logger.debug( "measure_current " + repr(resp))
        self.current = float(resp)
        return self.current
    
    def get_current_range(self):
        '''Queries the current range see page 53'''
        resp = self.ask("SENS:CURR:RANG:UPP?")
        if self.debug:
            logger.debug( "current_range (A)" + repr(resp))
        self.current_range = float(resp)
        return self.current_range
        
    def close(self):
        '''close instrument communciation'''
        return self.pm.close()

if __name__ == '__main__':
    
    power_meter = ThorlabsPM100D(debug=True)
    power_meter.get_wavelength()
    power_meter.get_average_count()
    power_meter.measure_power()