# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 10:05:20 2022

@author: zahmed
program uses pyvisa to access PM100d 
this class is used in the laser power measurement code

"""

import numpy as np
import pyvisa, usb.core
from ThorlabsPM100 import ThorlabsPM100
#import time


class PM100D():

    def __init__(self, _address='USB0::4883::32888::P0031649::0::INSTR', _timeout=1000, _power_meter=None):
        self._address = _address
        self._timeout = 1000
        self._power_meter = None

    def on_activate(self):
        """ Startup the module """

        rm = pyvisa.ResourceManager()
        self._inst = rm.open_resource(self._address, timeout=self._timeout)
        
        self._power_meter = ThorlabsPM100(inst=self._inst)

    def on_deactivate(self):
        """ Stops the module """
        self._inst.close()

    def getData(self):
        """ SimpleDataInterface function to get the power from the powermeter """
        return np.array([self.get_power()])

    def getChannels(self):
        """ SimpleDataInterface function to know how many data channel the device has, here 1. """
        return 1

    def get_power(self):
        """ Return the power read from the ThorlabsPM100 package """
        return self._power_meter.read

    def get_process_value(self):
        """ Return a measured value """
        return self.get_power()

    def get_process_unit(self):
        """ Return the unit that hte value is measured in as a tuple of ('abreviation', 'full unit name') """
        return ('W', 'watt')

    def get_wavelength(self):
        """ Return the current wavelength in nanometers """
        return self._power_meter.sense.correction.wavelength

    def set_wavelength(self, value=None):
        """ Set the new wavelength in nanometers """
        mini, maxi = self.get_wavelength_range()
        if value is not None:
            if mini <= value <= maxi:
                self._power_meter.sense.correction.wavelength = value
            else:
                self.log.error('Wavelength {} is out of the range [{}, {}].'.format(
                    value, mini, maxi
                ))
        return self.get_wavelength()

    def get_wavelength_range(self):
        """ Return the wavelength range of the power meter in nanometers """
        return self._power_meter.sense.correction.minimum_beamdiameter,\
               self._power_meter.sense.correction.maximum_wavelength