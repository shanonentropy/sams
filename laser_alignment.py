# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:52:22 2023

@author: zahmed

this code is to be used in the process of nv alignement. It restricts the 
user to <1mW of laser power post circulator or 10% of 

"""
# import modules
import sys
sys.path.append('c:\\sams\instrument_control')

from dlnsec import DLnsec # ,* test if star is necessary

laser = DLnsec('com7')  # check if this is the correct port
laser.set_power(10)
# check initial condition
print('laser is currently off and default laser power is set to {} percent'.format(laser.get_power()))
# set mode
laser.set_mode('LAS')

#prompt user to turn the laser on

while True:
    status = input('Press L to turn the laser on, Press anyother key to turn the laser off =   ')
    if status.lower()=='l':
        print('laser is on')
        laser.on()
    elif status.lower()!='l':
        laser.off()
        print('laser turned off')
        break
