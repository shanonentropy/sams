# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:11:56 2023

@author: zahmed

testing dlnsec control software


"""
import modules
import sys
sys.path('c:\\sams\instrument_control')
from dlnsec import DLnsec # ,* test if star is necessary
#from waiting import wait
from time import sleep

laser = DLnsec('com7')  # check if this is the correct port
print(laser.get_power())  # check initial condition

#set initial laser power
p = 1
laser.set_power(p)

# set mode
laser.set_mode('LAS')

# laser action
laser.on()

#turn laser off
laser.off()

#close comm
#laser.close()

# cycle laser on/off and through power levels

power_level = [1,5,10,20,30,50,80,40,20,10,5,1]

laser = DLnsec('com7')
laser.set_mode('LAS')
laser.on()
for p in power_level:
    laser.set_power(p)
    sleep(60)

laser.close()
