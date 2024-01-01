# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 09:09:52 2023

@author: zahmed


this code will generate an array of either single ramp of temp
or a pattern of thermal cycles that the ZPL or ESR code will iterate over
to map out the thermal behaviour of NV diamond

The code requires as an input start and end points of temperature scale,
step size and number of cycles. Outputs an array
"""
# import modules

import numpy as np


class Cycling:

    def __init__(self, start=-30, stop=70, step=5, cycles =3):
        ''' base parameters  '
        start =  lowest temp, default is set to -30 C
        stop =   highest temp, default is set to 70 C
        step =   temp step size, default is 5 C
        cycles = number of cycles, default is 0 returning an array
        there are two funcs:
        * params: prints out the parameters
        * temperatures: generates the temp profile as an array
        '''

        self.start = start # start temperature
        self.stop = stop # stop temperature
        self.step = step # step size
        self.cycles = cycles # number of cycles
        self.temp_ramp = np.arange(self.start, self.stop+self.step, self.step) # generates a single ramp
        self.temp_cycle = np.concatenate((self.temp_ramp, self.temp_ramp[::-1][1:])) #generates a single cycle
        self.temp_cycles_minus_one = np.tile(self.temp_cycle[:-1],self.cycles) # generates a pattern of replicate cycles, but its missing the lowest temp index
        self.temp_cycles = np.concatenate((self.temp_cycle,self.temp_cycle[-1:])) # tacks on the lowest temp val
    def params(self):
        ''' provides access to base parameters being executed '''
        print('start = {}'.format( self.start))
        print('stop = {}'.format( self.stop))
        print('step = {}'.format( self.step))
        print('cycles ={}'.format(self.cycles))
        print('the temperature profile of a single cycle is: ',self.temp_cycle)
    def temperatures(self):
        if self.cycles==0:
            #print('array generated', self.temp_ramp)
            return self.temp_ramp
        elif self.cycles ==1:
            #print('single cycle requested', self.temp_cycle)
            return self.temp_cycle
        else:
            #print('cycle generated', self.temp_cycles)
            return self.temp_cycles
