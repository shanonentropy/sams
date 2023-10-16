# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:45:11 2023

@author: zahmed

basic class for processing temp dependent PL data
"""

#import modules
import sys
import re, time, datetime, peakutils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate, optimize
from scipy import stats
#import os
from peakutils.plot import plot as pplot
from scipy.optimize import curve_fit

sys.path.append('c:/sams/data_processing/')
from file_location import filelister




class zpl:
    ''' takes in class twoD and then calculates volume'''
    def __init__(self,path, fn,laser_power_id, tempeature, acq_lenth, grating_center, time_stamp, f_num):
        super().__init__(path, fn)
        self.laser_power_id = re.compile('laser_power_\d*_')
        self.temperature = re.compile('-?\d*,\d*')
        self.acq_lenth = re.compile('\d*.\d*ms')
        self.grating_center = re.compile('CWL\d*.\d*nm')
        self.time_stamp = re.compile('2023 May\s\d* \d*\w\d*_\d*')
        self.f_num = re.compile('Frame-\d*')



    def laser_power(self,x):
        lp = int(str(self.laser_power_id.findall(x)).split('_')[2])
        return lp
    def temp(x):
        t = float((self.temperature.findall(x))[0].replace(',','.'))
        return t
    def acq(x):
        a = float(self.acq_lenth.findall(x)[0].replace('ms',''))
        return(a)
    def grating(x):
        float(self.grating_center.findall(x)[0].strip('CWL').strip('nm'))
        
    def time_st(x):
        g = self.time_stamp.findall(x)[0].split(' ')
        year, month, day, hr = g[0], g[1],g[2], g[3]
        h, m, s = hr.split('_')
        month = 5
        mt = time.mktime(datetime.datetime(int(year), int(month), int(day), int(h),int(m),int(s)).timetuple())
        return mt
    
    def frame_num(x):
        fnm = int(self.f_num.findall(x)[0].split('-')[1])    
        return fnm
    
    def strp_atr(x):
        lp = int(str(self.laser_power_id.findall(x)).split('_')[2])
        t = float((self.temperature.findall(x))[0].replace(',','.'))
        a = float(self.acq_lenth.findall(x)[0].replace('ms',''))
        b = float(self.grating_center.findall(x)[0].strip('CWL').strip('nm'))
        g = self.time_stamp.findall(x)[0].split(' ')
        year, month, day, hr = g[0], g[1],g[2], g[3]
        h, m, s = hr.split('_')
        month = 5
        mt = time.mktime(datetime.datetime(int(year), int(month), int(day), int(h),int(m),int(s)).timetuple())
        fnm = int(self.f_num.findall(x)[0].split('-')[1])
        #append to lists
        return mt,lp,t,a,b,g,fnm
    
    
class fit_funcs:    
    
    '''defines common functions classically used to fit the ZPL'''
    
    def gaussian(x, amp, u, std):
        return amp*np.exp(-((x-u)**2/(2*std**2)))
    
    def two_gaussian(x, amp1, amp2, u1, u2, std1, std2):
        return ((amp1*np.exp(-((x-u1)**2/(2*std1**2))) + (amp2*np.exp(-((x-u2)**2/(2*std2**2))) )))
    
    def lorentzian( x, x0, a, gam ):
        '''x =    '''
        return a * gam**2 / ( gam**2 + ( x - x0 )**2)
      
    def lorentzian_2( x, x0, x01,a,a2, gam, gam2 ):
        '''   '''
        return ( (a * gam**2 / ( gam**2 + ( x - x0 )**2)) + (a2 * gam2**2 / ( gam2**2 + ( x - x01 )**2))) 
    
