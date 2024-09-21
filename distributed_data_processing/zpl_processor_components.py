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
from file_location import Filelister


class SortList(FileLister):
    ''' takes in class twoD and then calculates volume'''
    def __init__(self, file_list):
        super().__init__()
        self.laser_power_id = re.compile('laser_power_\d*_')
        self.temperature = re.compile('-?\d*,\d*')
        self.acq_lenth = re.compile('\d*.\d*ms')
        self.grating_center = re.compile('CWL\d*.\d*nm')
        self.time_stamp = re.compile('2023 May\s\d* \d*\w\d*_\d*')
        self.f_num = re.compile('Frame-\d*')
        self.file_list = file_list

    def laser_power(self):
        lp = int(str(self.laser_power_id.findall(self.file_list)).split('_')[2])
        print(self.file_list)
        print(lp)
        return lp
    
    def temp(self, x):
        t = float((self.temperature.findall(x))[0].replace(',', '.'))
        return t
    
    def acq(self, x):
        a = float(self.acq_lenth.findall(x)[0].replace('ms', ''))
        return a
    
    def grating(self, x):
        float(self.grating_center.findall(x)[0].strip('CWL').strip('nm'))

# Example usage
lister = SortList([])
files = lister.get_files()
sorted_files = sorted(files, key=lister.temp)  
# Replace `lister.temp` with your desired sorting function
print(sorted_files)



 
    
    
class fit_funcs:   
    ''' in this class we provide a list of common functions that be called 
    to fit the ZPL. note: this class is for traditional analysis 
    
    available functions include gaussian(s), lorenztian(s), and spline fit'''
    
    def __init__(self,x):
        self.x = x
    
    def gaussian(self, x, amp, u, std):
        return amp*np.exp(-((x-u)**2/(2*std**2)))
    
    def two_gaussian(self, x, amp1, amp2, u1, u2, std1, std2):
        return ((amp1*np.exp(-((x-u1)**2/(2*std1**2))) + (amp2*np.exp(-((x-u2)**2/(2*std2**2))) )))
    
    def lorentzian(self,  x, x0, a, gam ):
        return a * gam**2 / ( gam**2 + ( x - x0 )**2)
    
    
    def lorentzian_2(self,  x, x0, x01,a,a2, gam, gam2 ):
        return ( (a * gam**2 / ( gam**2 + ( x - x0 )**2)) + (a2 * gam2**2 / ( gam2**2 + ( x - x01 )**2))) 
    
    def spline_fit(self):
        
    def processor(self, zpl, filenames, fnm, time_step, laser_pow, amplitude, peak_center, width):
        ''' zpl range list nv_zpl for nv negative and nv0_zpl nv_negative
        filename specifies the container,
        nv is the output list holder nv_decomp is for nv_negative and nv0_decomp is for nv_zero  '''
        for f in filenames[1:]:
            #print(f)
            fnm.append(f.split('\\')[1])
            ###### open and clean data ####
            df=pd.read_csv(f, sep=',', header = 0, engine='python')
            df.sort_values(by='Wavelength', ascending=True)
            df.drop_duplicates(subset='Wavelength', keep='first', inplace=True)
            x,y=df['Wavelength'],df['Intensity']
            ### mark out zpl range of interest #####
            x_zpl, y_zpl = x[(np.abs(x-zpl[0])).argmin():(np.abs(x-zpl[1])).argmin() ],\
            y[(np.abs(x-zpl[0])).argmin():(np.abs(x-zpl[1])).argmin() ]
            indexes = peakutils.indexes(y_zpl, thres=100, min_dist=3)
            #plt.figure(figsize=(10,6))
            #pplot(x_zpl, y_zpl, indexes)
            ##### remove baseline #########
            base = peakutils.baseline(y_zpl, 1)
            y_zpl_base = y_zpl-base
            #plt.figure(figsize=(10,6))
            plt.plot(x_zpl, y_zpl_base)
            plt.title("ZPL data with baseline removed")
            time_step.append(frame_num(f))
            popt, pcov = curve_fit(gaussian,x_zpl, y_zpl_base, [4000, 637.5,1.5], maxfev=50000 )
            amp, center_wavelength, FWHM = popt
            peak_center.append(center_wavelength);
            width.append(FWHM);
            amplitude.append(amp);
            laser_pow.append(laser_power(f))