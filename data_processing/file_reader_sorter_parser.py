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
from pathlib import Path
#sys.path.append('c:/sams/data_processing/')
#from file_location import Filelister
    
import glob


 
  
class SortList():
    ''' 
    this class is designed to pull in data files as a list that can be fed
    into a pandas to read in and process the spectroscopy data
    
    the default path is c:/sams/saved_data and file type is set to csv
    
    by running filelist.getfiles() you can get a list of all files with 
    txt or csv tags and feed it to pandas
    
    a handful of regex based calls are provided to help sort the files
    and pull relevant information about the measurement from the file names
    '''    
    def __init__(self):
        self.path = input("Enter the path for files: ")
        if not self.path:
            self.path =  'c:/sams/saved_data'
        self.fn =  input('Enter filetype, default filetype is *.csv,')
        self.file_list = []
        if not self.fn:
            self.fn = '/*.csv'
        self.laser_power_id = re.compile('laser_power_\d*_')
        self.temperature = re.compile('-?\d*,\d*')
        self.acq_lenth = re.compile('\d*.\d*ms')
        self.grating_center = re.compile('CWL\d*.\d*nm')
        self.time_stamp = re.compile('2023 May\s\d* \d*\w\d*_\d*')
        self.f_num = re.compile('Frame-\d*')
        #self.file_list = file_list

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
    
    def get_files(self):
        self.f_path = self.path+self.fn
        self.file_list = glob.glob(self.f_path)
        print(self.f_path)
        return self.file_list  






'''
# Example usage
lister = SortList()
files = lister.get_files()
sorted_files = sorted(files, key=lister.temp)  
# Replace `lister.temp` with your desired sorting function
print(sorted_files)

'''

 
    