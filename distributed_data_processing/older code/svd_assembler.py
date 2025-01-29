# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:36:33 2024

@author: zahmed
"""

import glob
import dask
import dask.dataframe as dd
import matplotlib.pyplot as plt
import time
import pandas as pd
from dask.distributed import Client
import numpy as np
import re
import datetime


# Set filtering conditions
k = ['Frame-00001', 'Frame-0001'] 
huang_rhys = [649, 780]
nv_zpl = [634.25, 640.25]
nv0_zpl = [572.0, 578]

# Compile regular expressions
laser_power_id = re.compile('laser_power_\\d*_')
temperature = re.compile('-?\\d*,\\d*')
acq_length = re.compile('\\d*\\.\\d*ms')
grating_center = re.compile('CWL\\d*\\.\\d*nm')
time_stamp = re.compile('202\\d \\w*\\s\\d* \\d*\\w\\d*_\\d*')
f_num = re.compile('Frame-\\d*')

# Functions to extract attributes
def extract_attributes(filename):
    lp = int(str(laser_power_id.findall(filename)).split('_')[2])
    t = float(temperature.findall(filename)[0].replace(',', '.'))
    a = float(acq_length.findall(filename)[0].replace('ms', ''))
    b = float(grating_center.findall(filename)[0].strip('CWL').strip('nm'))
    
    g = time_stamp.findall(filename)[0].split(' ')
    year, month_str, day, hr = g[0], g[1], g[2], g[3]
    h, m, s = hr.split('_')
    month = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 
              'May': 5, 'June': 6, 'July': 7, 'August': 8, 
              'September': 9, 'October': 10, 'November': 11, 'December': 12}[month_str]
    
    mt = time.mktime(datetime.datetime(int(year), month, int(day), int(h), int(m), int(s)).timetuple())
    fnm = int(f_num.findall(filename)[0].split('-')[1])
    
    return mt, lp, t, a, b, fnm

# File filtering function
def filter_files(file_list):
    filtered_files = sorted([f for f in file_list if all(kw not in f for kw in k)], key=lambda x: extract_attributes(x)[0])
    return filtered_files 

# File path
fpath = 'C:/nv_data/nv_sensor_2/sensor2_12192023week1/sensor2_12192023week1/laser_power_loop/*.csv'
files = filter_files(glob.glob(fpath))

#provide a name for svd_data_matrix file
export_name = 'default'


# Load reference spectrum and normalize
df0 = pd.read_csv(files[1], sep=',' ,  usecols= ['Wavelength'])



# assemble svd_matrix
def process_svd_matrix(file_path, nv_type='nv', fit_params=[4000, 637.5, 1.5], max_fev=50000):
    
    df_ = pd.read_csv(file_path, sep=',', usecols=['Intensity'])
    df_.columns = [extract_attributes(file_path)[2]]
    return df_
    
# Initialize Dask client
client = Client(n_workers=1, threads_per_worker=1, memory_limit='14GB')



# Create and compute delayed tasks
start_time = time.time()
delayed_results = [dask.delayed(process_svd_matrix)(f) for f in files[:]]
results = dask.compute(*delayed_results)
df__ = pd.concat(results, axis=1)
df_s = pd.concat([df0,df__], axis = 1)

df_sm = df_s.T
#col_header = df_sm.iloc[0]
df_sm.columns = df_sm.iloc[0]
df_sm.drop('Wavelength', inplace=True)
df_sm = df_sm - df_sm.mean()

#save the data matrix

save_path = '../saved_data/demeaned_data_matrix'+export_name
df_sm.to_csv(save_path)

print(time.time() -  start_time)

client.close()

