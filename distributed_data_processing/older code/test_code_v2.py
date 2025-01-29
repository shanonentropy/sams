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
from scipy import stats, interpolate, optimize
from scipy.optimize import curve_fit
from scipy.stats import wasserstein_distance
from scipy.integrate import trapz
import peakutils

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

# Load reference spectrum and normalize
df0 = pd.read_csv(files[1], sep=',')
spectrum1 = df0['Intensity'] / np.sum(df0['Intensity'])

# KL Divergence function
def kl_divergence(spectrum, spectrum1):
    spectrum = spectrum / np.sum(spectrum)
    return np.sum(np.log(spectrum1 / spectrum) * spectrum1)

# Gaussian function for fitting
def gaussian(x, amp, u, std):
    return amp * np.exp(-((x - u) ** 2 / (2 * std ** 2)))

def lorentzian(x_zpl, x0, a, gam ):
    '''fits a Lorentzian to the curve   '''
    return a * gam**2 / ( gam**2 + ( x_zpl - x0 )**2)


# Process a single file and extract features
def process_file(file_path, nv_type='nv', fit_params=[4000, 637.5, 1.5], max_fev=50000, func = 'gaussian'):
    if nv_type == 'nv':
        zp = nv_zpl
    else:
        zp = nv0_zpl

    df = pd.read_csv(file_path, sep=',')
    df.sort_values(by='Wavelength', inplace=True)
    df.drop_duplicates(subset='Wavelength', inplace=True)
    
    x, y = df['Wavelength'].values, df['Intensity'].values
    x_zpl_range = (np.abs(x - zp[0])).argmin(), (np.abs(x - zp[1])).argmin()
    x_zpl, y_zpl = x[x_zpl_range[0]:x_zpl_range[1]], y[x_zpl_range[0]:x_zpl_range[1]]
    
    base = peakutils.baseline(y_zpl, 1)
    y_zpl_base = y_zpl - base
    dx_val = (x[0] - x[10]) / 10
    
    area_zpl = trapz(y[x_zpl_range[0]:x_zpl_range[1]], dx=dx_val)
    area_psb = trapz(y[(np.abs(x - huang_rhys[0])).argmin():(np.abs(x - huang_rhys[1])).argmin()], dx=dx_val)
    dw = area_zpl / area_psb
    
    
    if func == 'gaussian':
        popt, _ = optimize.curve_fit(gaussian, x_zpl, y_zpl_base, p0=fit_params, maxfev=max_fev)
        amp, center_wavelength, FWHM = popt
        
    else:
        popt, _ = optimize.curve_fit(lorentzian, x_zpl, y_zpl_base, p0=fit_params, maxfev=max_fev)
        amp, center_wavelength, FWHM = popt
    

        
    return {
        'debye_waller': dw,
        'amplitude': amp,
        'peak_center': center_wavelength,
        'width': FWHM,
        'time': extract_attributes(file_path)[0],
        'laser_power': extract_attributes(file_path)[1],
        'temperature': extract_attributes(file_path)[2],
        'kld_divergence': kl_divergence(y, spectrum1),
        'wasserstein': wasserstein_distance(y, spectrum1),
        'func': func
    }

# Initialize Dask client
client = Client(n_workers=2, threads_per_worker=1, memory_limit='14GB')

# Create and compute delayed tasks
start_time = time.time()
delayed_results = [dask.delayed(process_file)(f, func='lorentzian') for f in files[:1000]]
results = dask.compute(*delayed_results)

client.close()

# Convert results to DataFrame
df2 = pd.DataFrame(results)

# Plotting results
plt.plot(df2['time'], df2['laser_power'])
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Temperature over Time')
plt.show()

print(f"Processing time: {time.time() - start_time:.2f} seconds")

# save the dataframe
export_name = '_'
df2.to_csv('../saved_data/'+export_name)
