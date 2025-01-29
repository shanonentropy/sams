# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 09:29:04 2024

@author: zahmed
"""
#import modules
import glob
import dask
import dask.dataframe as dd
import matplotlib.pyplot as plt
import time
import pandas as pd
from dask.distributed import Client
import numpy as np
from dask import delayed
import peakutils
import numpy as np
import pandas as pd
from scipy import interpolate, optimize
from scipy import stats
from peakutils.plot import plot as pplot
from scipy.optimize import curve_fit
from scipy.stats import wasserstein_distance
from numpy import trapz
import sys
from scipy import linalg
import re, time, datetime, peakutils
#set filtering conditions

k = ['Frame-00001', 'Frame-0001'] 
huang_rhys = [649, 780]
nv_zpl = [634.25,640.25]
nv0_zpl = [572.0, 578]



laser_power_id = re.compile('laser_power_\d*_')
temperature = re.compile('-?\d*,\d*')
acq_lenth = re.compile('\d*.\d*ms')
grating_center = re.compile('CWL\d*.\d*nm')
time_stamp = re.compile('202\d \w*\s\d* \d*\w\d*_\d*')
f_num = re.compile('Frame-\d*')
#sort_index = self.sort_index
#file_list = file_list
   
def laser_power(x):
    lp = int( str(laser_power_id.findall(x)).split('_')[2])
    return lp

def temp(x):
    t = float((temperature.findall(x))[0].replace(',', '.'))
    return t

def acq( x):
    a = float(acq_lenth.findall(x)[0].replace('ms', ''))
    return a
def grating(x):
    float(grating_center.findall(x)[0].strip('CWL').strip('nm'))

def file_num(x):
    fn = f_num.findall(x)[0].split('-')[1]   
    return int(fn)

def time_st(x):
    g = time_stamp.findall(x)[0].split(' ')
    year, month_str, day, hr = g[0], g[1],g[2], g[3]
    h, m, s = hr.split('_')
    month_cal ={'January':1, 'Feburary':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7, 'August':8, 'September':9, 'Octuber':10, 'November':11, 'December':12}
    month = month_cal[month_str]
    mt = time.mktime(datetime.datetime(int(year), int(month), int(day), int(h),int(m),int(s)).timetuple())
    return mt

def strp_atr(x):
    lp = int(str(laser_power_id.findall(x)).split('_')[2])
    t = float((temperature.findall(x))[0].replace(',','.'))
    a = float(acq_lenth.findall(x)[0].replace('ms',''))
    b = float(grating_center.findall(x)[0].strip('CWL').strip('nm'))
    g = time_stamp.findall(x)[0].split(' ')
    year, month_str, day, hr = g[0], g[1],g[2], g[3]
    h, m, s = hr.split('_')
    #month_cal ={'Jan':1, 'Feb':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7, 'August':8, 'Sept':9, 'Oct':10, 'Nov':11, 'Dec':12}
    month_cal ={'January':1, 'Feburary':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7, 'August':8, 'Sept':9, 'Oct':10, 'Nov':11, 'Dec':12}
    month = month_cal[month_str]
    mt = time.mktime(datetime.datetime(int(year), int(month), int(day), int(h),int(m),int(s)).timetuple())
    fnm = int(f_num.findall(x)[0].split('-')[1])
    #append to lists
    return mt,lp,t,a,b,g,fnm


#fpath1 = '../sensor2_week5_LN2/*.csv'
#fpath2 = 'C:/nv_data/data_holder/LN2_bath_CWL650nm/*.csv'
#fpath = 'C:/nv_data/data_holder/cooling_to_LN2_second_attempt_CWL_650nm/*.csv'
fpath = 'C:/nv_data/nv_sensor_2/sensor2_12192023week1/sensor2_12192023week1/laser_power_loop/*.csv'

f = glob.glob(fpath)


#filter files func
def filter_list(f):
    ''' drop any files with a particular key in them in their names'''
    filenames = sorted(f, key=time_st)
    if len(k) <2:
        filtered_files = [name for name in filenames if k not in name]
    else: 
        filtered_files = [name for name in filenames if all(k not in name for k in k)]
    return filtered_files 



files_ = filter_list(f)

files = sorted(files_, key=time_st) 

df0=pd.read_csv(files[1], sep=',', header = 0)
df0.sort_values(by='Wavelength', ascending=True)
df0.drop_duplicates(subset='Wavelength', keep='first', inplace=True)
spectrum1 = df0['Intensity']/np.sum(df0['Intensity'])
print('x') # test line to see if this is invoked once or many times


   
def kl_divergence(y, spectrum1):
    ''' spectrum1 is the reference file
        spectrum2 is the current file in the loop being processed
        
        need to write the loop here
        
        '''
    #df_=pd.read_csv(spectrum2, sep=',', header = 0, engine='python')
    #df_.sort_values(by='Wavelength', ascending=True)
    #df_.drop_duplicates(subset='Wavelength', keep='first', inplace=True)
    # Normalize the spectra
    spectrum = y  / np.sum(y)
    
    # Calculate the logarithm of the ratio of the two spectra
    ratio = np.log(spectrum1 / spectrum)
    # Multiply the ratio by the normalized spectra
    result = ratio*spectrum1
    # Sum the resulting values to obtain the KL divergence
    kl_div = np.sum(result)
    return(kl_div)
        

   


def gaussian(x_zpl, amp, u, std):
    ''' gaussian fit'''
    return amp*np.exp(-((x_zpl-u)**2/(2*std**2)))


#


def process_file(f1, nv_type='nv', func='gaussian', fit_params=[4000, 637.5, 1.5], max_fev=50000, dx=0.01):
    ''' Process a single file and extract features '''
    
    if nv_type == 'nv':
        zp = nv_zpl
    else:
        zp = nv0_zpl
    
    df = pd.read_csv(f1, sep=',', header=0)
    df = df.sort_values(by='Wavelength', ascending=True).drop_duplicates(subset='Wavelength', keep='first')
    x, y = df['Wavelength'], df['Intensity']
    
    # Mark out ZPL range of interest
    x_zpl_range = (np.abs(x - zp[0])).argmin(), (np.abs(x - zp[1])).argmin()
    x_zpl, y_zpl = x[x_zpl_range[0]:x_zpl_range[1]], y[x_zpl_range[0]:x_zpl_range[1]]
        
    base = peakutils.baseline(y_zpl, 1)
    y_zpl_base = y_zpl - base
    
    dx_val = (x[0] - x[10]) / 10
    area_zpl = trapz(y[x_zpl_range[0]:x_zpl_range[1]], dx=dx_val)
    area_psb = trapz(y[(np.abs(x - huang_rhys[0])).argmin():(np.abs(x - huang_rhys[1])).argmin()], dx=dx_val)
    dw = area_zpl / area_psb
    
    result = {'debye_waller': dw}
    
    if func == 'gaussian':
        def gaussian(x_zpl, amp, u, std):
            return amp * np.exp(-((x_zpl - u) ** 2 / (2 * std ** 2)))
        
        popt, _ = curve_fit(gaussian, x_zpl, y_zpl_base, p0=fit_params, maxfev=max_fev)
        amp, center_wavelength, FWHM = popt
        
        result.update({'amplitude': amp, 'peak_center': center_wavelength, 
                       'width': FWHM,'time': time_st(f1), 
                       'laser_pow': laser_power(f1), 
                       'temperature': float(temp(f1)),
                       'kld_divergence': kl_divergence(y, spectrum1),'wasserstein': wasserstein_distance(y, spectrum1)}) 
    
    return result


# Initialize Dask client with dashboard
client = Client(n_workers=2, threads_per_worker=2, memory_limit='8GB')

c =time.time()
# Create and compute delayed tasks
delayed_results = [delayed(process_file)(f) for f in files[:]]
results = dask.compute(*delayed_results)


client.close()

print(results)
print(time.time()-c)

df2 = pd.DataFrame(results)
df2.count()

plt.plot(df2.time, df2.temperature)