# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 12:52:04 2024

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
from dask import delayed

# Start a local Dask cluster with specific configuration
#client = Client(n_workers=1, threads_per_worker=2, memory_limit='40GB')

# Print dashboard link
#print(client)


fpath1 = 'C:/nv_data/data_holder/LN2_bath/*.csv'
fpath2 = 'C:/nv_data/data_holder/LN2_bath_CWL650nm/*.csv'
fpath = 'C:/nv_data/data_holder/cooling_to_LN2_second_attempt_CWL_650nm/*.csv'
f = glob.glob(fpath)

'''
df = pd.read_csv(f[0], sep=',', header=0, usecols=['Wavelength'] )

ddf = dd.read_csv(f[0], sep=',', header=0, usecols=['Wavelength'])

a_ =  time.time()
max_amps_df = []
for i in f:
    df_ = pd.read_csv(i, sep=',', header=0, usecols=['Intensity'])
    max_amp = df_.Intensity.max()
    max_amps_df.append(max_amp)
print(time.time()-a_)



a =  time.time()
max_amps = []


def reader(x):
    ddf_ = dd.read_csv(x, sep=',', header=0, usecols=['Intensity'])
    max_amp = ddf_.Intensity.max().compute()
    max_amps.append(max_amp)
#reader(f)
ddf = dd.read_csv(f, sep=',', header=0, usecols=['Wavelength', 'Intensity'])
ddf.groupby('Wavelength')['Intensity'].max().compute()
print(time.time()-a)


'''
# =============================================================================
# dask opt
# =============================================================================

fnm, time_step = [],[]
laser_pow, amplitude, peak_center =[],[],[]
width, debye_waller, frame_num = [], [],[]
kld, wasserstein_dist =[], []
amplitude2, peak_center2, width2 =[],[],[]
temps = []

filtered_files = f
huang_rhys = [649, 780]
nv_zpl = [634.25,640.25]
nv0_zpl = [572.0, 578]

def reference_spectra(filtered_files, ref_index =1):
    ''' returns the reference spectra for kl_div computation
    it is separated so I can call it once at the start of the computation
    loop and not have to reload it with each new spectra'''
    df0=pd.read_csv(filtered_files[ref_index], sep=',', header = 0, engine='python')
    df0.sort_values(by='Wavelength', ascending=True)
    df0.drop_duplicates(subset='Wavelength', keep='first', inplace=True)
    spectrum1 = df0['Intensity']/np.sum(df0['Intensity'])
    #print('x') # test line to see if this is invoked once or many times
    return spectrum1

reference_spectra(filtered_files)
   
def kl_divergence(y, ref):
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
    kld.append(kl_div)
        

   


def gaussian(x_zpl, amp, u, std):
    ''' gaussian fit'''
    return amp*np.exp(-((x_zpl-u)**2/(2*std**2)))


def main_processor( nv_type='nv', func = 'gaussian', fit_params = [4000, 637.5,1.5], max_fev=50000, dx = 0.01 ):
    ''' nv_type = enter nv for nv(-) or nv0 for nv_zero; default is nv
    
    func is the fitting function used. default is gaussian, other options 
    include lorenztian, two_gaussian or two_lorenztian and spline_fit
    
    fit_parms: default for gaussian
    for lorenzian 
    
    '''
    if nv_type == 'nv':
       zp= nv_zpl
    else:
        zp=nv0_zpl
    ref = reference_spectra(filtered_files)
    
    for f1 in filtered_files[:]:
        #print(f1)
        #fnm.append(f1) #.split('\\')[1])
        #frame_num.append((f1))
        ###### open and clean data ####
        df=pd.read_csv(f1, sep=',', header = 0, engine='python')
        df.sort_values(by='Wavelength', ascending=True)
        df.drop_duplicates(subset='Wavelength', keep='first', inplace=True)
        x,y=df['Wavelength'],df['Intensity']
        ### mark out zpl range of interest #####
        x_zpl, y_zpl = x[(np.abs(x-zp[0])).argmin():(np.abs(x-zp[1])).argmin() ],\
        y[(np.abs(x-zp[0])).argmin():(np.abs(x-zp[1])).argmin() ]

        base = peakutils.baseline(y_zpl, 1)
        y_zpl_base = y_zpl-base
        #time_step.append(time_st(f1))
        #kl_divergence(y, ref)
        #wasserstein_dist.append(wasserstein_distance(y,spectrum1))
        dx_val = (x[0]-x[50])/50
        area_zpl = trapz(y[(np.abs(x-zp[0])).argmin():(np.abs(x-zp[1])).argmin() ], dx= dx_val)
        area_psb = trapz(y[(np.abs(x-huang_rhys[0])).argmin():(np.abs(x-huang_rhys[1])).argmin() ], dx= dx_val)
        dw = area_zpl/area_psb
        debye_waller.append(dw); 
        if func == 'gaussian': 
             popt, pcov = curve_fit(gaussian,x_zpl, y_zpl_base, [4000, 637.5,1.5], maxfev=max_fev )
             amp, center_wavelength, FWHM = popt
             peak_center.append(center_wavelength);
             width.append(FWHM);
             amplitude.append(amp);
             #lp = laser_power(f1)
             #laser_pow.append(lp)
             #temps.append(float(temp(f1)))


a =time.time()
main_processor()
b = time.time() - a



# =============================================================================
# dask
# =============================================================================


from dask.distributed import Client, progress
from dask import delayed
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import trapz
import peakutils
import dask

# Initialize Dask client with dashboard
client = Client(n_workers=4, threads_per_worker=2, memory_limit='40GB', dashboard_address=':8787')

def process_file(f1, nv_type='nv', func='gaussian', fit_params=[4000, 637.5, 1.5], max_fev=50000, dx=0.01):
    ''' Process a single file and extract features '''
    
    if nv_type == 'nv':
        zp = nv_zpl
    else:
        zp = nv0_zpl
    
    df = pd.read_csv(f1, sep=',', header=0, engine='python')#.compute()
    df = df.sort_values(by='Wavelength', ascending=True).drop_duplicates(subset='Wavelength', keep='first')
    x, y = df['Wavelength'], df['Intensity']
    
    # Mark out ZPL range of interest
    x_zpl_range = (np.abs(x - zp[0])).argmin(), (np.abs(x - zp[1])).argmin()
    x_zpl, y_zpl = x[x_zpl_range[0]:x_zpl_range[1]], y[x_zpl_range[0]:x_zpl_range[1]]
    
    base = peakutils.baseline(y_zpl, 1)
    y_zpl_base = y_zpl - base
    
    dx_val = (x[0] - x[50]) / 50
    area_zpl = trapz(y[x_zpl_range[0]:x_zpl_range[1]], dx=dx_val)
    area_psb = trapz(y[(np.abs(x - huang_rhys[0])).argmin():(np.abs(x - huang_rhys[1])).argmin()], dx=dx_val)
    dw = area_zpl / area_psb
    
    result = {'debye_waller': dw}
    
    if func == 'gaussian':
        def gaussian(x_zpl, amp, u, std):
            return amp * np.exp(-((x_zpl - u) ** 2 / (2 * std ** 2)))
        
        popt, _ = curve_fit(gaussian, x_zpl, y_zpl_base, p0=fit_params, maxfev=max_fev)
        amp, center_wavelength, FWHM = popt
        
        result.update({'amplitude': amp, 'peak_center': center_wavelength, 'width': FWHM})
    
    return result

# List of files to process
files = filtered_files




a =time.time()
main_processor()
b = time.time() - a



c =time.time()
# Create and compute delayed tasks
delayed_results = [delayed(process_file)(f) for f in files]
results = dask.compute(*delayed_results)

# Monitor task progress
progress(delayed_results)

# Compute results

#results = dask.compute(*delayed_results)


d = time.time() - c



print(b, d)



client.close()

#client = Client(n_workers=2, threads_per_worker=2, memory_limit='40GB')
'1:2:199'
'4:2:249.4'
'4:1:252'