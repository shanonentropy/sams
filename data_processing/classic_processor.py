# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 13:23:00 2023

# Example usage
lister = SortList()
files = lister.get_files()
sorted_files = sorted(files, key=lister.temp)  
# Replace `lister.temp` with your desired sorting function
print(sorted_files)

# to extract info from an individual file 
t = lister.temp(files[0])




"""
import peakutils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate, optimize
from scipy import stats
from peakutils.plot import plot as pplot
from scipy.optimize import curve_fit
import sys







   
class processor:
    
    '''this class takes in the list of files provided by the Sortlist(FileLister) class
    it reads in the data into pandas and then with the chosen fit function it extracts
    the relevant ZPL parameter and outputs it to a file. 
    This class can process both the NV and NV_zero through NV is the default.
    To select NV_zero pass the keyword "nv_zero" to nv_type
    
    nv_zpl and nv0_zpl provide a list with start and end range of respective zpl
    similarly hyang_rhys provides a list with start and end range over which the sideband
    intensity is integrated.
    
    filesnames is set to "None"; here you pass in the list of filenames globbed
    from file-reader_sorter_parser
    
    Available functions include:
        
        spline_fit: cubic spline fit
        gaussian = 
        two_gaussians=
        lorentzian =
        two_lorentzian = 
        
        nv_zpl = [634.25,640.25]
        nv0_zpl = [572.0, 578]
        huang_rhys = [649, 780]
        
        
        to do: self the functions
        implement fitting 
        add notes 
        then add plotter fuctions or just create a plotter class
        
        
        
        
        
        '''
    
    def __init__(self, nv_type = 'nv', nv_zpl = [634.25,640.25], nv0_zpl = [572.0, 578], huang_rhys = [649, 780], filenames='None', ):
        self.nv_zpl = nv_zpl
        self.nv0_zpl = nv0_zpl
        self.huang_rhys = huang_rhys
        self.filenames = filenames
        self.nv_type = nv_type
        self.filename, self.fnm, self.time_step = [],[],[]
        self.laser_pow, self.amplitude, self.peak_center =[],[],[]
        self.width, self.debye_waller, self.fnm = [], [],[]
        
    
    def gaussian(self, x_zpl, amp, u, std):
        ''' '''
        return amp*np.exp(-((x_zpl-u)**2/(2*std**2)))
    
    def two_gaussian(self,x_zpl, amp1, amp2, u1, u2, std1, std2):
        return ((amp1*np.exp(-((x-u1)**2/(2*std1**2))) + (amp2*np.exp(-((x-u2)**2/(2*std2**2))) )))
    
    def lorentzian(self, x_zpl, x0, a, gam ):
        '''x =    '''
        return a * gam**2 / ( gam**2 + ( x_zpl - x0 )**2)
      
    def two_lorentzian(self, x_zpl, x0, x01,a,a2, gam, gam2 ):
        '''   '''
        return ( (a * gam**2 / ( gam**2 + ( x_zpl - x0 )**2)) + (a2 * gam2**2 / ( gam2**2 + ( x_zpl - x01 )**2))) 
    
    def spline_fit(self):
        ''' '''
        tck_zpl = interpolate.splrep(x_zpl,y_zpl_base,s=0.0001) # s =m-sqrt(2m) where m= #datapts and s is smoothness factor
        x_zpl_sim = np.arange (np.min(x_zpl),np.max(x_zpl), 0.1)
        y_zpl_sim = interpolate.splev(x_zpl_sim, tck_zpl, der=0)
        HM=(np.max(y_zpl_sim)-np.min(y_zpl_sim))/2
        w= splrep(x_zpl_sim, y_zpl_sim - HM, k=3)
        if len(sproot(w))==2:
            r1,r2= sproot(w)
            FWHM=np.abs(r1-r2)
            center_wavelength = r1 + FWHM/2
            peak_center.append(center_wavelength)
            width.append(FWHM)
        else:
            peak_center.append('NaN')
            width.append('NaN')
            print('pass');pass
            
        
    def processor(self, nv_type=self.nv_type, func = 'gaussian', fit_params = [4000, 637.5,1.5], max_fev=50000 ):
        ''' nv_type = enter nv for nv(-) or nv0 for nv_zero; default is nv
        
        func is the fitting function used. default is gaussian, other options 
        include lorenztian, two_gaussian or two_lorenztian and spline_fit
        
        fit_parms: default for gaussian
        for lorenzian 
        
        '''
        if self.nvtype == 'nv':
           zpl= self.nv_zpl
        else:
            zpl=self.nv0_zpl
        for f in self.filenames[1:]:
            #print(f)
            self.fnm.append(f.split('\\')[1])
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
            self.time_step.append(self.frame_num(f))
            if func == 'gaussian':
                 self.popt, self.pcov = curve_fit(self.gaussian,x_zpl, y_zpl_base, [4000, 637.5,1.5], maxfev=max_fev )
                 self.amp, self.center_wavelength, self.FWHM = self.popt
                 self.peak_center.append(self.center_wavelength);
                 self.width.append(self.FWHM);
                 self.amplitude.append(self.amp);
                 self.laser_pow.append(self.laser_power(f))