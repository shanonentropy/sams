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

in main processor self.filenames needs to be changed to handle filetered files


"""
import peakutils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate, optimize
from scipy import stats
from peakutils.plot import plot as pplot
from scipy.optimize import curve_fit
from scipy.stats import wasserstein_distance
from numpy import trapz
import sys
from file_reader_sorter_parser import SortList


class processor(SortList):
    
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
        gaussian = single gaussian 
        two_gaussians= two gaussian functions fits are not so good
        lorentzian = single lorentzian fits generally are bad
        two_lorentzian = two lorentizans fits are terrible
        
        nv_zpl = [634.25,640.25]
        nv0_zpl = [572.0, 578]
        huang_rhys = [649, 780]
        
        
        to do: 
        * implement fitting   --- validate fitting routines --- simplify the code 
            --------- check gaussian fit against traditional fit
        * add notes ---
        * implement kld computation
        * implement wasserstein distance computation
        * implement debye_waller thermometry protocol
        * add a save lists as a data frame function
        * add plotter fuctions or just create a plotter class that inherits processor class
        * save plots
        
        
        '''
    
    def __init__(self, nv_type = 'nv', nv_zpl = [634.25,640.25], nv0_zpl = [572.0, 578], huang_rhys = [649, 780], k = 'Frame-00001' ):
        super().__init__()
        self.nv_zpl = nv_zpl
        self.nv0_zpl = nv0_zpl
        self.huang_rhys = huang_rhys
        self.nv_type = nv_type
        lister = SortList()
        files = lister.get_files()
        self.filenames = sorted(files, key=lister.time_st)  
        self.fnm, self.time_step = [],[]
        self.laser_pow, self.amplitude, self.peak_center =[],[],[]
        self.width, self.debye_waller, self.frame_num = [], [],[]
        self.kld, self.wasserstein_dist =[], []
        self.amplitude2, self.peak_center2, self.width2 =[],[],[]
        self.temps = []
        self.k = k
        
        '''
        
    def sorted_files(self):
        #sort files by desired key
        self.sort_files = sorted(self.get_files(), key=self.strp_atr)
        return self.sort_files'''
    
    def filter_list(self):
        ''' drop any files with a particular key in them in their names'''
        self.filtered_files = [name for name in self.filenames if self.k not in name]
        return self.filtered_files    

    def reference_spectra(self, ref_index =1):
        ''' returns the reference spectra for kl_div computation
        it is separated so I can call it once at the start of the computation
        loop and not have to reload it with each new spectra'''
        df0=pd.read_csv(self.filtered_files[ref_index], sep=',', header = 0, engine='python')
        df0.sort_values(by='Wavelength', ascending=True)
        df0.drop_duplicates(subset='Wavelength', keep='first', inplace=True)
        self.spectrum1 = df0['Intensity']/np.sum(df0['Intensity'])
        #print('x') # test line to see if this is invoked once or many times
        return self.spectrum1
    
        
       
    def kl_divergence(self,y):
        ''' spectrum1 is the reference file
            spectrum2 is the current file in the loop being processed
            
            need to write the loop here♣
            
            '''
        #df_=pd.read_csv(spectrum2, sep=',', header = 0, engine='python')
        #df_.sort_values(by='Wavelength', ascending=True)
        #df_.drop_duplicates(subset='Wavelength', keep='first', inplace=True)
        # Normalize the spectra
        spectrum = y  / np.sum(y)
        # Calculate the logarithm of the ratio of the two spectra
        ratio = np.log(self.spectrum1 / spectrum)
        # Multiply the ratio by the normalized spectra
        result = ratio * self.spectrum1
        # Sum the resulting values to obtain the KL divergence
        kl_div = np.sum(result)
        self.kld.append(kl_div)
        
        

    
    def covariance_spectral_plot(self):
        ''' re-write this func to compute the sync and async 
        components of the '''
        
            
            
        #return self.corr_list
        pass    
    def svd_data_matrix(self):
        '''this function will assemble the all measurements into a data matrix to be used
        in svd and pca computations'''
        pass
    
    
    def svd_computation(self):
        ''' this function will take in the data matrix from svd_data_matrix
        and then compute the svd. User is expected to then decide where to 
        place the trunation'''
        
        pass
    
    def svd_truncated(self):
        '''this function implements a truncated svd compuation '''
    
        pass
    
    def latent_var_marg(self):
        ''' interprets spectral diffusivity as variance of means to 
        estimate strain/temp dependence'''
        
        pass
    
    
    def gaussian(self, x_zpl, amp, u, std):
        ''' '''
        return amp*np.exp(-((x_zpl-u)**2/(2*std**2)))
    
    def two_gaussian(self,x_zpl, amp1, amp2, u1, u2, std1, std2):
        return ((amp1*np.exp(-((x_zpl-u1)**2/(2*std1**2))) + (amp2*np.exp(-((x_zpl-u2)**2/(2*std2**2))) )))
    
    def lorentzian(self, x_zpl, x0, a, gam ):
        '''x =    '''
        return a * gam**2 / ( gam**2 + ( x_zpl - x0 )**2)
      
    def two_lorentzian(self, x_zpl, x0, x01,a,a2, gam, gam2 ):
        '''   '''
        return ( (a * gam**2 / ( gam**2 + ( x_zpl - x0 )**2)) + (a2 * gam2**2 / ( gam2**2 + ( x_zpl - x01 )**2))) 
    
    def spline_fit(self, x_zpl, y_zpl_base):
        ''' '''
        tck_zpl = interpolate.splrep(x_zpl,y_zpl_base,s=0.0001) # s =m-sqrt(2m) where m= #datapts and s is smoothness factor
        x_zpl_sim = np.arange (np.min(x_zpl),np.max(x_zpl), 0.1)
        y_zpl_sim = interpolate.splev(x_zpl_sim, tck_zpl, der=0)
        HM=(np.max(y_zpl_sim)-np.min(y_zpl_sim))/2
        w= splrep(x_zpl_sim, y_zpl_sim - HM, k=3)
        if len(sproot(w))==2:
            r1,r2= sproot(w)
            self.FWHM=np.abs(r1-r2)
            self.center_wavelength = r1 + self.FWHM/2
            self.peak_center.append(self.center_wavelength)
            self.width.append(self.FWHM)
            dx_val = (x[0]-x[50])/50
            area_zpl = trapz(y[(np.abs(x-zp[0])).argmin():(np.abs(x-zp[1])).argmin() ], dx= dx_val)
            area_psb = trapz(y[(np.abs(x-self.huang_rhys[0])).argmin():(np.abs(x-self.huang_rhys[1])).argmin() ], dx= dx_val)
            dw = area_zpl/area_psb
            self.debye_waller.append(dw)
        else:
            self.peak_center.append('NaN')
            self.width.append('NaN')
            dx_val = (x[0]-x[50])/50
            area_zpl = trapz(y[(np.abs(x-zp[0])).argmin():(np.abs(x-zp[1])).argmin() ], dx= dx_val)
            area_psb = trapz(y[(np.abs(x-self.huang_rhys[0])).argmin():(np.abs(x-self.huang_rhys[1])).argmin() ], dx= dx_val)
            dw = area_zpl/area_psb
            self.debye_waller.append(dw)
            print('pass');pass
            
            
        
    def main_processor(self, nv_type='nv', func = 'gaussian', fit_params = [4000, 637.5,1.5], max_fev=50000, dx = 0.01 ):
        ''' nv_type = enter nv for nv(-) or nv0 for nv_zero; default is nv
        
        func is the fitting function used. default is gaussian, other options 
        include lorenztian, two_gaussian or two_lorenztian and spline_fit
        
        fit_parms: default for gaussian
        for lorenzian 
        
        '''
        if self.nv_type == 'nv':
           zp= self.nv_zpl
        else:
            zp=self.nv0_zpl
        ref = self.reference_spectra()
        
        for f1 in self.filtered_files[:]:
            print(f1)
            self.fnm.append(f1.split('\\')[1])
            self.frame_num.append(self.file_num(f1))
            ###### open and clean data ####
            df=pd.read_csv(f1, sep=',', header = 0, engine='python')
            df.sort_values(by='Wavelength', ascending=True)
            df.drop_duplicates(subset='Wavelength', keep='first', inplace=True)
            x,y=df['Wavelength'],df['Intensity']
            ### mark out zpl range of interest #####
            x_zpl, y_zpl = x[(np.abs(x-zp[0])).argmin():(np.abs(x-zp[1])).argmin() ],\
            y[(np.abs(x-zp[0])).argmin():(np.abs(x-zp[1])).argmin() ]
            #indexes = peakutils.indexes(y_zpl, thres=100, min_dist=3)
            #plt.figure(figsize=(10,6))
            #pplot(x_zpl, y_zpl, indexes)
            ##### remove baseline #########
            base = peakutils.baseline(y_zpl, 1)
            y_zpl_base = y_zpl-base
            #plt.figure(figsize=(10,6))
            plt.plot(x_zpl, y_zpl_base)
            plt.title("ZPL data with baseline removed")
            self.time_step.append(self.time_st(f1))
            self.kl_divergence(y)
            self.wasserstein_dist.append(wasserstein_distance(y, self.spectrum1))
            dx_val = (x[0]-x[50])/50
            area_zpl = trapz(y[(np.abs(x-zp[0])).argmin():(np.abs(x-zp[1])).argmin() ], dx= dx_val)
            area_psb = trapz(y[(np.abs(x-self.huang_rhys[0])).argmin():(np.abs(x-self.huang_rhys[1])).argmin() ], dx= dx_val)
            dw = area_zpl/area_psb
            self.debye_waller.append(dw)
            if func == 'gaussian': 
                 self.popt, self.pcov = curve_fit(self.gaussian,x_zpl, y_zpl_base, [4000, 637.5,1.5], maxfev=max_fev )
                 self.amp, self.center_wavelength, self.FWHM = self.popt
                 self.peak_center.append(self.center_wavelength);
                 self.width.append(self.FWHM);
                 self.amplitude.append(self.amp);
                 lp = self.laser_power(f1)
                 self.laser_pow.append(lp)
                 self.temps.append(self.temp(f1))
                 
            elif func == 'lorentzian':
                self.popt, self.pcov = curve_fit(self.lorentzian,x_zpl, y_zpl_base, [4000, 637.5,1.5], maxfev=max_fev )
                self.amp, self.center_wavelength, self.FWHM = self.popt; print(self.center_wavelength)
                self.peak_center.append(self.center_wavelength);
                self.width.append(self.FWHM);
                self.amplitude.append(self.amp);
                self.laser_pow.append(self.laser_power(f1))
                self.temps.append(self.temp(f1))
                
            elif func == 'two_lorentzian' :
                self.popt, self.pcov = curve_fit(self.two_lorentzian,x_zpl, y_zpl_base, [4000,5000, 636.5,637.5,1.5,1.5], maxfev=max_fev )
                self.amp, self.amp2, self.center_wavelength,self.center_wavelength2 ,self.FWHM, self.FWHM = self.popt
                self.peak_center.append(self.center_wavelength);
                self.peak_center2.append(self.center_wavelength2)
                ''' do just add self.amp2, self.center_wavelength2, self.FWHM2 '''
                self.width.append(self.FWHM);
                self.amplitude.append(self.amp);
                self.width2.append(self.FWHM2);
                self.amplitude2.append(self.amp2);
                self.laser_pow.append(self.laser_power(f1))
                self.temps.append(self.temp(f1))

            elif func == 'two_gaussian':
                self.popt, self.pcov = curve_fit(self.two_gaussian,x_zpl, y_zpl_base, [4000,5000, 636.5,637.5,1.5,1.5], maxfev=max_fev )
                self.amp, self.amp2, self.center_wavelength, self.center_wavelength2 ,self.FWHM, self.FWHM2 = self.popt
                self.peak_center.append(self.center_wavelength);
                self.peak_center2.append(self.center_wavelength2);
                self.width.append(self.FWHM);
                self.amplitude.append(self.amp);
                self.width.append(self.FWHM);
                self.amplitude.append(self.amp);
                self.width2.append(self.FWHM2);
                self.amplitude2.append(self.amp2);
                self.laser_pow.append(self.laser_power(f1))
                self.laser_pow.append(self.laser_power(f1))
                self.temps.append(self.temp(f1))

            else:
                if 'sproot' not in sys.modules:
                    from scipy.interpolate import splrep, sproot
                self.spline_fit(x_zpl, y_zpl_base)
                #self.temps.append(self.temp(f1))
                
    def create_dataframe(self, func = 'gaussian'):
        
        if func == 'gaussian' or func == 'lorentzian':
            self.dframe = pd.DataFrame(list(zip(self.fnm, self.time_step, self.temps,
                                       self.frame_num, self.laser_pow, self.amplitude, 
                                       self.peak_center, self.width, self.debye_waller, 
                                       self.kld, self.wasserstein_dist )))
            self.dframe.columns = ['filename',  'time', 'temperature', 'frame_num', 'laser_power', 'amplitude',
                      'peak_center', 'width', 'debye_waller', 'kl_divergence', 'wasserstein']
        elif func == 'spline':
            self.dframe = pd.DataFrame(list(zip(self.fnm, self.time_step, self.temps,
                                       self.frame_num, self.laser_pow, self.amplitude, 
                                       self.peak_center, self.width, self.debye_waller, 
                                       self.kld, self.wasserstein_dist )))
            self.dframe.columns = ['filename',  'time', 'temperature','frame_num', 'laser_power', 'amplitude',
                      'peak_center', 'width', 'debye_waller', 'kl_divergence', 'wasserstein']
        else:
            self.dframe = pd.DataFrame(list(zip(self.fnm, self.time_step, self.temps,
                                       self.frame_num, self.laser_pow, self.amplitude, 
                                       self.peak_center, self.width, self.debye_waller, 
                                       self.kld, self.wasserstein_dist, self.amplitude2, self.peak_center2, self.width2 )))
            self.dframe.columns = ['filename',  'time','temperature', 'frame_num', 'laser_power', 'amplitude',
                      'peak_center', 'width', 'debye_waller', 'kl_divergence', 'wasserstein',
                      'amplitude2', 'peak_center2', 'width2']
        
    def export_dataframe(self, export_name = 'name_me'):
        self.dframe.to_csv('c:/sams/saved_data/'+export_name)
        
    
    def export_dim_red_data(self):
        ''' export dimensionally reduced data set'''
        pass
        
        

                
                