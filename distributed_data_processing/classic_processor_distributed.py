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
from scipy import linalg
import dask
import dask.dataframe as dd
from dask.distributed import Client, progress
from dask import delayed, compute



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
    
    def __init__(self, nv_type = 'nv', nv_zpl = [634.25,640.25], nv0_zpl = [572.0, 578], huang_rhys = [649, 780], k = ['Frame-00001', 'Frame-0001'] ):
        super().__init__()
        self.nv_zpl = nv_zpl
        self.nv0_zpl = nv0_zpl
        self.huang_rhys = huang_rhys
        self.nv_type = nv_type
        lister = SortList()
        files = lister.get_files()
        self.filenames = sorted(files, key=lister.time_st)  
        self.k = k
        self.fsave_path = '../saved_data/'
        
        '''
        self.fnm, self.time_step = [],[]
        self.laser_pow, self.amplitude, self.peak_center =[],[],[]
        self.width, self.debye_waller, self.frame_num = [], [],[]
        self.kld, self.wasserstein_dist =[], []
        self.amplitude2, self.peak_center2, self.width2 =[],[],[]
        self.temps = []
        
        
        
        
    def sorted_files(self):
        #sort files by desired key
        self.sort_files = sorted(self.get_files(), key=self.strp_atr)
        return self.sort_files'''
    
    def filter_list(self):
        ''' drop any files with a particular key in them in their names'''
        if len(self.k) <2:
            self.filtered_files = [name for name in self.filenames if self.k not in name]
        else: 
            self.filtered_files = [name for name in self.filenames if all(k not in name for k in self.k)]
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
            
            need to write the loop here
            
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
        return kl_div
        
        

    
    def twodim_corr_spect_plot(self, f_saveas = 'dummy_give_me_name'):
        ''' func compute the sync and async 
        components of the spectra as laid out in Noda's paper '''
        freq_order = True  # if True sets the freq axis to run max to min 
        mean_center = True
        num_contour = 10
        
        df_ss = self.df_s.T
        df_ss.columns = df_ss.iloc[0, :]
        spec = df_ss
        #plot data for visual inspection
        spec.T.iloc[:, 1:].plot(legend=None); plt.title('Spectroscopy Curves')
        if freq_order: plt.xlim(max(spec.columns), min(spec.columns))
        plt.show()
        if mean_center:
            spec = spec - spec.mean()
        spec.T.iloc[:, 1:].plot(legend=None); plt.title('mean centered spectroscopy curves');plt.show()   
        plt.show()

        # create contour plots
        def contourplot(spec):
            x = spec.columns
            y = spec.index
            z = spec.values
            zmax = np.abs(z).max()
            plt.figure(figsize=(6, 6))
            contour = plt.contour(x, y, z, num_contour, cmap="bwr", vmin=-1 * zmax, vmax=zmax) #coolwarm or seismic
            plt.colorbar(contour)
            if freq_order:
                plt.xlim(max(x), min(x))
                plt.ylim(max(y), min(y))     
        # synchronous correlation
        sync = pd.DataFrame(spec.values.T @ spec.values / (len(spec) - 1))
        sync.index = spec.columns
        sync.columns = spec.columns
        sync = sync.T
        # plot sync data
        contourplot(sync)
        plt.savefig(f_saveas+'synch', dpi=700)
        #save data
        sync.to_csv(f_saveas+"_sync.csv") # note: fn[:-4] drops the ".csv"
        # Hilbert-Noda transformation matrix
        noda = np.zeros((len(spec), len(spec)))
        for i in range(len(spec)):
            for j in range(len(spec)):
                if i != j: noda[i, j] = 1 / np.pi / (j - i)
                
        # asynchronouse correlation
        asynch = pd.DataFrame(spec.values.T @ noda @ spec.values / (len(spec) - 1))
        asynch.index = spec.columns
        asynch.columns = spec.columns
        asynch = asynch.T
        #plot async data
        contourplot(asynch)
        plt.savefig(self.fsave_path+f_saveas+'asynch', dpi=700)
        #save data
        asynch.to_csv(f_saveas+"_asynch.csv") # note: fn[:-4] drops the ".csv"
    
    
    def var_step_marker(self, var =  'temperature'):
        ''' this function takes in the dataframe of processed data and 
        creates a list of indecies where variable of interest such as temperature
        change
        fucntion outputs ramp_indicator_index, this is the index in the list
        from var_step-marker func marking where the ramp ends
        used in svd data_matrix, check it using ramp_plot_check
        '''
        self.ls = self.dframe[self.dframe[var].diff().abs() > 1].index
        self.ls = self.ls.insert(0,0)
        self.ls = self.ls.append( pd.Index([self.dframe.index[-1]]) )
        print('step index list is:',self.ls)
        
    def ramp_plot_check(self, var='temperature'):
        indicator_ = len(self.ls)/2
        print(indicator_)
        self.ramp_indicator_index = int(np.floor(indicator_))
        for j in range(len(self.ls)-1):
            #print(i, i+1)
            plt.plot(self.dframe.time, self.dframe[var])
            plt.plot(self.dframe.time.iloc[self.ls[j]], self.dframe[var].iloc[self.ls[j]], 'o')
            plt.plot(self.dframe.time.iloc[self.ls[self.ramp_indicator_index]], self.dframe[var].iloc[self.ls[self.ramp_indicator_index]], 'rx')

            print("index marking the end of ramp is {}".format(self.ramp_indicator_index))
            plt.savefig('temp_time_history_plot_', dpi=700)
    # write a decorator to enable overide of self.ls
        # this will create the dataframe to put the data in during distributed
        # assembly of data matrix in scd_data matrix call
        
    
    
    def svd_data_matrix(self, f, ramp_index = 0 ,var =  'temperature'):
        '''this function, adapted for distributed computing, will assemble the all the filtered measurements into a data matrix to be used
        in svd and pca computations. func takes prepared dataframe as an input
        tacks on temperature as the header
        
        ramp_index = default is 0 which will set to ramp_indicator_index
        determined in ramp_plot_check. Otherwise the function will use user 
        provided number
        
        '''
        # create the dataframe to log each averaged out observation
        temps = self.temp(f); print(temps)
        df_ = pd.read_csv(f, sep=',', header=0, engine='python', usecols = ['Intensity'])
        df_.columns = [temps]
        return df_
    
    def create_svd_matrix(self, f_save = 'dummy'):
        delayed_results = [delayed(self.svd_data_matrix)(f, ramp_index=0, var = 'temperature') for f in self.filter_list()]

        # Compute the results
        results = compute(*delayed_results)

        self.df_s = pd.concat(results, axis=1)
        df_s_index = pd.read_csv( self.filtered_files[0], header = 0, engine='python', sep=',',  usecols=['Wavelength'])
        self.df_s = pd.concat([df_s_index,self.df_s], axis = 1)
        
        self.df_s.to_csv(f_save+'_svd_data_matrix_avg')
        
        
    def svd_computation(self, fullmatrix = False, demean = 'yes', f_save = 'dummy'):
        ''' this function will take in the data matrix from svd_data_matrix
        and then compute the svd. Plots show the first 3 U, s and VT modes
        
        fullmatrix default is False, it sets the np.linalg.svd() full-matrix kw
        to True if desired
        
        default is to compute via the de-centered data matrix which 
        is saved at the end.
        
              
        
        add wavelength info to X-axis
        take columns headers as a np.array??
        
        '''
                       
        if demean == 'yes':
            self.df_sm = self.df_s.T
            col_header = self.df_sm.iloc[0]
            self.df_sm.columns = col_header
            self.df_sm = self.df_sm[1:]
            self.df_sm = self.df_sm - self.df_sm.mean()
            
            self.U, self.s, self.Vt = np.linalg.svd(self.df_sm, full_matrices=True ) 
        
            print('eigenmode matrix is {}'.format(self.U.shape))
            print('singular value matrix is {}'.format(self.s.shape))
            print('loading matrix is {}'.format(self.Vt.T.shape))
            print('singular value matrix is {}'.format(self.s))
            # define the x-axis for plotting
            x_axis =  np.array(self.df_sm.columns)
            ''' make and save plots '''
            plt.plot( self.U[:,0], 'm')
            plt.title('Coefficients of mode #1')
            plt.savefig('Coeffcient_of_mode_1_', dpi=700)
            plt.show()
            plt.plot(self.U[:,1], 'orange')
            plt.title('Coeficient of mode #2')
            plt.savefig('Coeffcient_of_mode_2_', dpi=700)
            plt.show()
            plt.plot(self.U[:,2], 'k')
            plt.title('Coefficients of mode #3')
            plt.savefig('Coeffcient_of_mode_3_', dpi=700)
            plt.show()
            plt.plot(self.U[:,3], 'k')
            plt.title('Coefficients of mode #4')
            plt.savefig('Coeffcient_of_mode_4_', dpi=700)
            plt.show()
            plt.plot(self.U[:,4], 'k')
            plt.title('Coefficients of mode #5')
            plt.xlabel('Wavelength (nm)')
            plt.savefig('Coeffcient_of_mode_5_', dpi=700)
            plt.show()
            plt.plot(x_axis,self.Vt[0,], 'm')
            plt.title('Mode_1')
            plt.xlabel('Wavelength (nm)')
            plt.savefig('Mode_1_', dpi=700)
            plt.show()
            plt.plot(x_axis, self.Vt[1,],'orange')
            plt.title('Mode #2')
            plt.xlabel('Wavelength (nm)')
            plt.savefig('Mode_2_', dpi=700)
            plt.show()
            plt.plot(x_axis,self.Vt[2,],'k')
            plt.title('Mode #3')
            plt.xlabel('Wavelength (nm)')
            plt.savefig('Mode_3_', dpi=700)
            plt.show()
            plt.plot(x_axis, self.Vt[3,],'k')
            plt.title('Mode #4')
            plt.xlabel('Wavelength (nm)')
            plt.savefig('Mode_4_', dpi=700)
            plt.show()
            plt.plot(self.Vt[4,],'k')
            plt.title('Mode #5')
            plt.xlabel('Wavelength (nm)')
            plt.savefig('Mode_5_', dpi=700)
            plt.show()
            plt.plot(self.s, 'o')
            plt.savefig('skee_plot', dpi=700)
            plt.show()
            self.df_sm.to_csv(f_save+'demean_pca_data_matrix')
            
        else:
            '''note: it is anticipated that this option will be used sparingly
            hence why the U, Vh, S are retained as local vairables and output
            not saved
            
            
            '''
            # perform SVD
            U, s, Vt = np.linalg.svd(self.df_s, full_matrices=False ) 
        
            print('eigenmode matrix is {}'.format(U.shape))
            print('singular value matrix is {}'.format(s.shape))
            print('loading matrix is {}'.format(Vt.T.shape))
            print('singular value matrix is {}'.format(s))
            
            plt.plot(U[:,0], 'm')
            plt.title('Mode #1')
            plt.show()
            plt.plot(U[:,1], 'orange')
            plt.title('Mode #2')
            plt.show()
            plt.plot(U[:,2], 'k')
            plt.title('Mode #3')
            plt.show()
            plt.plot(Vt[0,], 'm')
            plt.title('Coeffcient of mode #1')
            plt.show()
            plt.plot(Vt[1,],'orange')
            plt.title('Coeffcient of mode #2')
            plt.show()
            plt.plot(Vt[2,],'k')
            plt.title('Coeffcient of mode #3')
            plt.show()
            plt.plot(s)
            plt.plot('singular values')
            plt.show()
            
    
        
   
    def svd_pca_regression(self, indexed_target = 'yes', number_of_modes = 3  ,ext_target='temperature'):
        '''this function takes in the entire dataset 
        for svd compuation and regresses it against a target
        
        '''
        if indexed_target =='yes':
            target_var= np.array(self.df_sm.index, dtype='float64')
            #x_tilda = np.linalg.pinv(self.df_sm.iloc[:]) @ np.array(self.df_sm.index).reshape(-1,1)
            Vt_k = self.Vt[:number_of_modes, :] 
            X_pca = self.df_sm @ Vt_k.T
            X_pca_bias = np.c_[np.ones(X_pca.shape[0]), X_pca]
            beta = np.linalg.inv(X_pca_bias.T @ X_pca_bias) @ X_pca_bias.T @ target_var
            pred =  X_pca_bias@beta
            plt.plot(target_var, pred)
            plt.title('training data plot of pred against truth')
            plt.savefig('svd_pca_regression_', dpi=700)
            plt.xlabel('Truth')
            plt.ylabel('predicted')
            #add computation of residulal 
            
        else:
            x_tilda = np.linalg.pinv(self.df_sm.iloc[:]) @ np.array(self.dframe[ext_target][::self.avgs])
            p=self.df_sm.iloc[:, :]@x_tilda; 
            p=(np.array(p))
            plt.plot( p)
            plt.title('prdicted outcome'); plt.show()
            pred = self.df_s@x_tilda
            plt.plot(self.dframe[ext_target], pred)
            plt.title('training data plot of pred again truth')
            plt.xlabel('Truth')
            plt.ylabel('predicted')
        
        
    def pca_regression(self, num_comps = 5, poly_deg = 1):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.metrics import mean_squared_error
        '''add sklearn's PCA regression routine chained to polynommial
        and/or GP regression
        NEEDS to be fixed, doesn't reco'''
        
        #### ADD A COLUMN OF ONES TO THE MATRIX
        ones_col = np.ones((self.df_sm.shape[0], 1))
        self.df_sm_ones = np.hstack((self.df_sm, ones_col))
        
        y = np.array(self.df_sm.index, dtype='float64')
        X_train, X_test, y_train, y_test = train_test_split(self.df_sm_ones, y, random_state=42)

        # training the model
        X_reduced = PCA(n_components=num_comps).fit_transform(X_train)
        plt.plot(X_reduced[:,0], X_reduced[:,1]);plt.show;
        plt.plot(X_reduced[:,0], X_reduced[:,2]);plt.show;
        plt.plot(X_reduced[:,1], X_reduced[:,2]);plt.show;
        
        
        #fit to a regression model
        pipeline = make_pipeline(PolynomialFeatures(degree = poly_deg), LinearRegression())
        pipeline.fit(X_reduced, y_train)
        print(f"Training PCR r-squared {pipeline.score(X_reduced, y_train):.3f}")
        pred_pca = pipeline.predict(X_reduced)
        print(f"mean sqaure error for training is {mean_squared_error(y_train, pred_pca):.3f}")
        resid_pc = y_train - pred_pca
        plt.plot(y_train, resid_pc, 'o'); 
        plt.title('training prediction against ground truth');
        plt.savefig('training error', dpi=700)

        plt.show()

        # vaildation error
        pred_pca_test =  pipeline.predict(PCA(n_components=num_comps).fit_transform(X_test))
        resid_pca_test = y_test - pred_pca_test
        plt.plot(y_test, resid_pca_test, 'o'); 
        plt.title('test prediction against ground truth');
        plt.savefig('testing error', dpi=700)
        plt.show()
        print('mean of the resiudals is {}, while the standard deviation is {}'.format(np.mean(resid_pca_test), np.std(resid_pca_test)))
        print(f"mean sqaure error for testing is {mean_squared_error(y_test, pred_pca_test):.3f}")
        pass

    def variance_of_means(self):
        ''' interprets spectral diffusivity as variance of means to 
        estimate strain/temp dependence'''
        
        pass
    
    
    def gaussian(self, x_zpl, amp, u, std):
        ''' gaussian fit'''
        return amp*np.exp(-((x_zpl-u)**2/(2*std**2)))
    
    def two_gaussian(self,x_zpl, amp1, amp2, u1, u2, std1, std2):
        '''fits two gaussians to the curve with each component being independent'''
        return ((amp1*np.exp(-((x_zpl-u1)**2/(2*std1**2))) + (amp2*np.exp(-((x_zpl-u2)**2/(2*std2**2))) )))
    
    def lorentzian(self, x_zpl, x0, a, gam ):
        '''fits a Lorentzian to the curve   '''
        return a * gam**2 / ( gam**2 + ( x_zpl - x0 )**2)
      
    def two_lorentzian(self, x_zpl, x0, x01,a,a2, gam, gam2 ):
        '''fits two Lorentzians to the curve with each component being independent'''  
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
            self.result.update({'peak_center': self.center_wavelength, 'width': self.FWHM, 'laser_pow': self.laser_power(f1), 'temps': float(self.temp(f1)) })


        else:
            result.update({'peak_center': 'NaN', 'width': 'NaN' })
            print('pass');pass
            
            
        
    def main_processor(self, f1, nv_type='nv', func = 'gaussian', fit_params = [4000, 637.5,1.5], max_fev=50000, dx = 0.01 ):
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
        
        print(f1)
        result= {'frame_num': self.file_num(f1)} 
        
        
        ###### open and clean data ####
        df=pd.read_csv(f1, sep=',', header = 0, engine='python')
        df.sort_values(by='Wavelength', ascending=True)
        df.drop_duplicates(subset='Wavelength', keep='first', inplace=True)
        x,y=df['Wavelength'],df['Intensity']
        
        ### mark out zpl range of interest #####
        x_zpl, y_zpl = x[(np.abs(x-zp[0])).argmin():(np.abs(x-zp[1])).argmin() ],\
        y[(np.abs(x-zp[0])).argmin():(np.abs(x-zp[1])).argmin() ]
        
        ##### remove baseline #########
        base = peakutils.baseline(y_zpl, 1)
        y_zpl_base = y_zpl-base
        #plt.figure(figsize=(10,6))
        #plt.plot(x_zpl, y_zpl_base)
        #plt.title("ZPL data with baseline removed")
        #result.update({'kl_divergence': self.kl_divergence(y)})
        #### compute stats ###

        dx_val = (x[0]-x[50])/50
        area_zpl = trapz(y[(np.abs(x-zp[0])).argmin():(np.abs(x-zp[1])).argmin() ], dx= dx_val)
        area_psb = trapz(y[(np.abs(x-self.huang_rhys[0])).argmin():(np.abs(x-self.huang_rhys[1])).argmin() ], dx= dx_val)
        dw = area_zpl/area_psb
        result.update({'time': self.time_st(f1), 'wasserstein': wasserstein_distance(y, self.spectrum1), 'debye_waller': dw })
        ### fit the ZPL ####
        
        if func == 'gaussian': 
             popt, pcov = curve_fit(self.gaussian,x_zpl, y_zpl_base, [4000, 637.5,1.5], maxfev=max_fev )
             amp, center_wavelength, FWHM = popt
             result.update({'peak_center': center_wavelength, 'width': FWHM, 'amplitude':amp, 
                            'laser_pow': self.laser_power(f1), 'temperature': float(self.temp(f1)) })

             
        elif func == 'lorentzian':
            self.popt, self.pcov = curve_fit(self.lorentzian,x_zpl, y_zpl_base, [4000, 637.5,1.5], maxfev=max_fev )
            self.amp, self.center_wavelength, self.FWHM = self.popt; print(self.center_wavelength)
            result.update({'peak_center': self.center_wavelength, 'width': self.FWHM, 'amplitude':self.amp, 
                           'laser_pow': self.laser_power(f1), 'temperature': float(self.temp(f1)) })
         
        elif func == 'two_lorentzian' :
            self.popt, self.pcov = curve_fit(self.two_lorentzian,x_zpl, y_zpl_base, [4000,5000, 636.5,637.5,1.5,1.5], maxfev=max_fev )
            self.amp, self.amp2, self.center_wavelength,self.center_wavelength2 ,self.FWHM, self.FWHM = self.popt
            result.update({'peak_center': self.center_wavelength,'peak_center2': self.center_wavelength2, 
                           'width': self.FWHM, 'width2': self.FWHM2,
                           'amplitude':self.amp, 'amplitude2':self.amp2, 
                           'laser_pow': self.laser_power(f1), 'temperature': float(self.temp(f1)) })

        elif func == 'two_gaussian':
            self.popt, self.pcov = curve_fit(self.two_gaussian,x_zpl, y_zpl_base, [4000,5000, 636.5,637.5,1.5,1.5], maxfev=max_fev )
            self.amp, self.amp2, self.center_wavelength, self.center_wavelength2 ,self.FWHM, self.FWHM2 = self.popt
            result.update({'peak_center': self.center_wavelength,'peak_center2': self.center_wavelength2, 
                           'width': self.FWHM, 'width2': self.FWHM2,
                           'amplitude':self.amp, 'amplitude2':self.amp2, 
                           'laser_pow': self.laser_power(f1), 'temperature': float(self.temp(f1)) })

        else:
            if 'sproot' not in sys.modules:
                from scipy.interpolate import splrep, sproot
                self.spline_fit(x_zpl, y_zpl_base)
                #### add rest of routine note: so gaussian/lorentzian seem to be enough and sproot had more issues
                ####  and is less physics oriented solution
            #self.temps.append(self.temp(f1))
            
        return result
       
      
    def create_dataframe(self, results_, func = 'gaussian'):
       
        self.dframe = pd.DataFrame(results_) 

        
    def export_dataframe(self, export_name = 'name_me'):
        self.dframe.to_csv('../saved_data/'+export_name)
        
    
    def export_dim_red_data(self):
        ''' export dimensionally reduced data set'''
        pass
    
    def plotter(self, f1):
        ''' plot the curve to visually see what the spectra look like
        function will be based to the dask loop in report generator
        '''
        
        print(f1)
        ###### open and clean data ####
        df=dd.read_csv(f1, sep=',', header = 0, engine='python')
        #df_comp = df.compute()
        plt.plot(df.Wavelength, df.Intensity)
            
        
        

                
                