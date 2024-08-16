# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:01:09 2023

@author: zahmed

this is a collection of common functions used for data exploration and modeling 
of the nv sensor data. The idea is to keep the fuction calls here and call them
in a jupyter notebook later on where a report can be generated and exported
in a user-friendly format

In a typical exploration mode we expect that one would want to import data 
and then generate:

1-  pair plots (seaborn)  - done
2-  heatmap (seaborn) - done
3-  kde plots (seaborn) -done
4-  2d correlation plots
5-  linear regression including polynomial regression (scikit-learn)
6-  L1 and L2 regularization of regression
7-  residual plots
8-  PCA an SVD regression
9-  kld and wasserstein plots - done
10- kernel PCA
11- random forest and decision tree
12- autoencoder regressor

# Example usage
## if only to get files
lister = SortList()
files = lister.get_files()
sorted_files = sorted(files, key=lister.temp)  


######### to do processing
# instantiate processor

sh =  processor()
# then filter files by taking out the first frame
shredder.filter_list()

# call main_processor with call of fitting routine needed, note: gaussian is default

sh.main_processor()

# create dataframe 
sh.create_dataframe()
#export dataframe
sh.export_dataframe()

##### then you can call indiviudual atr as needed to create plots

"""
import sys
sys.path.append('c:/sams/data_processing/')
#from file_reader_forter_parser import SortList
from classic_processor import processor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# import data
#lister = SortList()
#files = lister.get_files()
#sorted_files = sorted(files, key=lister.strp_atr)  

''' drop any files with Frame-0001 in their names'''

#filtered_files = [name for name in sorted_files if "Frame-00001" not in name]

''' classic processor: nv_zpl = [634.25,640.25] '''

shredder =  processor(nv_zpl = [632,642], k = ['Frame-00001', 'Frame-0001'], huang_rhys=[649, 700])
#file_list = shredder.get_files()
#sort_list = shredder.sorted_files()

shredder.filter_list()

print(shredder.filtered_files[0])

shredder.main_processor()

# create datafrme

shredder.create_dataframe()

#export dataframe
shredder.export_dataframe( export_name = 'sensor_2_week_1_third_cycle')#'sensor_2_week1_second_cycle' )

shredder.var_step_marker(var='temperature')

shredder.ramp_plot_check('temperature')

idx_ramp = shredder.ls[15]

# create data matrix for dim reduction
shredder.svd_data_matrix(avgs = 100,ramp_index=0, var = 'temperature' ,f_save= 'sensor_2_week1_third_cycle')#'sensor_2_week3_third_cycle_data_matrix_uptp40Conly')

##### examine SVD modes

shredder.svd_computation(f_save='sensor_2_week_1_third_cycle')


#make some plots
shredder.svd_pca_regression(ext_target='temperature', number_of_modes=10)

#PCA regression with polynomial fit

#shredder.pca_regression( n_comps = 5, poly_deg=1)

# perform noda's 2D correlation analysis
shredder.twodim_corr_spect_plot(f_saveas= 'sensor_2_week_1_third_cycle_2d_corr')#'sensor_2_week3_third_cycle_2dcorr_uptp40Conly')




#make pairplots
#sns.pairplot(shredder.dframe, hue='temperature')
#sns.pairplot(shredder.dframe, kind = 'kde', height = 10) # set height
#plt.show()
sns.pairplot(shredder.dframe, corner=True)
plt.savefig('pairplot_', dpi=700)
## kde plot overlaid scatter plot
#a = sns.pairplot(shredder.dframe, diag_kind='kde')
#a.map_lower(sns.kdeplot, levels= 3,color=".2")
plt.show()
# resize the 

#make heatmap
##### drop laser_power column and then plot 
sub_cols =  ['time', 'temperature', 
       'amplitude', 'peak_center', 'width', 'debye_waller', 'kl_divergence',
       'wasserstein']
df_ = shredder.dframe[sub_cols]
sns.heatmap(df_.corr(), annot=True)
plt.plot(figsize=(10, 8))
plt.title('Pearson correlation heatmap')
plt.savefig('pearson heatmap', dpi=700)
plt.show()

sns.heatmap(df_.corr(method='spearman'), annot=True)
plt.plot(figsize=(10, 8))
plt.title('Spearman correlation heatmap')
plt.savefig('Spearman correlation heatmap', dpi=700)
plt.show()

sns.regplot(x=shredder.dframe.temperature, y=shredder.dframe.kl_divergence, order=2)
plt.title('Temperature vs KL Divergence')
plt.savefig('temperature_vs_kld', dpi=700)
plt.show()

sns.regplot(x=shredder.dframe.temperature, y=shredder.dframe.wasserstein)
plt.title('Temperature vs Wasserstein')
plt.savefig('temperature_vs_kld', dpi =700)
plt.show()

sns.regplot(x=shredder.dframe.time, y=shredder.dframe.kl_divergence)
plt.title('Time vs KL Divergence')
#plt.savefig('regplot_time_vs_kld', dpi =700)
plt.show()

sns.regplot(x=shredder.dframe.time, y=shredder.dframe.wasserstein)
plt.title('Time vs Wasserstein')
#plt.savefig('regplot_time_vs_wasserstein', dpi =700)
plt.show()

plt.plot(shredder.dframe.kl_divergence, shredder.dframe.wasserstein)
plt.title('KL Divergence vs Wasserstein')
#plt.savefig('kld_vs_wasserstein', dpi =700)
plt.show()

plt.plot(shredder.dframe.time, shredder.dframe.wasserstein)
plt.plot(shredder.dframe.time, (shredder.dframe.temperature*200)+16000)
plt.title('Temperature and Wasserstein over time')
plt.savefig('temperature_and_time_vs_wasserstein', dpi =700)

plt.show()

plt.plot(shredder.dframe.time, shredder.dframe.kl_divergence)
plt.plot(shredder.dframe.time, (shredder.dframe.temperature/15000)+.001)
plt.title('Temperature and KL Divergence over time')
plt.savefig('temperature_and_time_vs_kld', dpi =700)
plt.show()

# =============================================================================
# make plots marked by ramp
# =============================================================================


plt.plot(shredder.dframe.time[:idx_ramp], shredder.dframe.laser_power[:idx_ramp], 'o')
plt.plot(shredder.dframe.time[idx_ramp:], shredder.dframe.laser_power[idx_ramp:])
plt.title('Time vs temperature')
plt.legend(['up', 'down'])
plt.show()



plt.plot(shredder.dframe.time[:idx_ramp], shredder.dframe.temperature[:idx_ramp], 'o')
plt.plot(shredder.dframe.time[idx_ramp:], shredder.dframe.temperature[idx_ramp:])
plt.title('Time vs temperature')
plt.legend(['up', 'down'])
plt.show()

plt.plot(shredder.dframe.time[:idx_ramp], shredder.dframe.kl_divergence[:idx_ramp])
plt.plot(shredder.dframe.time[idx_ramp:], shredder.dframe.kl_divergence[idx_ramp:])
plt.title('Time vs KL Divergence')
plt.legend(['up', 'down'])
plt.show()

plt.plot(shredder.dframe.temperature[:idx_ramp], shredder.dframe.kl_divergence[:idx_ramp])
plt.plot(shredder.dframe.temperature[idx_ramp:], shredder.dframe.kl_divergence[idx_ramp:])
plt.title('Temperature vs KL Divergence')
plt.legend(['up', 'down'])
plt.show()

plt.plot(shredder.dframe.temperature[:idx_ramp], shredder.dframe.wasserstein[:idx_ramp])
plt.plot(shredder.dframe.temperature[idx_ramp:], shredder.dframe.wasserstein[idx_ramp:])
plt.title('Time vs Wasserstein')
plt.legend(['up', 'down'])
plt.show()

plt.plot(shredder.dframe.temperature[:idx_ramp], shredder.dframe.debye_waller[:idx_ramp], 'o')
plt.plot(shredder.dframe.temperature[idx_ramp:], shredder.dframe.debye_waller[idx_ramp:], 'o')
plt.title('Temperature vs Debye-Waller Factor')
plt.legend(['up', 'down'])
plt.savefig('temperature_vs_dwf', dpi =700)
plt.show()


plt.plot(shredder.dframe.temperature[:idx_ramp], shredder.dframe.amplitude[:idx_ramp], 'o')
plt.plot(shredder.dframe.temperature[idx_ramp:], shredder.dframe.amplitude[idx_ramp:], 'o')
plt.title('Temperature vs Amplitude')
plt.legend(['up', 'down'])
plt.savefig('temperature_vs_ampliude', dpi =700)
plt.show()


plt.plot(shredder.dframe.temperature[:idx_ramp], shredder.dframe.width[:idx_ramp])
plt.plot(shredder.dframe.temperature[idx_ramp:], shredder.dframe.width[idx_ramp:])
plt.title('Temperature vs FWHM')
plt.legend(['up', 'down'])
plt.savefig('temperature_vs_width', dpi =700)

plt.show()

plt.plot(shredder.dframe.temperature[:idx_ramp], shredder.dframe.peak_center[:idx_ramp])
plt.plot(shredder.dframe.temperature[idx_ramp:], shredder.dframe.peak_center[idx_ramp:])
plt.ylim(636.5,638.5)
plt.title('Temperature vs Peak Center')
plt.legend(['up', 'down'])
plt.savefig('temperature_vs_peak_center', dpi =700)
plt.show()

plt.plot(shredder.dframe.time[:idx_ramp], shredder.dframe.debye_waller[:idx_ramp], 'o')
plt.plot(shredder.dframe.time[idx_ramp:], shredder.dframe.debye_waller[idx_ramp:], 'o')
plt.title('Time vs Debye-Waller Factor')
plt.legend(['up', 'down'])
plt.savefig('time_vs_dwf', dpi =700)
plt.show()

plt.plot(shredder.dframe.time[:idx_ramp], shredder.dframe.amplitude[:idx_ramp], 'o')
plt.plot(shredder.dframe.time[idx_ramp:], shredder.dframe.amplitude[idx_ramp:], 'o')
plt.title('Time vs amplitude')
plt.legend(['up', 'down'])
plt.savefig('time_vs_amplitude', dpi =700)
plt.show()


plt.plot(shredder.dframe.time[:idx_ramp], shredder.dframe.peak_center[:idx_ramp], 'o')
plt.plot(shredder.dframe.time[idx_ramp:], shredder.dframe.peak_center[idx_ramp:], 'o')
plt.title('Time vs peak center')
plt.legend(['up', 'down'])
plt.savefig('time_vs_peak_center', dpi =700)
plt.show()

plt.plot(shredder.dframe.time[:idx_ramp], shredder.dframe.width[:idx_ramp], 'o')
plt.plot(shredder.dframe.time[idx_ramp:], shredder.dframe.width[idx_ramp:], 'o')
plt.title('Time vs peak width')
plt.legend(['up', 'down'])
plt.savefig('time_vs_width', dpi =700)
plt.show()


# =============================================================================
# build a regression model using PCA module in sklearn
# =============================================================================

# specify the number of components to use
n_comps = 10
y = np.array(shredder.df_sm.index, dtype='float64')
X_train, X_test, y_train, y_test = train_test_split(shredder.df_sm, y, random_state=42)

# training the model
X_reduced = PCA(n_components=n_comps).fit_transform(X_train)
plt.plot(X_reduced[:,0], X_reduced[:,1]);plt.show;
plt.plot(X_reduced[:,0], X_reduced[:,2]);plt.show;
plt.plot(X_reduced[:,1], X_reduced[:,2]);plt.show;


#fit to a regression model
ln = LinearRegression()

pcr = ln.fit(X_reduced, y_train)

print(f"Training PCR r-squared {pcr.score(X_reduced, y_train):.3f}")
pred_pca = pcr.predict(X_reduced)
resid_pc = y_train - pred_pca
plt.plot(y_train, resid_pc, 'o'); 
plt.title('training prediction against ground truth (10 modes)');
plt.show()

# vaildation error
pred_pca_test =  pcr.predict(PCA(n_components=n_comps).fit_transform(X_test))
resid_pca_test = y_test - pred_pca_test
plt.plot(y_test, resid_pca_test, 'o'); 
plt.title('valiation prediction against ground truth (10 modes)');
plt.savefig('validation error', dpi=700)
plt.show()
print('mean of the resiudals is {}, while the standard deviation is {}'.format(np.mean(resid_pca_test), np.std(resid_pca_test)))


pipeline = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
pipeline.fit(X_reduced, y_train)
print(f"Training PCR r-squared {pipeline.score(X_reduced, y_train):.3f}")
pred_pca_train = pipeline.predict(X_reduced)
print(f"mean sqaure error for training is {mean_squared_error(y_train, pred_pca_train):.3f}")
resid_pc_train = y_train - pred_pca
plt.plot(y_train, resid_pc_train, 'o'); 
plt.title('training prediction against ground truth (10 modes)');
plt.savefig('quadratic pca training fit (10 modes)', dpi=700)
plt.show()

# vaildation error

pred_pca_test =  pipeline.predict(PCA(n_components=n_comps).fit_transform(X_test))
resid_pca_test = y_test - pred_pca_test
plt.plot(y_test, resid_pca_test, 'o'); 
plt.title('test prediction against ground truth');
plt.savefig('testing_error_quad_fit_10_modes', dpi=700)
plt.show()
print('mean of the resiudals is {}, while the standard deviation is {}'.format(np.mean(resid_pca_test), np.std(resid_pca_test)))
print(f"mean sqaure error for testing is {mean_squared_error(y_test, pred_pca_test):.3f}")
pass




