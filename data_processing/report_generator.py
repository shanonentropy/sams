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
import matplotlib.pyplot as plt
import seaborn as sns

# import data
#lister = SortList()
#files = lister.get_files()
#sorted_files = sorted(files, key=lister.strp_atr)  

''' drop any files with Frame-0001 in their names'''

#filtered_files = [name for name in sorted_files if "Frame-00001" not in name]

''' classic processor: nv_zpl = [634.25,640.25] '''

shredder =  processor(nv_zpl = [632,642], k = ['Frame-00001', 'Frame-0001'], huang_rhys=[649, 720])
#file_list = shredder.get_files()
#sort_list = shredder.sorted_files()

shredder.filter_list()

print(shredder.filtered_files[0])

shredder.main_processor()

# create datafrme

shredder.create_dataframe()

#export dataframe
shredder.export_dataframe( 'sensor_2_week_1_first_cycle')#'sensor_2_week1_second_cycle' )

shredder.var_step_marker(var='temperature')

shredder.ramp_plot_check('temperature')



# create data matrix for dim reduction
shredder.svd_data_matrix(avgs = 100,ramp_index=0, var = 'temperature' ,f_save= 'sensor_2_week1_first_cycle')#'sensor_2_week1_second_cycle')#'sensor_2_week3_third_cycle_data_matrix_uptp40Conly')

##### examine SVD modes

shredder.svd_computation(f_save='sensor_2_week1_first_cycle')

# perform noda's 2D correlation analysis
shredder.twodim_corr_spect_plot(f_saveas= 'sensor_2_week1_first_cycle')#'sensor_2_week3_third_cycle_2dcorr_uptp40Conly')



#make some plots

shredder.svd_regression(ext_target='temperature')


#make pairplots
#sns.pairplot(shredder.dframe, hue='temperature')
#sns.pairplot(shredder.dframe, kind = 'kde', height = 10) # set height
#plt.show()
sns.pairplot(shredder.dframe, corner=True)
## kde plot overlaid scatter plot
#a = sns.pairplot(shredder.dframe, diag_kind='kde')
#a.map_lower(sns.kdeplot, levels= 3,color=".2")
plt.show()
# resize the 

#make heatmap
sns.heatmap(shredder.dframe.iloc[:,1:].corr(), annot=True)
plt.plot(figsize=(10, 8))
plt.title('Pearson correlation heatmap')
plt.show()

sns.heatmap(shredder.dframe.iloc[:,1:].corr(method='spearman'), annot=True)
plt.plot(figsize=(10, 8))
plt.title('Spearman correlation heatmap')
plt.show()
'''
sns.regplot(x=shredder.dframe.temperature, y=shredder.dframe.kl_divergence, order=2)
plt.title('Temperature vs KL Divergence')
plt.show()

sns.regplot(x=shredder.dframe.temperature, y=shredder.dframe.wasserstein)
plt.title('Temperature vs Wasserstein')
plt.show()

sns.regplot(x=shredder.dframe.time, y=shredder.dframe.kl_divergence)
plt.title('Time vs KL Divergence')
plt.show()

sns.regplot(x=shredder.dframe.time, y=shredder.dframe.wasserstein)
plt.title('Time vs Wasserstein')
plt.show()
'''
plt.plot(shredder.dframe.kl_divergence, shredder.dframe.wasserstein)
plt.title('KL Divergence vs Wasserstein')
plt.show()

plt.plot(shredder.dframe.time, shredder.dframe.wasserstein)
plt.plot(shredder.dframe.time, (shredder.dframe.temperature*200)+16000)
plt.title('Temperature and Wasserstein over time')
plt.show()

plt.plot(shredder.dframe.time, shredder.dframe.kl_divergence)
plt.plot(shredder.dframe.time, (shredder.dframe.temperature/15000)+.001)
plt.title('Temperature and KL Divergence over time')
plt.show()

# =============================================================================
# define index marking end of first ramp
# =============================================================================
idx_ramp = 26997

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
plt.show()

plt.plot(shredder.dframe.temperature[:idx_ramp], shredder.dframe.width[:idx_ramp])
plt.plot(shredder.dframe.temperature[idx_ramp:], shredder.dframe.width[idx_ramp:])
plt.title('Temperature vs FWHM')
plt.legend(['up', 'down'])
plt.show()

plt.plot(shredder.dframe.temperature[:idx_ramp], shredder.dframe.peak_center[:idx_ramp])
plt.plot(shredder.dframe.temperature[idx_ramp:], shredder.dframe.peak_center[idx_ramp:])
plt.ylim(636.5,638.5)
plt.title('Temperature vs Peak Center')
plt.legend(['up', 'down'])
plt.show()

plt.plot(shredder.dframe.time[:idx_ramp], shredder.dframe.debye_waller[:idx_ramp], 'o')
plt.plot(shredder.dframe.time[idx_ramp:], shredder.dframe.debye_waller[idx_ramp:], 'o')
plt.title('Time vs Debye-Waller Factor')
plt.legend(['up', 'down'])
plt.show()


