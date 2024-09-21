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
3-  kde plots (seaborn) 
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

# import data
#lister = SortList()
#files = lister.get_files()
#sorted_files = sorted(files, key=lister.strp_atr)  

''' drop any files with Frame-0001 in their names'''

#filtered_files = [name for name in sorted_files if "Frame-00001" not in name]

''' classic processor: nv_zpl = [634.25,640.25] '''

shredder =  processor(nv_zpl = [632,642])

#file_list = shredder.get_files()
#sort_list = shredder.sorted_files()

shredder.filter_list()

shredder.main_processor()

# create datafrme

shredder.create_dataframe()

#export dataframe
shredder.export_dataframe('sensor2_week5_LN2_zpl_632_642nm')


#make some plots

import matplotlib.pyplot as plt
import seaborn as sns


#make pairplots
#sns.pairplot(shredder.dframe, hue='temperature')
sns.pairplot(shredder.dframe, kind = 'kde', height = 10) # set height
#sns.pairplot(shredder.dframe, corner=True)
## kde plot overlaid scatter plot
a = sns.pairplot(shredder.dframe, diag_kind='kde')
a.map_lower(sns.kdeplot, levels= 3,color=".2")
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

sns.regplot(x=shredder.dframe.temperature, y=shredder.dframe.kl_divergence, order=2)
plt.title('Temperature vs KL Divergence')
plt.show()

sns.regplot(x=shredder.dframe.temperature, y=shredder.dframe.wasserstein)
plt.title('Temperature vs Wasserstein')
plt.show()

sns.regplot(x=shredder.dframe.frame_num, y=shredder.dframe.kl_divergence)
plt.title('Time vs KL Divergence')
plt.show()

sns.regplot(x=shredder.dframe.frame_num, y=shredder.dframe.wasserstein)
plt.title('Time vs Wasserstein')
plt.show()

plt.plot(shredder.dframe.kl_divergence, shredder.dframe.wasserstein)
plt.title('KL Divergence vs Wasserstein')
plt.show()

plt.plot(shredder.dframe.time, shredder.dframe.wasserstein)
plt.plot(shredder.dframe.time, (shredder.dframe.temperature*56)+16000)
plt.title('Temperature and Wasserstein over time')
plt.show()

plt.plot(shredder.dframe.time, shredder.dframe.kl_divergence)
plt.plot(shredder.dframe.time, (shredder.dframe.temperature/25000)+.001)
plt.title('Temperature and KL Divergence over time')
plt.show()

# =============================================================================
# define index marking end of first ramp
# =============================================================================
idx_ramp = 209979


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

plt.plot(shredder.dframe.temperature[:idx_ramp], shredder.dframe.debye_waller[:idx_ramp])
plt.plot(shredder.dframe.temperature[idx_ramp:], shredder.dframe.debye_waller[idx_ramp:])
plt.title('Time vs debye_waller')
plt.legend(['up', 'down'])
plt.show()

plt.plot(shredder.dframe.temperature[:idx_ramp], shredder.dframe.width[:idx_ramp])
plt.plot(shredder.dframe.temperature[idx_ramp:], shredder.dframe.width[idx_ramp:])
plt.title('Temperature vs FWHM')
plt.legend(['up', 'down'])
plt.show()

plt.plot(shredder.dframe.temperature[:idx_ramp], shredder.dframe.peak_center[:idx_ramp])
plt.plot(shredder.dframe.temperature[idx_ramp:], shredder.dframe.peak_center[idx_ramp:])
plt.title('Temperature vs Peak Center')
plt.legend(['up', 'down'])
plt.show()



plt.plot(shredder.dframe.frame_num, shredder.dframe.peak_center)
plt.plot(shredder.dframe.frame_num, shredder.dframe.peak_center[idx_ramp:])
plt.title('Peak Center over time')
#plt.legend(['up', 'down'])
plt.show()


plt.plot(shredder.dframe.frame_num, shredder.dframe.width)
plt.title('width over time')
#plt.legend(['up', 'down'])
plt.show()

plt.plot(shredder.dframe.frame_num, shredder.dframe.debye_waller)
plt.title('DWF over time')
#plt.legend(['up', 'down'])
plt.show()
