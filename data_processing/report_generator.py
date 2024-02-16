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

1-  pair plots (seaborn)
2-  heatmap (seaborn)
3-  kde plots (seaborn)
4-  2d correlation plots
5-  linear regression including polynomial regression (scikit-learn)
6-  L1 and L2 regularization of regression
7-  residual plots
8-  PCA an SVD regression
9-  kld and wasserstein plots
10- kernel PCA
11- random forest and decision tree
12- autoencoder regressor

# Example usage
lister = SortList()
files = lister.get_files()
sorted_files = sorted(files, key=lister.temp)  
# Replace `lister.temp` with your desired sorting function
print(sorted_files)

#pass the sorted/filetered list of files to the processor class

sh =  processor()
# then call routines of choice


"""
import sys
sys.path.append('c:/sams/data_processing/')
from file_reader_sorter_parser import SortList
from classic_processor import processor

# import data
#lister = SortList()
#files = lister.get_files()
#sorted_files = sorted(files, key=lister.strp_atr)  

''' drop any files with Frame-0001 in their names'''

#filtered_files = [name for name in sorted_files if "Frame-00001" not in name]

''' classic processor: ZPL [632, 642]  '''

shredder =  processor()

#file_list = shredder.get_files()
sort_list = shredder.sorted_files()

filt_list = shredder.filter_list('Frame-00001')












