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

1- pair plots (seaborn)
2- heatmap (seaborn)
3- kde plots (seaborn)
4- 2d correlation plots
5- linear regression including polynomial regression (scikit-learn)
6- L1 and L2 regularization of regression
7- residual plots
8



"""
import sys
sys.path.append('c:/sams/data_processing/')
from file_reader_forter_parser import SortList
from classic_processor import processor





