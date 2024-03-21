# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 08:56:41 2024

@author: zahmed

here we prepare the data matrix for use in SVD, PCA and 2Dcorr analysis 
packages

requirements: all the files have the same length and cover the same 
freq/wavelength range


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob, re
import time, datetime

# folder path
folder = 'C:/nv_data/06202023/LN2_bath/LN2_*60*.csv'
#read in file names
f=glob.glob(folder)
print(f)

time_stamp = re.compile('202\d \w*\s\d* \d*\w\d*_\d*')
def time_st(x):
    g = time_stamp.findall(x)[0].split(' ')
    year, month_str, day, hr = g[0], g[1],g[2], g[3]
    h, m, s = hr.split('_')
    month_cal ={'January':1, 'Feburary':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7, 'August':8, 'Sept':9, 'Oct':10, 'Nov':11, 'December':12}
    month = month_cal[month_str]
    mt = time.mktime(datetime.datetime(int(year), int(month), int(day), int(h),int(m),int(s)).timetuple())
    return mt

filenames = sorted(f, key = time_st)


# read in the first file in the time sereis. Store just the intensity columns. All other files get appended to this file
df_s = pd.read_csv( filenames[0], header = 0, encoding= 'unicode_escape', engine='python', sep=',',  usecols=['Wavelength'])

# read in files
read_line = np.arange(1, len(filenames), 10) 


for i in range(len(read_line[:])-1):
    #print(read_line[i], read_line[i+1])
    arr1 = read_line[i]
    arr2 = read_line[i+1]
    df_ = pd.read_csv( filenames[1], header = 0, encoding= 'unicode_escape', engine='python', sep=',', usecols=[4])
    for j in range(arr1, arr2, 1):
        #print(j)
        df_t = pd.read_csv(filenames[j], header = 0, encoding= 'unicode_escape', engine='python', usecols=[5])
        df_t.head(2)
        df_ = pd.concat([df_, df_t], axis = 1)
        #print(df_.head(2))
    df_avg = df_.iloc[:, 1:].mean(axis=1)
    df_s = pd.concat([df_s, df_avg], axis = 1)

plt.plot(df_s.iloc[:, 0],df_s.iloc[:, 1::10])   

df_s.to_csv('data_matrix_week1')
