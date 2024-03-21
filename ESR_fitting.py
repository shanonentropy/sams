# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 19:27:26 2024

@author: zahmed
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import optimize
import matplotlib.pyplot as plt



path  = 'C:/nv_data/ESR_2024-02-24_04h02m34s.txt'
df = pd.read_csv(path, header = 0, sep= '\t')

plt.plot(df.iloc[:,0], df.iloc[:,2]/df.iloc[:,1])

plt.show()


def lorentzian(x, x0, a, gam ):
    return a * gam**2 / ( gam**2 + ( x - x0 )**2)


def two_lorentzian( x, x0, x01,a, gam):
    '''   '''
    return ( (a * gam**2 / ( gam**2 + ( x - x0 )**2)) + (a * gam**2 / ( gam**2 + ( x - x01 )**2))) 

x = df.iloc[25:,0]*1e-9
y = (df.iloc[25:,2]/df.iloc[25:,1])-1

popt2,pcov2 = curve_fit(two_lorentzian,x, y, p0=[ 2.865,2.88,.03,0.0001], maxfev=5000000 )   

popt,pcov = curve_fit(lorentzian,x, y, p0=[ 2.872,1.03,0.0001], maxfev=5000000 )   


plt.plot(x, lorentzian(x, *popt))
plt.plot(x, y, 'b*'); 
resid_1l = y - lorentzian(x, *popt)
plt.plot(x, resid_1l)
plt.legend(['exp curve', 'fitted curve', 'resid'])
plt.show()

resid = (y- two_lorentzian(x, *popt2))
plt.plot(x, y,'b')
plt.plot(x, two_lorentzian(x, *popt2),'r*');
plt.plot(x, resid, 'k')
plt.legend(['exp. curve', 'fitted curve', 'residual'])
plt.title(['Bi_Lorentzian'])




# =============================================================================
# given the native width and position (individual), observed spectral width and position
# (sigma_ensemble, mu_observed) compute the marginal distribution
# =============================================================================

