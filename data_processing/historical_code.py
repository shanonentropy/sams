# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:02:11 2023

@author: zahmed
"""


class sort_list(filelister):
    ''' regex based sorter'''
    def __init__(self, file_list):
        super().__init__()
        self.laser_power_id = re.compile('laser_power_\d*_')
        self.temperature = re.compile('-?\d*,\d*')
        self.acq_lenth = re.compile('\d*.\d*ms')
        self.grating_center = re.compile('CWL\d*.\d*nm')
        self.time_stamp = re.compile('2023 May\s\d* \d*\w\d*_\d*')
        self.f_num = re.compile('Frame-\d*')
        self.file_list = file_list



    def laser_power(self):
        lp = int(str(self.laser_power_id.findall(self.file_list)).split('_')[2])
        print(self.file_list)
        print(lp)
        return lp

    def temp(self, x):
        t = float((self.temperature.findall(x))[0].replace(',','.'))
        return t
    def acq(self, x):
        a = float(self.acq_lenth.findall(x)[0].replace('ms',''))
        return(a)
    def grating(self, x):
        float(self.grating_center.findall(x)[0].strip('CWL').strip('nm'))

    def time_st(self, x):
        g = self.time_stamp.findall(x)[0].split(' ')
        year, month, day, hr = g[0], g[1],g[2], g[3]
        h, m, s = hr.split('_')
        month = 5
        mt = time.mktime(datetime.datetime(int(year), int(month), int(day), int(h),int(m),int(s)).timetuple())
        return mt

    def frame_num(self, x):
        fnm = int(self.f_num.findall(x)[0].split('-')[1])
        return fnm

    def strp_atr(self, x):
        lp = int(str(self.laser_power_id.findall(x)).split('_')[2])
        t = float((self.temperature.findall(x))[0].replace(',','.'))
        a = float(self.acq_lenth.findall(x)[0].replace('ms',''))
        b = float(self.grating_center.findall(x)[0].strip('CWL').strip('nm'))
        g = self.time_stamp.findall(x)[0].split(' ')
        year, month, day, hr = g[0], g[1],g[2], g[3]
        h, m, s = hr.split('_')
        month = 5
        mt = time.mktime(datetime.datetime(int(year), int(month), int(day), int(h),int(m),int(s)).timetuple())
        fnm = int(self.f_num.findall(x)[0].split('-')[1])
        #append to lists
        return mt,lp,t,a,b,g,fnm

 # Example usage
lister = sort_list([])
files = lister.get_files()
sorted_files = sorted(files, key=lister.temp)  # Replace `lister.temp` with your desired sorting function
print(sorted_files)



############


# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:08:30 2023

@author: zahmed

class for designating file path and globbing up files
"""
#import modules
from pathlib import Path
import glob

def filepath():

    print ('default filetype is *.csv, to change to another type, enter as text')
    fn = input() or'*.csv'#placeholder
    print('provide filepath- txt not string')
    folder = Path(input())  or 'c:/sams/data'

    f_path = Path(folder/fn)
    return f_path


def datafiles(f_path):
    return glob.glob(str(f_path)) #not 'raw' ?submstracted
