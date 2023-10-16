# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:08:30 2023

@author: zahmed

class for designating file path and globbing up files
"""
#import modules
#from pathlib import Path
import glob

class Filelister:
    '''
    this class is designed to pull in data files as a list that can be fed
    into a pandas to read in and process the spectroscopy data

    the default path is c:/sams/saved_data and file type is set to csv

    by running filelist.getfiles() you can get a list of all files with
    txt or csv tags and feed it to pandas
    '''

    def __init__(self):
        self.path = input("Enter the path for files: ")
        if not self.path:
            self.path =  'c:/sams/saved_data'
        self.fn =  input('Enter filetype, default filetype is *.csv,')
        self.file_list = []
        if not self.fn:
            self.fn = '/*.csv'



    def get_files(self):
        self.f_path = self.path+self.fn
        self.file_list = glob.glob(self.f_path)
        print(self.f_path)
        return self.file_list
