# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 12:37:21 2023

@author: zahmed


count and timeout needs to be improved

"""

import time
from time import sleep


########## wait module 
def wait_for_x(funk_cls, sleep_seconds = 30, timeout_seconds= 3000):
    ''' funk is the instance of a clas whose component func's binary output you wait on,
    sleep_seconds = refresh period between queries
    timeout_seconds= total time to wait before the function breaks out of the loop'''
    count = 0;
    print('setting the start counter at:{}'.format(count))
    #to= time.time()
    while count < timeout_seconds//sleep_seconds:
        if funk.read_stability_status()== 0:
            sleep(sleep_seconds)
            count +=1 
            print(count)
        elif funk.read_stability_status() ==1:
                print('stable'); #print(time.monotonic()-to)
                break
        else:
            sleep(sleep_seconds)
            count = count+1
            print('unstable output', count)
    else:
        print('timed out')

#### drywell specific wait module

def wait_for_drywell(drywell_cls=drywell, sleep_seconds = 30, timeout_seconds= 3000):
    ''' drywell_cls is an instantiation of the Dry_well class. drywell.read_stability_status()
    is the function whose binary output you wait on.
    sleep_seconds = refresh period between queries, default 30 sec
    timeout_seconds= total time to wait before the function breaks out of the loop; default 3000 s'''
    count = 1; 
    print('starting counter:', count); 
    to = time.monotonic() 
    while count < timeout_seconds//sleep_seconds:
        if drywell.read_stability_status()== 0:
            sleep(sleep_seconds)
            count+= 1
            print(count, drywell.read_temp(), drywell.read_stability_status())
        elif drywell.read_stability_status() ==1:
                print('stable'); print(time.monotonic()-to)
                break
        elif drywell.read_stability_status() !=1:
            sleep(sleep_seconds)
            print('went to bad place')
            count+=1
    else:
        print('timed out')
            

