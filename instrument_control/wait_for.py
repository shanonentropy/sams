# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 12:37:21 2023

@author: zahmed

this function was created to a) serve as a wait module that holds the system in
sleep mode until the drywell has reached equilibriation. b) since the drywell
occasionally outputs a non-sensical response i.e. non-binary output, a catch was
implemented that counts any such output as 0

The reason there are two funcs is that the drywell function was created first
and it resided inside the same file where the drywell was instantiated. It 
was later recognized that this func could be broadly utilized across the SAMS
infrastructure and hence wait_for_x was created to make it easier to remember 
that at heart it is just a wait module

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
        if funk_cls.read_stability_status()== 0:
            sleep(sleep_seconds)
            count +=1 
            print(count)
        elif funk_cls.read_stability_status() ==1:
                print('stable'); #print(time.monotonic()-to)
                break
        else:
            sleep(sleep_seconds)
            count = count+1
            print('unstable output', count)
    else:
        print('timed out')

#### drywell specific wait module

def wait_for_drywell(drywell_cls, sleep_seconds = 30, timeout_seconds= 3000):
    ''' drywell_cls is an instantiation of the Dry_well class. drywell.read_stability_status()
    is the function whose binary output you wait on.
    sleep_seconds = refresh period between queries, default 30 sec
    timeout_seconds= total time to wait before the function breaks out of the loop; default 3000 s'''
    count = 1; 
    print('starting counter:', count); 
    to = time.monotonic() 
    while count < timeout_seconds//sleep_seconds:
        if drywell_cls.read_stability_status()== 0:
            sleep(sleep_seconds)
            count+= 1
            print(count, drywell_cls.read_temp(), drywell_cls.read_stability_status())
        elif drywell_cls.read_stability_status() ==1:
                print('stable'); print(time.monotonic()-to)
                break
        elif drywell_cls.read_stability_status() !=1:
            sleep(sleep_seconds)
            print('went to bad place')
            count+=1
    else:
        print('timed out')
            

