# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:51:10 2023

@author: zahmed

this is simple script to log temperature readings from a 4-wire thermistor 
using pymeausre 
"""


import pyvisa 
import time, datetime
import numpy as np
from datetime import datetime, date
from pathlib import Path

import time
import csv
import sys



####### declare the instrument 
# Connect to the multimeter
rm = pyvisa.ResourceManager()
dmm = rm.open_resource("GPIB0::21::INSTR") # check to see if this is true
# Set the measurement

#rest the device
dmm.write("*RST")
#configure to measure resistnace/voltage
dmm.write("*RST")  #dmm.write(":CONF:VOLT:DC")
# set range to auto
dmm.write(":RES:RANG:AUTO:ON")
#set the integration time to 1 sec for resistance/ for voltage 10 cycles
dmm.write(":RES:APER 1") #dmm.write(":CONF:VOLT:DC")
# set source trig to immediate
dmm.write(":TRIG:SOUR IMM")
#set num of readings to 5
dmm.write(":SAMP:COUN 5")
# take the readings 
dmm.write(":SAMP:COUNT:AUTO ONCE")
dmm.write(":FORM:ELEM READ")
# put readings into a container
a = np.fromstring((dmm.query(":READ?")).replace('\n',','), sep=',').mean()
print('the mean resistance value is {}'.format(a))
              

# path to where file with given name is stored
folder = Path("c:/sams/saved_data")
<<<<<<< HEAD
date =  datetime.now().strftime("%Y_%m_%d_%H_%M_%S") 
=======
date =  datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") #str(date.today()).replace('-','')
>>>>>>> ad98bbdb63b94044be96e4be0caabc0b25d4f3da
fnt = 'datalog_'+date+'_check_thermistor.txt'
file_open_temp = folder / fnt



# open file in write module
# write a loop to measure over time frame covering the temperature experiment
# log machine time, monotnic time and temperatuer and/or resistance 


#elapsed_time_temp, time_step_temp, resistance = [],[],[]


def temp_writer(elapsed_time_temp =[], time_step_temp=[], resistance=[], time_spacing = 60):
    ''' this code'''

    print(" please note that the program as configured samples at 0.5 Hz")
    # main code: ask for user input on when to start data collection and for how long
    while True:
        recorder =  input("to start recording temperature data press S, to quit the loop press Q ")
        if recorder.lower() == 's':
            time_duration =  input("how many hours of data do you want? enter a number ")
            try:
                time_duration = int(time_duration)*3600 #convert hours into seconds
                print('lab data collection has began')
                to = time.monotonic()
<<<<<<< HEAD
                while (time.monotonic()-to) < time_duration:
=======
                while time.monotonic()-to < time_duration:
>>>>>>> ad98bbdb63b94044be96e4be0caabc0b25d4f3da
                    t_e = np.round((time.monotonic()-to), 2)
                    time_stp = time.monotonic()
                    a = np.fromstring((dmm.query(":READ?")).replace('\n',','), sep=',').mean()
                    elapsed_time_temp.append(t_e),time_step_temp.append(time_stp), resistance.append(a)
                    time.sleep(time_spacing) # wait 10 secon
<<<<<<< HEAD
                    print(time.monotonic()-to)
                break
            except ValueError:
                print('not an integer!')
                break
        elif recorder.lower() == 'q':
            print('exiting program')
            break
        else:
            print("please hit the s key")
            #break
    data = list(zip(elapsed_time_temp, time_step_temp, resistance))
    print(data)   
    with open(file_open_temp, mode ='w', newline='') as f:
        fieldnames = [ 'elapsed_time', 'time_step', 'resistance']
        data_writer = csv.DictWriter(f, fieldnames=fieldnames)
        data_writer.writeheader()
        for r in data:
            data_writer.writerow({'elapsed_time':r[0], 'time_step':r[1], 'resistance':r[2]})
        
temp_writer()


'''
def writer(file_open):
 
    with open(file_open, mode ='w', newline='') as data:
        fieldnames = ['index', 'elapsed_time', 'time', 'temp']
        writer = csv.DictWriter(data, fieldnames=fieldnames)
        writer.writeheader()
        print(" please note that the program as configured samples at 0.5 Hz")
        # main code: ask for user input on when to start data collection and for how long
        while True:
            recorder =  input("to start recording temperature data press S, to quit the loop press Q ")
            if recorder.lower() == 's':
                time_duration =  input("how many hours of data do you want? enter a number ")
                try:
                    time_duration = int(time_duration)*60 #convert hours into seconds
                    print('lab data collection has began')
                    to = time.monotonic()
                    for x in range(int(time_duration)):
                        print(x,time_duration, 'check1')
                        t_e = np.round((time.monotonic()-to), 2); print(x,time_duration, 'check2')
                        time_stp = time.monotonic(); print('2')
                        a = np.fromstring((dmm.query(":READ?")).replace('\n',','), sep=',').mean(); print(x,time_duration, type(float(a)),'check3')
                        print({'index':x,'elapsed_time':t_e,'time':time_stp, 'resistance':float(a)})
                        #writer.writerow({'index':x,'elapsed_time':t_e,'time':time_stp, 'resistance':'b'})
                        writer.writerow({'index':x,'elapsed_time':t_e,'time':time_stp, 'resistance':a}); ; print(x,time_duration, 'check4')
                        #data.flush()# forces python to write to disk rather than writing to file in memory
                        sleep(120) # wait 10 secon
                        print(x,time_duration, 'check5')
                except ValueError:
                    print('not an integer!')
                    break
            elif recorder.lower() == 'q':
                print('exiting program')
                break
            else:
                print("please hit the s key")
                #break


writer(file_open)
'''

=======
                break
            except ValueError:
                print('not an integer!')
                break
        elif recorder.lower() == 'q':
            print('exiting program')
            break
        else:
            print("please hit the s key")
            #break
            
    '''        
    build a calitbration module: in numpy instantiate a new array
where calibration is applied to the resistance array and the resulting 
data is stored as an array labelled temperature
'''      
    data = list(zip(elapsed_time_temp, time_step_temp, resistance))
       
    with open(file_open_temp, mode ='w', newline='') as f:
        fieldnames = [ 'elapsed_time', 'time', 'resistance']
        data_writer = csv.DictWriter(f, fieldnames=fieldnames)
        data_writer.writeheader()
        for r in data:
            data_writer.writerow({'elapsed_time':r[0], 'time':r[1], 'resistance':r[2]})
        
temp_writer()
>>>>>>> ad98bbdb63b94044be96e4be0caabc0b25d4f3da



# =============================================================================
# def temp_recorder(time_duration, time_spacing = 60):
#     ''' time_duration: in secs, the total length of the measurement
#     time_spacing  is the sleep time between readings'''
#     for x in range(time_duration//2):
#         t_e = np.round((time.monotonic()-to), 2)
#         time_stp = time.monotonic()
#         a = np.fromstring((dmm.query(":READ?")).replace('\n',','), sep=',').mean()
#         data_writer.writerow({'index':x,'elapsed_time':t_e,'time':time_stp, 'resistance':a})
#         data.flush()# forces python to write to disk rather than writing to file in memory
#         time.sleep(time_spacing) # wait 10 secon
# =============================================================================



'''
# Set the measurement mode to resistance
dmm.mode = Agilent34401A.Mode.resistance

# Set the number of readings to take
dmm.trigger_count = 10

# Take the readings and print them
readings = dmm.take_reading()
print(readings)
'''

