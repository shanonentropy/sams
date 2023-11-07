
import pyvisa 
import time
import numpy as np
from datetime import datetime, date
from pathlib import Path

import time
import csv
import sys



####### declare the instrument 

## Connect to the multimeter
rm = pyvisa.ResourceManager()
dmm = rm.open_resource("GPIB0::21::INSTR") # check to see if this is true

## Set the measurement

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
date =  datetime.now().strftime("%Y_%m_%d_%H_%M_%S") #str(date.today()).replace('-','')
fn = 'datalog_'+date+'_check_thermistor.txt'
file_open = folder / fn

# function for recording temperature to file_open

# num of hours
hr = 1
time_duration = 600 # update this number accordingly

# write a loop to measure over time frame covering the temperature experiment
# log machine time, monotnic time and temperatuer and/or resistance 


# collection func

index, elapsed_time, time_step, resistance = [],[],[],[]



def writer(time_duration, time_spacing=60):
    ''' this code'''

    print(" please note that the program as configured samples at {} Hz".format(1/time_spacing))
    to = time.monotonic() #time at the start of the measurement

    #temp_recorder(time_duration)
    for x in range(time_duration//time_spacing):
        t_e = np.round((time.monotonic()-to), 2)
        time_stp = time.monotonic()
        a = np.fromstring((dmm.query(":READ?")).replace('\n',','), sep=',').mean()
        #row = {'index':x,'elapsed_time':t_e,'time':time_stp, 'resistance':a}
        index.append(x),elapsed_time.append(t_e),time_step.append(time_stp), resistance.append(a)
        time.sleep(time_spacing) # wait 10 secon
        print(a)
writer(time_duration)


data = zip(index, elapsed_time, time_step, resistance)
with open(file_open, mode ='w', newline='') as f:
    fieldnames = ['index', 'elapsed_time', 'time_step', 'resistance']
    data_writer = csv.DictWriter(f, fieldnames=fieldnames)
    data_writer.writeheader()
    for r in data:
        data_writer.writerow({'index': r[0], 'elapsed_time':r[1], 'time_step':r[2], 'resistance':r[3]})
print(data)
dmm.close()
print('done')



