# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 15:42:25 2019
@author: Axel Griesmaier, LABS electronics
Copyright 2019 Axel Griesmaier
Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

@mod 09/2023: a shutdown fucntion was added by ZA; see note below
"""
import serial
#import six
import sys
#import os
from time import sleep
import glob
open_connections = {}
nconnected = 0
def available_serial_ports():
    portsavail = []
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        ports = glob.glob('/dev/tty[A-Za-z]*')
    else:
        raise EnvironmentError('platform not supported')
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            portsavail.append(port)
        except (OSError, serial.SerialException):
            pass
    return portsavail
def find_laser():
    ports = available_serial_ports()
    nfound = 0
    lasers = {}
    for port in ports:
        try:
            s = serial.Serial(port=port, baudrate=9600, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, timeout = 1)
            s.read()
            strn = 'HOWDY\n'
            s.write(strn.encode())
            answer = s.readline().strip().decode()
            s.read()
            if answer.startswith('Ready'):
                s.write(b'*IDN\n')
                answer = s.readline().strip().decode()
                s.read()
                nfound = nfound + 1
                if ('DLNSEC' in answer.upper()):
                    try:
                        model, serno = answer.split('_')
                    except:
                        model='unknown'
                        serno='00000'
                    lasers[nfound] = (port, serno, model)
                else:
                    model = 0
                    serno = 0
                    lasers[nfound] = (port, serno, model)
            s.close()
        except:
            pass
    print('')
    if (nfound>1):
        print("WARNING: More than one laser connected.")
    print('')
    return lasers
class DLnsec():
    def __init__(self, port = ''):
        self.port = port
        if ( port != '' ):
            self.open()
        else:
            self = connect()
        self.pre = None
        self.width = None
        self.laserison = None
        self.modeis = None
        self.powerset = None
        self.t_cycle = None
        self.freq = None
    def open(self):
        self.serial = serial.Serial(port=self.port,baudrate=9600,bytesize=serial.EIGHTBITS,parity=serial.PARITY_NONE,timeout=2)
        self.serial.write_timeout = 1
        self.serial.read_timeout = 1
    def close(self):
        self.serial.close()
        del self
    def write(self,cmd):
        self.serial.write(cmd + b'\n')
    def read(self, cmd):
        self.serial.write(cmd + b'\n')
        answer = self.serial.readline().strip().decode()
        self.serial.read()
        return answer
    def on(self):
        self.write(b'*ON')
        self.laserison = 1
    def off(self):
        self.write(b'*OFF')
        self.laserison = 0
    def set_power(self,pwr):
        strn = 'PWR' + '{:0d}'.format(pwr)
        self.write(strn.encode())
        self.get_power()
    def get_power(self):
        answer = self.read(b'PWR?')
        self.powerset = int(answer)
        return int(answer)
    def set_mode(self, mode):
        assert mode in ['LAS', 'INT', 'EXT', 'STOP']
        self.write(bytes(mode, encoding='utf-8')+b'')
        self.modeis = mode
    def set_width(self, width):
        assert type(width) == int
        assert width >=0
        assert width <=255
        self.width = width
        self.write(b'WID %i'%int(width))
        self.t_width = 1/16e6*self.pre*(width+1)
    def set_prescaler(self, pre):
        assert type(pre) == int
        assert pre in [1, 8, 64, 256, 1024]
        self.pre = pre
        self.write(b'PRE %i'%int(pre))
        self.freq = 16e6/256/pre
        self.t_cycle = 1 / self.freq
    def shutdown(self):
        ''' this function is written by ZA to turn down the laser at the end of the measurement
        the program brings the power down by half, waits 30 sec and then drops by another half. waits
        another 30 sec and then turn the laser off. Comm channel is left open'''
        lp = self.get_power()
        if lp > 10:
            self.set_power(lp//2)
            sleep(30)
            self.set_power(lp//4)
            sleep(30)
            self.set_mode('STOP')
            self.off()
            print('laser is off')
        else:
            self.set_mode('STOP')
            self.off()
            print('laser is off')
        

def connect(ser = ''):
    """open conneciton to laser or return handle if already open"""
    if ser == '':
        if len(open_connections) == 0: #no open connections
            all_lasers = find_laser() #find all available lasers
            if (len(all_lasers) == 0):
                raise RuntimeError('No lasers found.')
            next_laser = all_lasers.popitem() #next available laser
            port = next_laser[1][0]
            ser = next_laser[1][1]
            model = next_laser[1][2]
            laser = DLnsec(port)
            open_connections[ser] = laser
            return laser
        else: #there are open connections
            if (len(all_lasers) > 0): #there are lasers connected but not yet open
                next_laser = all_lasers.popitem() #next available laser
                port = next_laser[1][0]
                ser = next_laser[1][1]
                model = next_laser[1][2]
                laser = DLnsec(port)
                open_connections[ser] = laser
                return laser
            else:
                raise RuntimeError('No more un-connected lasers.')
    else:
        if ser in open_connections: # ser ist schon geöffnet
            return open_connections[ser]
        else: # ser ist noch nicht offen
            all_lasers = find_laser()
            if ser in all_lasers: #is ser a valid port?
                port = all_lasers[ser][1][0]
                laser = DLnsec(port)
                open_connections[ser] = laser
                return laser
            else:
                raise RuntimeError( 'Couldn\'t find DLnsec on port ' + ser )
def get_open_connections():
    return open_connections
if __name__ == '__main__':
    lasers = find_laser()
    print ('Connected lasers:')
    print (lasers)
