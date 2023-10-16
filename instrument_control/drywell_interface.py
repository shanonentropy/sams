# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:14:00 2020
@author:    Kevin Reed Williamson
            kevin.williamson@flukecal.com
            Fluke Calibration
Code to initiate communication and control Dry Wells 917X from Fluke 
Calibration through serial communication. 
This software is not supported by Fluke. Users are welcome to clone this 
repository for their own use.
This program has only been tested with Windows 10 and a Fluke 9173 dry well. 
Users with a variety of connected COM ports and/or a different operating 
system, should adjust the __init__ function for their circumstances. 
------------------------------------------------------------------------------
Copyright 2020 Fluke
Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
------------------------------------------------------------------------------
Dependencies:
- python            3.7.7
- pyserial          3.4
- vs2015_runtime    14.16.27012      
- dpython-dateutil  2.8.1
"""

import serial
import serial.tools.list_ports as port_list
from time import sleep
from datetime import datetime


class Dry_well(object):
    """ Class to control Fluke Calibration's Dry Wells 917X series dry wells """


    def __init__(self):
        """ Sets up connection to Fluke device
        Searches through available COM connections and chooses 5310
        """
        # Listing all available COM ports on windows computer
        ports = list(port_list.comports())

        for p in ports:
            print (p)

            # Choosing COM port from list of available connections 
            if "USB-to-Serial Comm Port (COM4)" in p[1]:
                try:
                    self.port = p[0]
                    # Setting up and connecting to device
                    self.ser = serial.Serial(port =     self.port,
                                             baudrate = 9600, #update after consulting NIMAX
                                             parity =   serial.PARITY_NONE,
                                             stopbits = serial.STOPBITS_ONE,
                                             bytesize = serial.EIGHTBITS,
                                             timeout =  0,
                                             write_timeout = 0)
                    if self.ser.is_open:
                        print("\n" + self.port + " has been opened.\n")
                        self.ser.write(b'*IDN? \r\n')
                        sleep(0.1)
                        print(bytes.decode(self.ser.read(256)))
                        self.ser.write(b'SYST:BEEP:IMM \r\n')
                    else:
                        print("\nDid not connect to " + self.port + "\n")
                    return
                except:
                    print("Failed to connect to " + p[0])


    def write_command(self,command):
        """Takes in string type AT command and returns string type responce"""
        self.ser.write(str.encode(command) + b'\r\n')
        sleep(0.1)
        response = (self.ser.readline()).decode('utf-8').split('\r')[0]
        if not response:
            sleep(0.1)
            response = (self.ser.readline()).decode('utf-8').split('\r')[0]
            self.ser.readline
        sleep(0.1)
        self.ser.readline()
        return(response)
    
    def enter_passcode(self,passcode=1234):
        self.write_command("SYST:PASS:CEN "+str(passcode))

    def beep(self):
        """ Makes a single beep from the dry well """
        self.write_command("SYST:BEEP:IMM ")
        return 


    def close(self):
        """ Closes serial connection with dry well """
        self.ser.close()
        sleep(0.1)
        if not self.ser.is_open:
            print("\n" + self.port + " has been closed.\n")
        return True


    def read_temp(self):
        """ Queries the control temperature returns float in C/F """
        try:
            temp = float(self.write_command("SOUR:SENS:DAT? TEMP "))
        except:
            try:
                temp = float(self.write_command("SOUR:SENS:DAT? TEMP "))
            except:
                print("There is a COM error!")
                temp = 0
        return(temp)
 

    def read_resistance(self):
        """ Queries control resistance returns float in ohms """
        res = float(self.write_command("SOUR:SENS:DAT? RES "))
        return(res)
       
        
    def read_set_temp(self):
        """ Queries temperature set point from dry well and returns a 
            float value of C/F """
        try:
            set_point = float(self.write_command("SOUR:SPO? "))
            return(set_point)
        except:
            sleep(0.1)
            self.ser.read(256)
            sleep(0.1)
            set_point = float(self.write_command("SOUR:SPO? "))
            return(set_point)


    def set_temp(self, set_point):
        """ Writes new temperature set point for the dry well 
            Takes a single float type variable """
        print(self.write_command("SOUR:SPO " + str(set_point) + " "))
        if float(set_point) == self.read_set_temp():
            print("Updated set point to: " + str(set_point) + "\xb0C")
            return True
        else:
            print("Failed to update set point!")
            print("Message delay or COM error. Line 123")
            return False


    def read_unit(self):
        """ Queries temperature unit of the dry well and returns a 
            float type value in Celsius """
        unit = self.write_command("UNIT:TEMP? ")
        return(unit)


    def set_unit(self,unit):
        """ Sets the temperature unit dry well 
            Takes 2 different input values:
                C   Celcius
                F   Fahrenheit """
        self.write_command("UNIT:TEMP " + unit + " ")
        if unit == self.read_unit():
            print("Updated units to: " + "\xb0" + unit)
            return True
        else:
            print("Failed to update set point!")
            return False


    def measure_rate(self):
        """ Measures the rate of temperature change in degrees per minute
            Returns a float value """
        try:
            response = float(self.write_command("SOUR:RATE? "))
        except:
            try:
                response = float(self.write_command("SOUR:RATE? "))
            except: 
                print("COM error!")
                response = 0
        return(response)


    def read_rate(self):
        """ Reads the control set-point rate
            Returns float type value of C/F per minute """
        try:
            response = float(self.write_command("SOUR:RATE? "))
        except:
            sleep(0.1)
            response = float(self.write_command("SOUR:RATE? "))
        return(response)


    def set_rate(self, rate):
        """ Sets the control set-point rate
            Takes float type value of C/F per minute """
        print(self.write_command("SOUR:RATE " + str(rate) + " "))
        if float(rate) == self.read_rate():
            print("Updated rate to: " + str(rate) + " K/min")
            return True
        else:
            print("Failed to update rate!")
            print("Message delay or COM error line line 182")
            return False


    def read_cutout(self):
        """ Queries the soft temperature limit of the dry well and returns
            a float type value """
        limit = float(self.write_command("SOUR:PROT:SCUT:LEV? "))
        return(limit)


    def set_cutout(self, limit):
        """ Sets the maximum temperature at which the output remains on """
        self.write_command("SOUR:PROT:SCUT:LEV " + str(limit) + " ")
        sleep(0.1)
        if float(limit) == self.read_cutout():
            print("Updated Temperature High limit to: " + str(limit))
            return True
        else:
            print("Failed to set Temperature High limit!")
            return False


    def reset_cutout(self):
        """ If the Metrology Well exceeds the temperature set in the soft 
            cutout menu or if it exceeds the maximum operating temperature of 
            the instrument, a cutout condition occurs. If this happens, the 
            unit enters cutout mode and will not actively heat or cool until 
            the user issues this command to clear the cutout. """
        if int(self.write_command("SOUR:PROT:TRIP? ")):
            self.write_command("SOUR:PROT:CLE ")
            if not int(self.write_command("SOUR:PROT:TRIP? ")):
                print("Reset temperature cut-out. Continue operations.")
                return True
        else:
            print(self.write_command("SOUR:PROT:TRIP? "))
            return False
        return


    def read_limit(self):
        """ Queries the soft temperature cut-out of the dry well and returns:
                1 for on
                0 for off """
        limit = float(self.write_command("SOUR:PROT:SOFT? "))
        return(limit)


    def read_output(self):
        """ Queries the dry well if the output is turned on. Returns values:
                0 for off 
                1 for on """
        response = int(self.write_command("OUTP:STAT? "))
        return(response)


    def set_output(self, value):
        """ Sets the output of the dry well heating element on or off.
            Receives the following values:
                True    or  1   for on
                False   or  0   for off """
        if value:
            self.write_command("OUTP:STAT 1 ")
            sleep(0.1)
            if self.read_output() == 1:
                print("Heating element turned on!")
        else:
            self.write_command("OUTP:STAT 0 ")
            sleep(0.1)
            if self.read_output() == 0:
                print("Heating element turned off!")


    def read_stability_limit(self):
        """ Queries the dry well stability limit
            Returns a float type value in C/F degrees """
        limit = float(self.write_command("SOUR:STAB:LIM? "))
        return(limit)


    def read_stability_status(self):
        """ Queries the dry well stability status indicating if the dry well 
            has reached temperature stability at the current set point. 
            Returns: 
                1 for 'stable'
                0 for 'not yet stable' """
        try:
            status = int(self.write_command("SOUR:STAB:TEST? "))
            return(status)
        except:
            try:
                status = int(self.write_command("SOUR:STAB:TEST? "))
                return(status)
            except:
                print("Error in checking stability.")
                return(0)


    def read_stability(self):
        """ Queries the dry well stability """
        try:
            stability = float(self.write_command("SOUR:STAB:DAT? "))
        except:
            try: 
                stability = float(self.write_command("SOUR:STAB:DAT? "))
            except:
                stability = 0.5
        return(stability)

    def create_data(self):
        """ Creates an active graph that may be continuously updated 
            Plots the following information with time as the x-axis """
        try:
            self.t = []
            self.t_string = []
            self.target = []
            self.temperature = []
            self.ramp = []
            self.stability = []
            return True
        except:
            print("Couldn't generate data attributes.")
            return False


    def update_data(self):
        """ Updates the active data atributes of the dry well object """
        # Checks to make sure data is consistent
        if len(self.t) != len(self.stability):
            self.t = self.t[:len(self.stability) - 2]
            self.t_string = self.t_string[:len(self.stability) - 2]
            self.target = self.target[:len(self.stability) - 2]
            self.temperature = self.temperature[:len(self.stability) - 2]
            self.ramp = self.ramp[:len(self.stability) - 2]
            self.stability = self.stability[:len(self.stability) - 2]
            print("Trimmed end of data ")
        try:
            self.t.append(datetime.now())
            self.t_string.append(str(self.t[-1]))
            self.target.append(self.read_set_temp())
            # Occasional value of 1000°C returned. removing these data points
            temp = self.read_temp()
            if temp < 740:
                self.temperature.append(temp)
            else: # Returning the last measured temperature instead
                self.temperature.append(self.temperature[-1])
            self.ramp.append(self.measure_rate())
            self.stability.append(self.read_stability())
            return True
        except:
            print('Data arrays have not been generated yet.\n' + 
                  'Please use the create_data() function!')
            return False


    def save_data(self):
        """ Saves data as CSV file with headers.
            Data is saved in 5 columns:
                Time
                Set Point
                Temperature
                Ramp Speed
                Stability """
        import csv
        try:
            with open(str(self.t[0])[0:10] + '_dry_well_log.txt', mode='w') as log:
                log_writer = csv.writer(log, delimiter=',')
                log_writer.writerow(['Time'] + 
                                    ['Set Point'] + 
                                    ['Temperature'] +
                                    ['Ramp Speed'] + 
                                    ['Stability'])
                for i in range(len(self.stability)):
                    log_writer.writerow([self.t_string[i]] +
                                        [str(self.target[i])] +
                                        [str(self.temperature[i])] +
                                        [str(self.ramp[i])] +
                                        [str(self.stability[i])] )
            return True
        except:
            print("Could not save data.")
            return False
