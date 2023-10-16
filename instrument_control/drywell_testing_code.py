# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:13:53 2023

@author: zahmed

drywell testing under SAMS logic
"""

#import modules
from time import sleep
from waiting import wait
import numpy as np
from pathlib import Path
import os # Import os module
import pandas as pd


import sys # Import python sys module
sys.path.append('c:\\sams\instrument_control')
from drywell_interface import dry_well # drywell control module
