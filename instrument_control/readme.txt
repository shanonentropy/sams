this folder holds the hardware control scripts, while in the strictest sense these are high level wrappers 
in the context of this program, this constitutes the low level logic. These scripts will be called by scripts in 
the experimental control folder to define a meausrement acquistion protocol

Instruments covered:


* drywell (code by Kevin Williamson)
* check thermometer (to be done)
* laser (code by LABS, function added by ZA: needs to be tested)
* powermeter (thorlabs)
* camera+spectrometer (base codes by PI, custom call by Zeeshan; needs to be tested)
* ESR (base code based on qdspectro, refactored, modified and recast in OOP format by Zeeshan; needs to be doneIn addition:

temperature_generator (by ZA: generates temperature steps) 
spectroscopy class need to integrate sticthing option;
spectroscopy class needs to integrate spectrometer grating control
