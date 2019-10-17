#To Do Next
#Continue on "Initialize Devices" section.


# ---------------------------------------------------------------------------- #
# Parameters
# ---------------------------------------------------------------------------- #
#Laser Sweep
lambda_start    = 1550
lambda_stop     = 1630
duration        = 5
trigger_step    = 0.01
power_dBm       = 7
#Data Collection
sample_rate     = 20e12
buffer          = 5 #Additional time around duration to prevent timeout.

#Save Data
#The first argument passed will be used as the file name.
filename_prefix = ""
filename_suffix = "%_NaCl"
data_directory  = "measurements/"
append_date     = True #Appends date to the beginning of the directory.
save_raw_data   = False #Save raw data collected from devices.
inc_temp_data   = False #Saves temporary sets of data during data processing.

#Oscilloscope
scope_IP        = "10.32.112.140" #Oscilloscope IP Address
active_channels = [1,2] #Channels to activate and use.
trigger_channel = 2 #Channel for trigger signal.
trigger_level   = 1 #Voltage threshold for postitive slope edge trigger.
channel_setting = {
    #Additional settings to pass to each channel if used.
    1: {"range": 2, "position": -2},
    2: {"range": 40,},
}

# ---------------------------------------------------------------------------- #
# Libraries
# ---------------------------------------------------------------------------- #
import os
from pathlib import Path
#import time
import sys
from datetime import date

import numpy as np
from scipy import signal
from scipy.signal import find_peaks
from scipy.stats import linregress
import scipy.io as sio
from matplotlib import pyplot as plt

from TSL550 import TSL550
import serial.tools.list_ports
from laser_control import *

from RTO.controller import RTO as connectScope

# ---------------------------------------------------------------------------- #
# Check Input
# ---------------------------------------------------------------------------- #
print("Checking inputs.")
#Get command line arguments.
args = sys.argv[1:]
filename = filename_prefix + sys.argv[0] + filename_suffix

#Check laser settings.
laser_sweep_rate = (lambda_stop - lambda_start) / duration
checkSweepRate(laser_sweep_rate)
checkWavelength(lambda_start)
checkWavelength(lambda_stop)

# ---------------------------------------------------------------------------- #
# Initialize Save Directory
# ---------------------------------------------------------------------------- #
today = date.today()
datePrefix = "{}_{}_{}_".format(today.year, today.month, today.day)
prefix = datePrefix if append_date else ""
folderName = prefix + data_directory
folderPath = Path(Path.cwd(), folderName)
print("Saving data to {} in current directory.".format(folderName))
if not os.path.exists(folderPath):
    print("Creating {} directory.".format(folderName))
    os.makedirs(folderPath)

# ---------------------------------------------------------------------------- #
# Initialize Devices
# ---------------------------------------------------------------------------- #
print("Initializing devices.")

#Initialize Laser
print("Initializing laser.")
laser = initLaser()
isOn = laser.on()
laser.power_dBm(power_dBm)
laser.openShutter()
laser.sweep_set_mode(
    continuous=True,
    twoway=True,
    trigger=False,
    const_freq_step=False
)
print("Enabling laser's trigger output.")
laser.trigger_enable_output()
triggerMode = laser.trigger_set_mode("Step")
triggerStep = laser.trigger_set_step(trigger_step)
print("Setting trigger to: {} and step to {}".format(triggerMode, triggerStep))

#Wait here for some reason? For the laser to turn on? Something else?

#Get number of samples to record. Add buffer just in case.
acquireTime = duration + buffer
numSamples = int((acquireTime) * sample_rate)
print("Collecting {} samples @ {} samples/s.".format(numSamples, sample_rate))

#Oscilloscope Settings
print("Initializing Oscilloscope")
scope = connectScope(scope_IP)
scope.acquisition_settings(
    sample_rate = sample_rate,
    duration = acquireTime
)
#Add channels.
for channel in active_channels:
    channelMode = "Trigger" if (channel == trigger_channel) else "Data"
    print("Adding Channel {} - {}".format(channel, channelMode))
    scope.add_channel(
        channel_num = channel,
        **channel_setting[channel]
    )
#Add trigger.
print("Adding Edge Trigger @ {} Volt(s).".format(trigger_level))
scope.edge_trigger(
    source_channel = trigger_channel,
    trigger_level = trigger_level
)

# ---------------------------------------------------------------------------- #
# Collect Data
# ---------------------------------------------------------------------------- #
print('Starting Acquisition')
scope.start_acquisition(
    timeout = duration*3
)

#Sweep Laser
print('Sweeping Laser')
laser.sweep_wavelength(
    start=lambda_start,
    stop=lambda_stop,
    duration=duration,
    number=1
)

#Wait for Measurement Completion
print('Waiting for acquisition to complete.')
scope.wait_for_device()

#Acquire Data
rawData = [None] #Ugly hack to make the numbers line up nicely.
rawData[1:-1] = [scope.get_data_ascii(channel) for channel in active_channels]
wavelengthLog = laser.wavelength_logging()
wavelengthLogSize = laser.wavelength_logging_number()

#Optional Save Raw Data
if save_raw_data:
    print("Saving raw data.")
    for channel in active_channels:
        with open(foldername + "CHAN{}_Raw.txt".format(channel), "w") as out:
            out.write(str(rawData[channel]))
    with open(foldername + "Wavelength_Log.txt") as out:
        out.write(str(wavelengthLog))

# ---------------------------------------------------------------------------- #
# Process Data
# ---------------------------------------------------------------------------- #
#Todo: check height and distance settings below and adjust if necessary.
peaks, _ = find_peaks(rawData[trigger_channel], height = 3, distance = 5)

print('=' * 30)
print("Expected number of wavelength points: {}".format(wavelengthLogSize))
print("Measured number of wavelength points: {}".format(len(peaks)))
print('=' * 30)

#ToDo: Make the following work for an arbitrary number of channels.

#Take data points between first and last peaks in trigger channel signal.
deviceData = rawData[0][peaks[0]:peaks[-1]]
#Time relative to the first collected data point.
deviceTime = np.arange(len(deviceData)) / sample_rate

modPeaks = peaks - peaks[0] #Make relative to the first point.
modTime = modPeaks / sample_rate
fitting = np.polyfit(modTime, wavelengthLog, 2) #Least-squares fit.
mapping = np.poly1d(fitting) #Create function mapping time to wavelength.
deviceWavelength = mapping(deviceTime) #Get wavelengths at times.

#Generate Data Hashes
wavelengthHash = hash(tuple(deviceWavelength))
dataHash = hash(tuple(deviceData))

# ---------------------------------------------------------------------------- #
# Generate Visuals & Save Data
# ---------------------------------------------------------------------------- #
#Create Visualization
plt.figure()
plt.plot(deviceWavelength, deviceData)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Power (au)')
plt.grid(True)
plt.tight_layout()

#Save Data
plt.savefig(folderName + filename + ".png")
sio.savemat(
    folderName + filename + ".mat",
    {
        'wavelength': wavelengthHash,
        'power': dataHash
    }
)
np.savez(
    folderName + filename + ".npz",
    wavelength = np.squeeze(deviceWavelength),
    power = np.squeeze(deviceData)
)

#Show Vizualization
plt.show()
# ---------------------------------------------------------------------------- #
# Cleanup
# ---------------------------------------------------------------------------- #
