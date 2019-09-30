# ---------------------------------------------------------------------------- #
# Sweep parameters
# ---------------------------------------------------------------------------- #
lambda_start    = 1550 #1500 is lower limit
lambda_stop     = 1630 #1630 is limit
duration        = 5
trigger_step    = 0.01
sample_rate     = 100e3   # DO NOT CHANGE
filename_prefix = "sample_device"
data_directory  = "demo_2/"
power_dBm       = 3
address = "COM4"

# ---------------------------------------------------------------------------- #
# Import libraries
# ---------------------------------------------------------------------------- #

import nidaqmx
from TSL550 import TSL550
import serial.tools.list_ports
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from testSweep import *
from scipy.signal import find_peaks
from scipy.stats import linregress
import os
import scipy.io as sio
from pathlib import Path
import time
import sys
from datetime import date

# ---------------------------------------------------------------------------- #
# Check input
# ---------------------------------------------------------------------------- #
args = sys.argv
numArgs = len(args)
device_type = None
description = None
output = []
if numArgs < 2:
    raise ValueError("Please specify device type.")
for i in range(1, numArgs):
    if i == 2:
        description = args[i]
    if i == 1:
        device_type = args[i]
    if i > 2:
        output.append(args[i])

# Check laser's sweep rate
laser_sweep_rate = (lambda_stop - lambda_start) / duration
if laser_sweep_rate > 100 or laser_sweep_rate < 0.5:
    raise AttributeError("Invalid laser sweep speed of %f. Must be between 0.5 and 100 nm/s." % laser_sweep_rate)

# TODO: Check for the triggers input too...

# Make folder for Today
today = date.today()

subdir = device_type + "/" + description

# Create directories if needed
foldername = str(today.month) + "_" + str(today.day) + "_measurements/" + subdir
print(foldername)
for i in range(len(output)):
    final_foldername = foldername + "/" + output[i]
    if not os.path.exists(foldername):
        os.makedirs(foldername)
# ---------------------------------------------------------------------------- #
# Setup devices
# ---------------------------------------------------------------------------- #

# Initialize laser
laser = initLaser(address)
isOn = laser.on()
laser.power_dBm(power_dBm)
laser.openShutter()
laser.sweep_set_mode(continuous=True, twoway=True, trigger=False, const_freq_step=False)
laser.trigger_enable_output()
print(laser.trigger_set_mode("Step"))
print(laser.trigger_set_step(trigger_step))

# Get number of samples to record. Add buffer just in case.
numSamples = int(duration * 2 * sample_rate)
print(numSamples)

time.sleep(0.3)

# Initialize DAQ
task = []
for i in range(len(output)):
    cardNum = int((i + 1) / 4) + 1
    task.append(initTask(device="cDAQ1Mod" + str(cardNum), channel=["ai" + str((i + 1) % 4), "ai0"], sampleRate=sample_rate,
                         samples_per_chan=numSamples))

# ---------------------------------------------------------------------------- #
# Run sweep
# ---------------------------------------------------------------------------- #

data = []
wavelength_data = []
for i in range(len(output)):
    laser.sweep_wavelength(start=lambda_start, stop=lambda_stop, duration=duration, number=1)
    data.append(np.array(task[i].read(number_of_samples_per_channel=numSamples, timeout=3*duration)))
    wavelength_data.append(laser.wavelength_logging())

wavelength_logging = laser.wavelength_logging()

# ---------------------------------------------------------------------------- #
# Process data
# ---------------------------------------------------------------------------- #

peaks = []
for i in range(len(output)):
    peak, _ = find_peaks(data[i][1, :], height=3, distance=5)
    peaks.append(peak)

print('==========================================')
print("Expected number of wavelength points: %d" % int(laser.wavelength_logging_number()))
print("Actual wavelength points measured in channel 1: %d" % len(peaks))
print('==========================================')

device_data = []
device_times = []
device_wavelengths = []
for i in range(len(output)):
    device_data.append(data[i][0, peaks[i][0]:peaks[i][-1]])
    device_times.append(np.arange(0, device_data[i].size) / sample_rate)


#device_time = np.arange(0, device_data[0].size) / sample_rate
for i in range(len(output)):
    modPeaks = peaks[i] - peaks[i][0]
    modTime = modPeaks / sample_rate
    z = np.polyfit(modTime, wavelength_data[i], 2)
    p = np.poly1d(z)
    device_wavelengths.append(p(device_times[i]))

# ---------------------------------------------------------------------------- #
# Visualize and save results
# ---------------------------------------------------------------------------- #

plt.figure()
colors = ['y', 'g', 'm', 'b']
for i in range(len(output)):
    plt.plot(device_wavelengths[i], device_data[i], colors[i])

legends = []
for i in range(len(output)):
    legends.append(output[i])

plt.legend(legends)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Power (au)')
plt.grid(True)
plt.tight_layout()

figname = foldername + "/graph"
for i in range(len(output)):
    figname += output[i]
figname += ".png"
plt.savefig(figname)

for i in range(len(output)):
    np.savez(foldername + "/" + output[i] + "_data.npz",
             wavelength=np.squeeze(device_wavelengths[i]), power=np.squeeze(device_data[i]))

plt.show()

# ---------------------------------------------------------------------------- #
# Cleanup devices
# ---------------------------------------------------------------------------- #

for i in range(len(output)):
    task[i].close()

quit()
