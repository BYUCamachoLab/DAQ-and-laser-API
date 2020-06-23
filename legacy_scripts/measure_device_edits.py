# ---------------------------------------------------------------------------- #
# Import libraries
# ---------------------------------------------------------------------------- #

import nidaqmx
from Laser import TSL550
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
# Sweep parameters
# ---------------------------------------------------------------------------- #
lambda_start    = 1500  # 1500 is lower limit
lambda_stop     = 1630  # 1630 is limit
duration        = 5
trigger_step    = 0.01
sample_rate     = 100e3   # DO NOT CHANGE
filename_prefix = "sample_device"
data_directory  = "demo_2/"
power_dBm       = -10

# Laser Port Parameters
address = "COM4"

# ---------------------------------------------------------------------------- #
# Check input
# ---------------------------------------------------------------------------- #
args = sys.argv
numArgs = len(args)
device_type = None
output = "test"
for i in range(1, numArgs):
    if i == 2:
        device_type = args[i]
    if i == 1:
        output = args[i]


# Check laser's sweep rate
laser_sweep_rate = (lambda_stop - lambda_start) / duration
if laser_sweep_rate > 100 or laser_sweep_rate < 0.5:
    raise AttributeError("Invalid laser sweep speed of %f. Must be between 0.5 and 100 nm/s." % laser_sweep_rate)

# TODO: Check for the triggers input too...

# Make folder for Today
today = date.today()

# Create path if needed

subdir = ""
if device_type is not "":
    subdir += device_type + "/"
else:
    subdir = "not_specified_device"

folder_name = str(today.month) + "_" + str(today.day) + "_measurements/" + subdir + "/" + output + "/"
folder_path = Path(os.getcwd(), folder_name)
print(folder_path)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
# ---------------------------------------------------------------------------- #
# Setup devices
# ---------------------------------------------------------------------------- #

# Initialize laser
print("Opening connection to laser...")
print("Laser response: ")
laser = initLaser(address)
laser.on()
laser.power_dBm(power_dBm)
laser.openShutter()
laser.sweep_set_mode(continuous=True, twoway=True, trigger=False, const_freq_step=False)
laser.trigger_enable_output()
print("Mode:", laser.trigger_set_mode("Step"))
print("Step size: dlambda = ", laser.trigger_set_step(trigger_step))

# Get number of samples to record. Add buffer just in case.
numSamples = int(duration * 2 * sample_rate)
print("Number of samples: ", numSamples)

time.sleep(0.3)

# Initialize DAQ
task = initTask(channel=["ai0", "ai1"], sampleRate=sample_rate, samples_per_chan=numSamples)

# ---------------------------------------------------------------------------- #
# Run sweep
# ---------------------------------------------------------------------------- #

laser.sweep_wavelength(start=lambda_start, stop=lambda_stop, duration=duration, number=1)
data = np.array(task.read(number_of_samples_per_channel=numSamples, timeout=3*duration))

wavelength_logging = laser.wavelength_logging()

# ---------------------------------------------------------------------------- #
# Process data
# ---------------------------------------------------------------------------- #

peaks, _ = find_peaks(data[1, :], height=3, distance=5)

print('==========================================')
print("Expected number of wavelength points: %d" % int(laser.wavelength_logging_number()))
print("Actual wavelength points measured: %d" % np.size(peaks))
print('==========================================')

device_data = data[0, peaks[0]:peaks[-1]]
device_time = np.arange(0, device_data.size) / sample_rate

modPeaks          = peaks - peaks[0]
modTime           = modPeaks / sample_rate
z                 = np.polyfit(modTime, wavelength_logging, 2)
p                 = np.poly1d(z)
device_wavelength = p(device_time)


device_wavelength_mod = hash(tuple(device_wavelength))
device_data_mod = hash(tuple(device_data))

# ---------------------------------------------------------------------------- #
# Visualize and save results
# ---------------------------------------------------------------------------- #

plt.figure()
plt.plot(device_wavelength, device_data)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Power (au)')
plt.title(output)
plt.grid(True)
plt.tight_layout()
plt.savefig(folder_name + output + ".png")
sio.savemat(folder_name + "datamat.mat",
            {'wavelength': device_wavelength_mod, 'power': device_data_mod})
np.savez(folder_name + "data.npz",
         wavelength=np.squeeze(device_wavelength), power=np.squeeze(device_data))
plt.show()

# ---------------------------------------------------------------------------- #
# Cleanup devices
# ---------------------------------------------------------------------------- #

laser.wavelength(1580)

task.close()
quit()
