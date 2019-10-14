# ---------------------------------------------------------------------------- #
# Import libraries
# ---------------------------------------------------------------------------- #

from DAQinterface import NIDAQInterface
from TSL550 import TSL550
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import os
import scipy.io as sio
from pathlib import Path
import time
import sys
from datetime import date

# ---------------------------------------------------------------------------- #
# Sweep parameters
# ---------------------------------------------------------------------------- #
wavelength_startpoint = 1560
wavelength_endpoint = 1620
duration = 5
trigger_step = 0.01
sample_rate = NIDAQInterface.CARD_TWO_MAX_SAMPLE_RATE
power_dBm = 5

# ---------------------------------------------------------------------------- #
# Check input
# ---------------------------------------------------------------------------- #
args = sys.argv
if len(args) < 2:
    raise ValueError("Please specify device type.")
device_type = args[1]
output = args[2]
description = ""
if len(args) >= 2:
    for i in range(len(args) - 2):
        description += args[i + 2]


# Check laser's sweep rate
laser_sweep_rate = (wavelength_endpoint - wavelength_startpoint) / duration
if laser_sweep_rate > 100 or laser_sweep_rate < 0.5:
    raise AttributeError("Invalid laser sweep speed of %f. Must be between 0.5 and 100 nm/s." % laser_sweep_rate)

# Make folder for Today
today = date.today()

# Create directory if needed

subdir = ""
if device_type is not None:
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
laser = TSL550(TSL550.LASER_PORT)
laser.on()
laser.power_dBm(power_dBm)
laser.openShutter()
laser.sweep_set_mode(continuous=True, twoway=True, trigger=False, const_freq_step=False)
laser.trigger_enable_output()
print("Mode:", laser.trigger_set_mode("Step"))
print("Step size: dlambda = ", laser.trigger_set_step(trigger_step))

# Get number of samples to record. Add buffer just in case.
num_samples = int(duration * 1.1 * sample_rate)
time.sleep(0.3)

# Initialize DAQ
daq = NIDAQInterface()
daq.initialize(["cDAQ1Mod1/ai0", "cDAQ1Mod1/ai1"], sample_rate=sample_rate, samples_per_chan=num_samples)

# ---------------------------------------------------------------------------- #
# Run sweep
# ---------------------------------------------------------------------------- #

laser.sweep_wavelength(start=wavelength_startpoint, stop=wavelength_endpoint, duration=duration, number=1)
data = np.array(daq.read(1.1*duration))

wavelength_logging = laser.wavelength_logging()

# ---------------------------------------------------------------------------- #
# Process data
# ---------------------------------------------------------------------------- #

# TODO: Update data postprocessing code

peaks, _ = find_peaks(data[0, :], height=3, distance=5)

print('==========================================')
print("Expected number of wavelength points: %d" % int(laser.wavelength_logging_number()))
print("Actual wavelength points measured: %d" % np.size(peaks))
print('==========================================')

device_data = data[0, peaks[0]:peaks[-1]]
device_time = np.arange(0, device_data.size) / sample_rate

modPeaks = peaks - peaks[0]
modTime = modPeaks / sample_rate
z = np.polyfit(modTime, wavelength_logging, 2)
p = np.poly1d(z)
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
