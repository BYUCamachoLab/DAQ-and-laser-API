# ---------------------------------------------------------------------------- #
# Import libraries
# ---------------------------------------------------------------------------- #

from DAQinterface import NIDAQInterface
from TSL550 import TSL550
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import os
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
device_type = None
description = None
output_ports = []
if len(args) < 2:
    raise ValueError("Please specify device type.")
for i in range(1, len(args)):
    if i == 1:
        device_type = args[i]
    if i == 2:
        description = args[i]
    if i > 2:
        output_ports.append(args[i])

# Check laser's sweep rate
laser_sweep_rate = (wavelength_endpoint - wavelength_startpoint) / duration
if laser_sweep_rate > 100 or laser_sweep_rate < 0.5:
    raise AttributeError("Invalid laser sweep speed of %f. Must be between 0.5 and 100 nm/s." % laser_sweep_rate)

# Make folder for Today
today = date.today()
subdir = device_type + "/" + description

# Create directories if needed
folder_name = str(today.month) + "_" + str(today.day) + "_measurements/" + subdir
print(folder_name)
for i in range(len(output_ports)):
    folder_name = folder_name + "/" + output_ports[i]

if os.path.exists(folder_name):
    user_choice = None
    print("This folder already exists, so it probably has measurement data in it.\n"
          "With this description the data will be overwritten, do you wish to proceed? y/n: ")
    while user_choice != "n" or user_choice != "y":
        user_choice = input()
        if user_choice == "n":
            print("Please choose a different description to prevent data deletion.")
        else:
            print("Invalid choice. Please type y or n: ")
else:
    os.makedirs(folder_name)

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
daq.initialize(["cDAQ1Mod1/ai0", "cDAQ1Mod1/ai1", "cDAQ1Mod1/ai2", "cDAQ1Mod1/ai3", "cDAQ1Mod2/ai0"],
               sample_rate=sample_rate, samples_per_chan=num_samples)

# ---------------------------------------------------------------------------- #
# Run sweep
# ---------------------------------------------------------------------------- #

laser.sweep_wavelength(start=wavelength_startpoint, stop=wavelength_endpoint, duration=duration, number=1)
data = daq.read(1.1*duration)
wavelength_logging = laser.wavelength_logging()

# ---------------------------------------------------------------------------- #
# Process data
# ---------------------------------------------------------------------------- #

peaks = []
for i in range(len(output_ports)):
    peak, _ = find_peaks(data[i][1, :], height=3, distance=5)
    peaks.append(peak)


device_data = []
device_times = []
device_wavelengths = []
for i in range(len(output_ports)):
    device_data.append(data[i][0, peaks[i][0]:peaks[i][-1]])
    device_times.append(np.arange(0, device_data[i].size) / sample_rate)

for i in range(len(output_ports)):
    modPeaks = peaks[i] - peaks[i][0]
    modTime = modPeaks / sample_rate
    z = np.polyfit(modTime, wavelength_logging[i], 2)
    p = np.poly1d(z)
    device_wavelengths.append(p(device_times[i]))

# ---------------------------------------------------------------------------- #
# Visualize and save results
# ---------------------------------------------------------------------------- #

plt.figure()
colors = ['y', 'g', 'm', 'b']
for i in range(len(output_ports)):
    plt.plot(device_wavelengths[i], device_data[i], colors[i])

legends = []
for i in range(len(output_ports)):
    legends.append(output_ports[i])

plt.legend(legends)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Power (au)')
plt.grid(True)
plt.tight_layout()

figname = folder_name + "/graph"
for i in range(len(output_ports)):
    figname += output_ports[i]
figname += ".png"
plt.savefig(figname)

for i in range(len(output_ports)):
    np.savez(folder_name + "/" + output_ports[i] + "_data.npz",
             wavelength=np.squeeze(device_wavelengths[i]), power=np.squeeze(device_data[i]))

plt.show()
