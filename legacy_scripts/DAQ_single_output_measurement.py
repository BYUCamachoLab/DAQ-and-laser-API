# ---------------------------------------------------------------------------- #
# Import libraries
# ---------------------------------------------------------------------------- #

from DAQinterface import NIDAQInterface
from Laser.TSL550 import TSL550
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import os
import scipy.io as sio
import time
import sys
from datetime import date


def single_output_measurement(device_type: str, description: str, output_port=None):

    # ---------------------------------------------------------------------------- #
    # Sweep parameters
    # ---------------------------------------------------------------------------- #
    wavelength_startpoint = 1560
    wavelength_endpoint = 1620
    duration = 5
    trigger_step = 0.01
    sample_rate = NIDAQInterface.CARD_TWO_MAX_SAMPLE_RATE
    power_dBm = 5
    measurement_folder = "Measurement_Data/"

    # Check laser's sweep rate
    laser_sweep_rate = (wavelength_endpoint - wavelength_startpoint) / duration
    if laser_sweep_rate > 100 or laser_sweep_rate < 0.5:
        raise AttributeError("Invalid laser sweep speed of %f. Must be between 0.5 and 100 nm/s." % laser_sweep_rate)

    # Make folder for Today
    today = date.today()
    subdir = None
    if output_port is not None and not "":
        subdir = device_type + "/" + description + "_" + output_port
    else:
        subdir = device_type + "/" + description

    # Create directories if needed
    folder_name = measurement_folder + str(today.month) + "_" + str(today.day) + "_measurements/" + subdir + "/"
    print("Folder for output: " + folder_name)

    if os.path.exists(folder_name):
        user_choice = None
        print("This folder already exists, so it probably has measurement data in it.\n"
              "With this description the data will be overwritten, do you wish to proceed? y/n: ")
        while user_choice != "n" and user_choice != "y":
            user_choice = input()
            if user_choice == "n":
                print("Please choose a different description to prevent data deletion.\n")
                return
            elif user_choice == "y":
                print("Overwriting data...")
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
    num_samples = int(duration * 1.5 * sample_rate)
    time.sleep(0.3)

    # Initialize DAQ
    daq = NIDAQInterface()
    daq.initialize(["cDAQ1Mod1/ai0", "cDAQ1Mod1/ai1"], sample_rate=sample_rate, samples_per_chan=num_samples)

    # ---------------------------------------------------------------------------- #
    # Run sweep
    # ---------------------------------------------------------------------------- #

    laser.sweep_wavelength(start=wavelength_startpoint, stop=wavelength_endpoint, duration=duration, number=1)
    data = np.array(daq.read(1.5*duration))
    times_read = np.arange(0, duration * 1.5, 1. / sample_rate)
    wavelength_logging = laser.wavelength_logging()

    # ---------------------------------------------------------------------------- #
    # Process data
    # ---------------------------------------------------------------------------- #

    peaks, _ = find_peaks(data[0, :], height=3, distance=5)

    device_data = data[0, peaks]
    device_time = times_read[peaks]

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
    plt.title(output_port)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(folder_name + "plot")
    sio.savemat(folder_name + "datamat.mat",
                {'wavelength': device_wavelength_mod, 'power': device_data_mod})
    np.savez(folder_name + "data.npz",
             wavelength=np.squeeze(device_wavelength), power=np.squeeze(device_data))
    plt.show()


if __name__ == "__main__":
    args = sys.argv
    if len(args) < 2:
        raise ValueError("Please specify device type.")
    if len(args) < 3:
        raise ValueError("Please write a description for the measurement.")
    device_type = args[1]
    description = args[2]
    output_port = ""
    if len(args) > 3:
        output_port = args[3]
    single_output_measurement(device_type, description, output_port)
