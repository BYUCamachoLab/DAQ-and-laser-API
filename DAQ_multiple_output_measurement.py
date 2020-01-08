# ---------------------------------------------------------------------------- #
# Import libraries
# ---------------------------------------------------------------------------- #

from DAQinterface import NIDAQInterface
from TSL550.TSL550 import TSL550
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import os
import time
import sys
from datetime import date


def multiple_output_sweep(device_type: str, description: str, output_ports: list):

    # ---------------------------------------------------------------------------- #
    # Sweep parameters
    # ---------------------------------------------------------------------------- #
    wavelength_startpoint = 1560
    wavelength_endpoint = 1620
    duration = 5
    trigger_step = 0.01
    sample_rate = NIDAQInterface.CARD_TWO_MAX_SAMPLE_RATE
    power_dBm = 10
    measurement_folder = "Measurement_Data/"
    output_channel_list = ["cDAQ1Mod1/ai1", "cDAQ1Mod1/ai2", "cDAQ1Mod1/ai3", "cDAQ1Mod2/ai0"]

    # ---------------------------------------------------------------------------- #
    # Check input
    # ---------------------------------------------------------------------------- #

    # Check laser's sweep rate
    laser_sweep_rate = (wavelength_endpoint - wavelength_startpoint) / duration
    if laser_sweep_rate > 100 or laser_sweep_rate < 0.5:
        raise AttributeError("Invalid laser sweep speed of %f. Must be between 0.5 and 100 nm/s." % laser_sweep_rate)

    # Make folder for Today
    today = date.today()
    subdir = device_type + "/" + description

    # Create directories if needed
    folder_name = measurement_folder + str(today.month) + "_" + str(today.day) + "_measurements/" + subdir + "/"
    for i in range(len(output_ports)):
        if i > 0:
            folder_name = folder_name + "_"
        folder_name = folder_name + output_ports[i]
    folder_name = folder_name + "/"
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

    # Get number of samples to record.
    num_samples = int(duration * sample_rate*1.5)
    time.sleep(0.3)

    # Initialize DAQ
    daq = NIDAQInterface()
    daq.initialize(["cDAQ1Mod1/ai0"],
                   sample_rate=sample_rate, samples_per_chan=num_samples)
    for i in range(len(output_ports)):
        daq.add_channel(output_channel_list[i])

    # ---------------------------------------------------------------------------- #
    # Run sweep
    # ---------------------------------------------------------------------------- #

    laser.sweep_wavelength(start=wavelength_startpoint, stop=wavelength_endpoint, duration=duration, number=1)
    data = np.array(daq.read(duration*1.5))
    times_read = np.arange(0, duration*1.5, 1./sample_rate)
    wavelength_logging = np.array(laser.wavelength_logging())

    # ---------------------------------------------------------------------------- #
    # Process data
    # ---------------------------------------------------------------------------- #

    peaks, _ = find_peaks(data[0, :], height=3, distance=5)
    device_data = []
    device_times = []
    for i in range(1, len(output_ports) + 1):
        device_data.append(data[i, peaks])
        device_times.append(times_read[peaks])

    peak_spacing = peaks - peaks[0]
    time_between_peaks = peak_spacing / sample_rate
    time_wavelength_conversion_fit = np.polyfit(time_between_peaks, wavelength_logging, 2)
    conversion_fit_function = np.poly1d(time_wavelength_conversion_fit)
    lined_up_wavelength_points = np.array(conversion_fit_function(device_times))

    # ---------------------------------------------------------------------------- #
    # Visualize and save results
    # ---------------------------------------------------------------------------- #

    plt.figure()
    colors = ['y', 'g', 'm', 'b']
    for i in range(len(output_ports)):
        plt.plot(lined_up_wavelength_points[i], device_data[i], colors[i])

    legends = []
    for i in range(len(output_ports)):
        legends.append(output_ports[i])

    plt.legend(legends)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Power (au)')
    plt.grid(True)
    plt.tight_layout()

    figname = folder_name + "graph"
    plt.savefig(figname)

    for i in range(len(output_ports)):
        np.savez(folder_name + output_ports[i] + "_data.npz",
                 wavelength=np.squeeze(lined_up_wavelength_points[i]), power=np.squeeze(device_data[i]))

    plt.show()


if __name__ == "__main__":
    args = sys.argv
    device_type = None
    description = None
    output_ports = []
    if len(args) < 2:
        raise ValueError("Please specify device type.")
    if len(args) < 3:
        raise ValueError("Please give a description to the measurement.")
    if len(args) < 4:
        raise ValueError("Please specify at least one output port")
    for j in range(1, len(args)):
        if j == 1:
            device_type = args[j]
        if j == 2:
            description = args[j]
        if j > 2:
            output_ports.append(args[j])
    multiple_output_sweep(device_type, description, output_ports)
