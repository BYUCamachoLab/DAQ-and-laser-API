from Measurement import *
from abc import ABC, abstractmethod
import os
from matplotlib import pyplot as plt
import numpy as np
from Measurement.Measurement import Measurement, _generate_random_color_hexcode


class SweepMeasurement(Measurement, ABC):
    """
    Class that represents a wavelength sweep measurement, implementing the Measurement abstract class.

    - The set_sweep_parameters function or the constructor that takes in all the parameters must be called before
    perform_measurement or a ParametersNotSpecified exception will be thrown.

    - The perform_measurement function must be called before the data visualization or save functions,
    or a MeasurementNotPerformedException will be thrown.

    Instance variables:


    """

    OSCOPE_CHANNEL_COLORS = ['y', 'g', 'm', 'b', 'k']

    def __init__(self):
        """
        Default constructor for initializing the class and then setting the sweep parameters later.
        """
        super().__init__()
        self.wavelength_startpoint = None
        self.wavelength_endpoint = None
        self.duration = None
        self.trigger_step = None
        self.sample_rate = None
        self.power_dBm = None
        self.measurement_folder = None
        self.output_channel_list = None
        self.device_type = None
        self.description = None
        self.output_ports = None
        self.laser = None
        self.directory = None
        self.result_data = None

    def __init__(self, lambda_start: float, lambda_end: float, dur: float, trig_step: float, samp_rate: float,
                 power: float, folder, chan_list: list, dev_type: str, desc: str, ports: list):
        """
        Constructor added for redundancy in case the user wants to set the sweep parameters at once.
        """
        super().__init__()
        self.set_sweep_parameters(lambda_start, lambda_end, dur, trig_step,
                                  samp_rate, power, folder, chan_list, dev_type, desc, ports)
        self.laser = None
        self.directory = None
        self.result_data = None
        self.parameters_set = True

    def set_sweep_parameters(self, lambda_start: float, lambda_end: float, dur: float, trig_step: float,
                             samp_rate: float, power: float, folder, chan_list: list, dev_type: str,
                             desc: str, ports: list):
        """
        Setter for all the variables of the sweep.
        """
        self.wavelength_startpoint = lambda_start
        self.wavelength_endpoint = lambda_end
        self.duration = dur
        self.trigger_step = trig_step
        self.sample_rate = samp_rate
        self.power_dBm = power
        self.measurement_folder = folder
        self.output_channel_list = chan_list
        self.device_type = dev_type
        self.description = desc
        self.output_ports = ports
        self.parameters_set = True

    def _create_directory(self):

        subdir = self.device_type + "/" + self.description

        # Create directories if needed
        self.directory = self.measurement_folder + "/" + subdir + "/"
        print("Folder for output: " + self.directory)

        if os.path.exists(self.directory):
            user_choice = None
            print("This folder already exists, so it probably has measurement data in it.\n"
                  "With this description the data will be overwritten, do you wish to proceed? y/n: ")
            while user_choice != "n" and user_choice != "y":
                user_choice = input()
                if user_choice == "n":
                    print("Please choose a different description to prevent data deletion.\n")
                    exit()
                elif user_choice == "y":
                    print("Overwriting data...")
                else:
                    print("Invalid choice. Please type y or n: ")
        else:
            os.makedirs(self.directory)

    @abstractmethod
    def _initialize_devices(self):
        pass

    @abstractmethod
    def _laser_sweep_start(self):
        pass

    @abstractmethod
    def _read_data(self):
        pass

    @abstractmethod
    def _interpolate_data(self):
        pass

    def _perform_measurement(self):
        self._check_parameters_set()
        self._create_directory()
        self._initialize_devices()
        self._laser_sweep_start()
        self._read_data()
        self.result_data = self._interpolate_data()

    def _visualize_data(self, save_figure):
        plt.figure()
        for i in range(1, len(self.output_ports) + 1):
            if i < len(self.OSCOPE_CHANNEL_COLORS) + 1:
                color = self.OSCOPE_CHANNEL_COLORS[i - 1]
            else:
                color = _generate_random_color_hexcode()
            plt.plot(self.result_data[0, :], self.result_data[i, :], color)

        legends = []
        for i in range(len(self.output_ports)):
            legends.append(self.output_ports[i])

        plt.legend(legends)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Voltage (au)')
        plt.grid(True)
        plt.tight_layout()

        if save_figure:
            plt.savefig(self.directory + "graph")

    def _save_data(self):
        for i in range(len(self.output_ports)):
            np.savez(self.directory + self.output_ports[i] + "_data.npz",
                     wavelength=np.squeeze(self.result_data[0, :]), power=np.squeeze(self.result_data[i + 1, :]))

    def _get_data(self):
        return self.result_data
