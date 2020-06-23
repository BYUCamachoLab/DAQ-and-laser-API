from Measurement import *
from abc import ABC, abstractmethod
import os
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import savemat
from Measurement.Measurement import Measurement, _generate_random_color_hexcode


class SweepMeasurement(Measurement, ABC):
    """
    Class that represents a wavelength sweep measurement, extending the Measurement abstract class.
    Responsible for performing a sweep measurement, delegating the device-specific behavior to implementing classes
    through the private abstract methods.

    - The set_sweep_parameters function or the constructor that takes in all the parameters must be called before
    perform_measurement or a ParametersNotSpecified exception will be thrown.

    - The perform_measurement function must be called before the data visualization or save functions,
    or a MeasurementNotPerformedException will be thrown.

    Instance variables:

    :param self.wavelength_startpoint - The starting wavelength point of the sweep.
    :param self.wavelength_endpoint - The end wavelength point of the sweep
    :param self.duration - The time it takes to do the sweep.
    :param self.trigger_step - The time step between points read.
    :param self.sample_rate - The time step between laser triggers.
    :param self.power_dBm - The power setting of the laser.
    :param self.measurement_folder - The folder to store measurement data in.
    :param self.output_channel_list - The list of addresses of the measurement device ports.
    :param self.device_type - The type of silicon photonic device being measured.
    :param self.description - Extra information about the measurement being done.
    :param self.output_ports - The label of the output ports of the silicon photonic device.
    :param self.path - The full path where the measurement data will be stored.
    :param self.result_data - The data resulting from the measurement.

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
        self.path = None
        self.result_data = None

    def __init__(self, lambda_start: float, lambda_end: float, dur: float, trig_step: float, samp_rate: float,
                 power: float, folder, chan_list: list, dev_type: str, desc: str, ports: list):
        """
        Constructor added for redundancy in case the user wants to set the sweep parameters at once.
        """
        super().__init__()
        self.set_sweep_parameters(lambda_start, lambda_end, dur, trig_step,
                                  samp_rate, power, folder, chan_list, dev_type, desc, ports)
        self.path = None
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
        """
        Creates the path where the measurement results will be saved.
        The path path will be the path/device type/description.

        If the requested path name already exists, then it prompts the user whether they want to
        overwrite the folder or not. If they select n, the program exits.
        """
        subdir = self.device_type + "/" + self.description

        # Create directories if needed
        self.path = self.measurement_folder + "/" + subdir + "/"
        print("Folder for output: " + self.path)

        if os.path.exists(self.path):
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
            os.makedirs(self.path)

    @abstractmethod
    def _initialize_devices(self):
        """
        Initializes any devices that need it.
        :return: n/a
        """
        pass

    @abstractmethod
    def _laser_sweep_start(self):
        """
        Starts the laser sweep.
        :return: n/a
        """
        pass

    @abstractmethod
    def _read_data(self):
        """
        Reads the data from the detectors.
        :return: n/a
        """
        pass

    @abstractmethod
    def _interpolate_data(self):
        """
        Processes the raw data from the devices, performs any post-processing that is required.
        :return:
        """
        pass

    def _perform_measurement(self):
        """
        Runs the measurement sequence.
        :return: n/a
        """
        self._check_parameters_set()
        self._create_directory()
        self._initialize_devices()
        self._laser_sweep_start()
        self._read_data()
        self.result_data = self._interpolate_data()

    def _visualize_data(self, save_figure, show_figure):
        """
        Plots the data from the measurement.
        :param save_figure: whether the figures should be saved.
        :param show_figure: whether the figure should be shown.
        :return: n/a
        """
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
            plt.savefig(self.path + "graph")
        if show_figure:
            plt.show()

    def _save_data(self, save_npz=True, save_mat=True):
        """
        Saves the data to the measurement directory.
        :param save_npz: whether the data should be saved as an npz file.
        :param save_mat: whether the data should be saved as a mat file.
        :return:
        """
        for i in range(len(self.output_ports)):
            if save_npz:
                np.savez(self.path + self.output_ports[i] + "_data.npz",
                         wavelength=np.squeeze(self.result_data[0, :]), power=np.squeeze(self.result_data[i + 1, :]))
            if save_mat:
                savemat(self.path + self.output_ports[i] + "data.mat",
                        {"wavelength": np.squeeze(self.result_data[0, :]),
                         "power": np.squeeze(self.result_data[i + 1, :])}, appendmat=False)

    def _get_data(self):
        """
        :return: the data from the measurement
        """
        return self.result_data
