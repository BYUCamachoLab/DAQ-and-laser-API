from abc import ABC, abstractmethod


class Measurement(ABC):

    """
    Abstract class for defining an interface that the different measurement classes should follow.
    """

    @abstractmethod
    def perform_measurement(self):
        """
        Runs the specified measurement.

        :return: The data in a form that is typical to the measurement type.
                 E.g. for a wavelength sweep, it should be a 2D array of Voltage vs. Wavelength readings, for a
                 multiple-shot quadrature measurement, it should be a 1D array of Voltage readings.
        """
        pass

    @abstractmethod
    def visualize_data(self):
        """
        Plots the data in a useful form for the measurement type.
        Requirement: The function perform_measurement must have been run before running this function.

        :return: n/a
        """
        pass

    @abstractmethod
    def save_data(self):
        """
        Saves the data in a useful form.
        Requirement: The function perform_measurement must have been run before running this function.

        :return: n/a
        """
        pass

    @abstractmethod
    def save_figure(self):
        """
        Saves plots of the data into a file in a useful form.

        :return: n/a
        """
        pass
