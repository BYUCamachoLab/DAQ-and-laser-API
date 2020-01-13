from abc import ABC, abstractmethod
import random

def _generate_random_color_hexcode():
    r = lambda: random.randint(0, 255)
    return '#%02X%02X%02X' % (r(), r(), r())


class Measurement(ABC):

    """
    Abstract class for defining an interface that the different measurement classes should follow.

    The user must implement the four abstract private methods in a child class for the specific measurement type
    they want to run. The abstract functions must have the same functionality as the public methods that match
    their name.

    Instance variables:

    :param self.measurement_performed - A boolean value indicating whether a measurement has been performed yet.
    """

    class MeasurementNotPerformedException(Exception):
        """
        To be thrown when a user tries to plot or save data before performing a measurement.
        """
        pass

    class ParametersNotSetException(Exception):
        """
        To be thrown when the user tries to perform the measurement without first setting the measurement parameters.
        """
        pass

    def __init__(self):
        self.measurement_performed = False
        self.parameters_set = False

    def _check_measurement_performed(self):
        """
        Raises an exception if no measurement was performed. In this case there is no data to be plotted
        and no data and figures to be saved.
        :return: void
        """
        if not self.measurement_performed:
            raise self.MeasurementNotPerformedException("Error: The measurement has not been performed yet"
                                                        "or the flag has not been set.")

    def _check_parameters_set(self):
        if not self.parameters_set:
            raise self.ParametersNotSetException("The sweep parameters have not been set yet. Set the parameters using"
                                                 "a constructor or the set_sweep_parameters() method.")

    @abstractmethod
    def _perform_measurement(self):
        pass

    @abstractmethod
    def _visualize_data(self, save_figure):
        pass

    @abstractmethod
    def _save_data(self):
        pass

    @abstractmethod
    def _get_data(self):
        pass

    def perform_measurement(self):
        """
        Runs the specified measurement.

        :return: The data in a form that is typical to the measurement type.
                 E.g. for a wavelength sweep, it should be a 2D array of Voltage vs. Wavelength readings, for a
                 multiple-shot quadrature measurement, it should be a 1D array of Voltage readings.
        """
        self._perform_measurement()
        self.measurement_performed = True

    def visualize_data(self, save_figure=False):
        """
        Plots the data in a useful form for the measurement type.
        Requirement: The function perform_measurement must have been run before running this function.

        :return: n/a
        """
        self._check_measurement_performed()
        self._visualize_data(save_figure)

    def save_data(self):
        """
        Saves the data in a useful form.
        Requirement: The function perform_measurement must have been run before running this function.

        :return: n/a
        """
        self._check_measurement_performed()
        self._save_data()

    def get_data(self):

        self._check_measurement_performed()
        self._get_data()