from Measurement.SweepMeasurement import SweepMeasurement
from Laser.TSL550 import TSL550
from NIDAQ.NIDAQ import NIDAQ
import time
import numpy as np
from scipy.signal import find_peaks


class NIDAQSweepMeasurement(SweepMeasurement):

    """
    Concrete sweep measurement class using the NIDAQ and the TSL 550 laser.
    """

    def __init__(self, lambda_start: float, lambda_end: float, dur: float, trig_step: float, samp_rate: float,
                 power: float, folder, chan_list: list, dev_type: str, desc: str, ports: list):

        super().__init__(lambda_start, lambda_end, dur, trig_step, samp_rate,
                         power, folder, chan_list, dev_type, desc, ports)
        self.daq = NIDAQ()
        self.laser = None

    def _initialize_daq(self):
        """
        Initializes the NIDAQ.
        :return: n/a
        """
        # Get number of samples to record.
        num_samples = int(self.duration * self.sample_rate * 1.5)
        time.sleep(0.3)

        # Initialize DAQ
        self.daq.initialize(["cDAQ1Mod1/ai0"],
                            sample_rate=self.sample_rate, samples_per_chan=num_samples)
        for i in range(len(self.output_ports)):
            self.daq.add_channel(self.output_channel_list[i])

    def _initialize_laser(self):
        """
        Initializes the TSL 550 laser.
        :return: n/a
        """
        # Check laser's sweep rate
        laser_sweep_rate = (self.wavelength_endpoint - self.wavelength_startpoint) / self.duration
        if laser_sweep_rate > 100 or laser_sweep_rate < 0.5:
            raise AttributeError(
                "Invalid laser sweep speed of %f. Must be between 0.5 and 100 nm/s." % laser_sweep_rate)

        # Initialize laser
        print("Opening connection to laser...")
        print("Laser response: ")
        self.laser = TSL550(TSL550.LASER_PORT)
        self.laser.on()
        self.laser.power_dBm(self.power_dBm)
        self.laser.openShutter()
        self.laser.sweep_set_mode(continuous=True, twoway=True, trigger=False, const_freq_step=False)
        self.laser.trigger_enable_output()
        print("Mode:", self.laser.trigger_set_mode("Step"))
        print("Step size: dlambda = ", self.laser.trigger_set_step(self.trigger_step))

    def _initialize_devices(self):
        self._initialize_laser()
        self._initialize_daq()

    def _laser_sweep_start(self):
        """
        Starts the laser sweep.
        :return: n/a
        """
        self.laser.sweep_wavelength(start=self.wavelength_startpoint, stop=self.wavelength_endpoint,
                                    duration=self.duration, number=1)

    def _read_data(self):
        self.data = np.array(self.daq.read(self.duration * 1.5))
        self.times_read = np.arange(0, self.duration * 1.5, 1. / self.sample_rate)
        self.wavelength_logging = np.array(self.laser.wavelength_logging())
        self.daq.close()

    def _interpolate_data(self):
        """
        The NIDAQ reads data at even time increments, but the laser's wavelength sweep is not constant with time.
        The wavelength points reach even points when the laser's trigger fires.
        This function selects the points at the times the trigger fired, and interpolates between them to get
        a function from wavelength -> voltage, instead of time -> voltage.

        :return: The wavelength vs. voltage measurement points.
        """
        peaks, _ = find_peaks(self.data[0, :], height=3, distance=5)
        device_data = self.data[1:, peaks]
        device_times = self.times_read[peaks]

        peak_spacing = peaks - peaks[0]
        time_between_peaks = peak_spacing / self.sample_rate
        time_wavelength_conversion_fit = np.polyfit(time_between_peaks, self.wavelength_logging, 2)
        conversion_fit_function = np.poly1d(time_wavelength_conversion_fit)
        lined_up_wavelength_points = np.array(conversion_fit_function(device_times))

        return np.concatenate((lined_up_wavelength_points[np.newaxis, :], device_data), axis=0)
