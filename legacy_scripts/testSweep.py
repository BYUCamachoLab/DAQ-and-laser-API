from DAQinterface import NIDAQInterface
from Laser import TSL550
import serial.tools.list_ports
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal


def runSweep(laser, task, lambdaStart=1530, lambdaEnd=1580, numLambda=1000, time=10e-3):

    numChannels = len(task.ai_channels)
    wavelengthPoints = np.linspace(lambdaStart, lambdaEnd, numLambda)
    muData = np.zeros((numLambda,numChannels))
    sigmaData = np.zeros((numLambda,numChannels))
    for k in range(numLambda):
        # Set the wavelength point
        changeWavelength = laser.wavelength(wavelengthPoints[k])

        # Pull data from the daq for *time* seconds
        numSamples = int(task.timing.samp_clk_rate * float(time))
        data = task.read(number_of_samples_per_channel=numSamples)

        # Pull the experiment's statistics
        muData[k, :]    = np.mean(data)
        sigmaData[k, :] = np.std(data)
    return muData, sigmaData, wavelengthPoints


if __name__ == "__main__":
    # ------------------------------------------------------------------------ #
    # Sweep parameters
    # ------------------------------------------------------------------------ #
    lambda_start = 1554
    lambda_stop  = 1556
    duration     = 4
    trigger_step = 0.0001
    sample_rate  = 100e3

    # ------------------------------------------------------------------------ #
    # Check input
    # ------------------------------------------------------------------------ #

    laser_sweep_rate = (lambda_stop - lambda_start) / duration
    if laser_sweep_rate > 100 or laser_sweep_rate < 0.5:
        raise AttributeError("Invalid laser sweep speed of %f. Must be between 0.5 and 100 nm/s." % laser_sweep_rate)

    # ------------------------------------------------------------------------ #
    # Setup devices
    # ------------------------------------------------------------------------ #

    # Initialize laser
    laser = TSL550()
    laser.on()
    laser.openShutter()
    laser.sweep_set_mode(continuous=True, twoway=True, trigger=False, const_freq_step=False)
    laser.trigger_enable_output()
    print(laser.trigger_set_mode("Step"))
    print(laser.trigger_set_step(trigger_step))

    # Get number of samples to record. Add buffer just in case.
    numSamples= int(duration * 2 * sample_rate)
    print(numSamples)

    # Initialize DAQ
    daq = NIDAQInterface()
    task = daq.initialize(channels=["ai0", "ai1"], sample_rate=sample_rate, samples_per_chan=numSamples)

    # ------------------------------------------------------------------------ #
    # Run sweep
    # ------------------------------------------------------------------------ #

    laser.sweep_wavelength(start=lambda_start,stop=lambda_stop,duration=duration,number=1)
    data = np.array(task.read(number_of_samples_per_channel=numSamples,timeout=3*duration))

    wavelength_logging = laser.wavelength_logging()

    # ------------------------------------------------------------------------ #
    # Process data
    # ------------------------------------------------------------------------ #

    pulse_width_time = 27e-6
    pulse_width_samples = int(sample_rate * pulse_width_time) + 1
    from scipy.signal import find_peaks
    from scipy.stats import linregress
    peaks, _ = find_peaks(data[1,:], height=3, distance=5)

    print('=============')
    print(laser.wavelength_logging_number())
    print(len(peaks))
    print('=============')

    modPeaks = peaks - peaks[0]
    modTime = modPeaks / sample_rate
    slope, intercept, r_value, p_value, std_err = linregress(modTime, wavelength_logging)

    print((data[1,:] > 3.25 ).sum())
    plt.figure()
    plt.plot(modTime,wavelength_logging,'.')
    plt.xlabel('Time (s)')
    plt.ylabel('Wavelength (nm)')
    plt.grid(True)
    plt.title("r-squared: %f" % r_value**2)
    plt.savefig('benchmark_2nm_4s.png')
    #plt.plot(data[1,:])
    #plt.plot(peaks, data[1,peaks], "x")
    plt.show()

    # ------------------------------------------------------------------------ #
    # Device cleanup
    # ------------------------------------------------------------------------ #
    closeTask(task)
