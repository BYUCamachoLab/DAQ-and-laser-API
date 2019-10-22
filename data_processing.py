import numpy as np
import scipy.io as sio
from scipy.signal import find_peaks
from scipy.stats import linregress
from matplotlib import pyplot as graph

class WavelengthAnalyzer:
    """Simple class that uses a trigger signal
    to convert datasets from time-domain to wavelength."""
    def __init__(self, sample_rate, trigger_data, wavelength_log):
        self.sample_rate = sample_rate
        self.wavelength_log = wavelength_log
        self.__analyze_trigger(trigger_data)

    def __analyze_trigger(self, trigger_data):
        self.peaks, _ = find_peaks(trigger_data, height = 3, distance = 5)

    def num_peaks(self):
        return len(self.peaks)

    def process_data(self, raw_data):
        deviceData = raw_data[self.peaks[0]:self.peaks[-1]]
        #Time relative to the first collected data point.
        deviceTime = np.arange(len(deviceData)) / self.sample_rate

        modPeaks = self.peaks - self.peaks[0] #Make relative to the first point.
        modTime = modPeaks / self.sample_rate
        fit = np.polyfit(modTime, self.wavelength_log, 2) #Least-squares fit.
        mapping = np.poly1d(fit) #Create function mapping time to wavelength.
        deviceWavelength = mapping(deviceTime) #Get wavelengths at given times.
        channelData = {
            "wavelengths": deviceWavelength,
            "data": deviceData,
            "wavelengthHash": hash(tuple(deviceWavelength)),
            "dataHash": hash(tuple(deviceData))
        }
        return channelData

def VisualizeData(
        save_path,
        channel,
        wavelengths,
        data,
        wavelengthHash,
        dataHash
    ):
    graph.figure(channel)
    graph.plot(wavelengths, data)
    graph.xlabel("Wavelength (nm)")
    graph.ylabel("Power (au)")
    graph.grid(True)
    graph.tight_layout()
    graph.savefig(save_path + "_Channel{}{}".format(channel, ".png"))
    sio.savemat(
        save_path + "_Channel{}{}".format(channel, ".mat"),
        {
            'wavelength': wavelengthHash,
            'power': dataHash
        }
    )
    np.savez(
        save_path + "_Channel{}{}".format(channel, ".npz"),
        wavelength = np.squeeze(wavelengths),
        power = np.squeeze(data)
    )
    graph.show()
