# Setup
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import argparse
import numpy as np
import matplotlib.pyplot as plt
import copy
import csaps

# Settings
smoothing = 1
height = 0.3
distance = 10
width = 3
NUM_PORTS = 4
middlefreq = 187.75


# Helper Functions
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def cos2(frequency, a, phi):
    return pow(np.cos(a * frequency + phi), 2)


def cos_squared(f, a, b, c, phi):
    return pow(a * np.cos(b * f + phi), 2) + c


def J(frequency, power, x):
    return np.sum(np.abs(cos2(frequency, x[0], x[1]) - power) ** 2)


def freq(wavelength):
    c = 299792458
    return c / wavelength


global COUNT
COUNT = 0


# Object class for finding crossover frequency
class DeviceResult:
    POLY_ORDER = 10
    PEAKFINDING_POLY = 10

    def __init__(self, filename, wavelength=None, power=None):
        self.local_result = None
        self.filename = filename
        self.wavelength = np.squeeze(wavelength)
        self.power = power

    def phase(self, frequency, port):
        phase = np.array(2 * self.local_result[port].x[0] * frequency + 2 * self.local_result[port].x[1])
        if phase.size > 1:
            phase = np.unwrap(np.angle(np.exp(1j * phase))) + np.pi
        else:
            phase = (np.angle(np.exp(1j * phase))) + np.pi
        return phase

    def plotPowerLinearNorm(self):
        plt.plot(self.wavelength, self.powerLinearNorm)

    def plotPowerLinearNormByFreq(self, port):
        plt.plot(self.frequency, self.powerLinearNormByFreq[:, port])

    def plotPowerHighSampNormByFreq(self, port):
        plt.plot(self.freqHighSamp, self.powerHighSampNorm[:, port])

    def plotGlobalOptimizationCurve(self, port):
        plt.plot(self.freqHighSamp,
                 cos2(self.freqHighSamp, self.global_result[port].x[0], self.global_result[port].x[1]))

    def plotLocalOptimizationCurve(self, port):
        plt.plot(self.freqHighSamp, cos2(self.freqHighSamp, self.local_result[port].x[0], self.local_result[port].x[1]))

    def plotPhase(self, port=None):
        if port is not None:
            plt.plot(self.frequency, self.phase(self.frequency, port), label='Port {}'.format(port))
        else:
            for i in range(NUM_PORTS):
                self.plotPhase(port=i)

    def plotPolarPhaseAtFreq(self, frequency, ax, normalize=False):
        phase0 = self.phase(frequency, 0)
        if not normalize:
            phi0 = np.exp(1j * phase0)
        else:
            phi0 = np.exp(1j * 0)
        plt.plot(np.real(phi0), np.imag(phi0), 'x', label='Port 0', markersize=10.0, markeredgewidth=3.0)
        for i in range(1, NUM_PORTS):
            if not normalize:
                phasei = self.phase(frequency, i)
            else:
                phasei = self.phase(frequency, i) - phase0
            phii = np.exp(1j * phasei)
            plt.plot(np.real(phii), np.imag(phii), 'x', label=('Port ' + str(i)), markersize=10.0, markeredgewidth=3.0)

        ax.add_patch(plt.Circle((0, 0), radius=1, edgecolor='0.0', facecolor='None', ls='--'))
        ax.set_aspect('equal', 'box')
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
        plt.grid(True)

    def plotPolarRange(self):
        frequency = 190
        phase0l = self.phase(frequency, 0)
        phase1l = self.phase(frequency, 1) - phase0l
        phase2l = self.phase(frequency, 2) - phase0l
        phase0l = phase0l - phase0l

        frequency = 197
        phase0h = self.phase(frequency, 0)
        phase1h = self.phase(frequency, 1) - phase0h
        phase2h = self.phase(frequency, 2) - phase0h
        phase0h = phase0h - phase0h

        phase0l = (phase0l + 2 * np.pi) % (2 * np.pi)
        phase1l = (phase1l + 2 * np.pi) % (2 * np.pi)
        phase2l = (phase2l + 2 * np.pi) % (2 * np.pi)
        phase0h = (phase0h + 2 * np.pi) % (2 * np.pi)
        phase1h = (phase1h + 2 * np.pi) % (2 * np.pi)
        phase2h = (phase2h + 2 * np.pi) % (2 * np.pi)

        phi0 = np.linspace(phase0l, phase0h)
        if phi0[0] > phi0[-1]:
            phi0 = np.flip(phi0)
        phi1 = np.linspace(phase1l, phase1h)
        if phi1[0] > phi1[-1]:
            phi1 = np.flip(phi1)
        phi2 = np.linspace(phase2l, phase2h)
        if phi2[0] > phi2[-1]:
            phi2 = np.flip(phi2)

        factor = 0.03
        global COUNT
        loc = 1 + factor * COUNT
        vals = np.linspace(loc, loc)
        COUNT += 1

        plt.polar(phi0[0], loc, 'rx')
        plt.polar(phi0[-1], loc, 'rx')
        plt.polar(phi0, vals)
        plt.polar(phi1[0], loc, 'bx')
        plt.polar(phi1[-1], loc, 'bx')
        plt.polar(phi1, vals)
        plt.polar(phi2[0], loc, 'kx')
        plt.polar(phi2[-1], loc, 'kx')
        plt.polar(phi2, vals)

        plt.grid(True)
        plt.show()
        return phi0, phi1, phi2


def find_phase_offset(file_names: list, TEST=False):

    wavelength = np.array([])
    power_data = np.zeros((5001, 4))

    for i in range(len(file_names)):
        file_data = np.load(file_names[i], mmap_mode='r')

        wavelength = np.array(file_data['wavelength']) * 1e-9
        power_data[:, i] = 20 * np.log10(np.array(file_data['power']).clip(min=0.0000001))

    device_results = DeviceResult('Device_21', wavelength, power_data)
    power_norm = copy.deepcopy(device_results.power)
    plt.figure()
    plt.suptitle('Device ' + device_results.filename)
    for i in range(0, NUM_PORTS):
        plt.subplot(NUM_PORTS, 1, i + 1)
        amplitude = device_results.power[:, i]
        plt.plot(device_results.wavelength * 1e9, amplitude, label='Measured')
        p = np.polyfit(device_results.wavelength - np.mean(device_results.wavelength), amplitude, device_results.POLY_ORDER)
        amplitude_baseline = np.polyval(p, device_results.wavelength - np.mean(device_results.wavelength))
        amplitude_corrected = amplitude - amplitude_baseline
        amplitude_corrected = amplitude_corrected + max(amplitude_baseline) - max(amplitude)
        plt.plot(device_results.wavelength * 1e9, amplitude_corrected, label='GC removed')
        plt.legend()
        plt.xlim((1560, 1620))
        power_norm[:, i] = amplitude_corrected
    if TEST:
        plt.show()

    powers = 10 ** (power_norm / 20)
    plt.figure()
    plt.suptitle('Device ' + device_results.filename)
    device_results.powerLinearNorm = copy.deepcopy(powers)
    for i in range(0, NUM_PORTS):
        x = device_results.wavelength

        # power_negative = -one.powerLinearNorm[:, i]
        # offset = -np.amin(power_negative)
        # power_negative += offset
        # lower_peaks = find_peaks(power_negative, height=height, distance=distance, width=width)[0]
        # lower_fit = interp1d(x[lower_peaks], power_negative[lower_peaks], kind='cubic', fill_value='extrapolate')
        # bottom_baseline = lower_fit(x)
        # power_negative /= bottom_baseline
        # power = np.array(-(power_negative - 1))

        power = device_results.powerLinearNorm[:, i]
        top_pkidx = find_peaks(power, height=height, distance=distance, width=width)[0]
        p = interp1d(x[top_pkidx], power[top_pkidx], kind='cubic', fill_value='extrapolate')
        top_baseline = p(x)

        plt.subplot(NUM_PORTS, 1, i + 1)
        plt.title('Port ' + str(i + 1))
        plt.plot(x[top_pkidx] * 1e9, power[top_pkidx], "x")
        plt.plot(x * 1e9, power)
        plt.plot(x * 1e9, top_baseline)
        plt.xlim((1560, 1620))
        device_results.powerLinearNorm[:, i] = power / top_baseline
    if TEST:
        plt.show()

    # print("Device \'" + one.filename + "\': Converting from wavelength to frequency")
    # Slice off the bottom of the array which has wild tails due to spline fitting
    device_results.frequency = np.flip(freq(device_results.wavelength) / 1e12)
    device_results.powerLinearNormByFreq = np.flipud(device_results.powerLinearNorm)
    print(device_results.frequency.shape, device_results.powerLinearNormByFreq.shape)
    device_results.frequency = device_results.frequency[50:]
    device_results.powerLinearNormByFreq = device_results.powerLinearNormByFreq[50:, :]
    print(device_results.frequency.shape, device_results.powerLinearNormByFreq.shape)
    # print("\n")
    # print("Data Preview:")
    # print("=============")
    # print("Frequency array:", one.frequency)
    # print("Power array:", one.powerLinearNormByFreq)
    plt.figure()
    for i in range(0, NUM_PORTS):
        plt.subplot(NUM_PORTS, 1, i + 1)
        plt.title('Port ' + str(i + 1))
        plt.plot(device_results.frequency, device_results.powerLinearNormByFreq[:, i])
    if TEST:
        plt.show()

    # print("Smoothing:", smoothing)
    device_results.sp = []
    device_results.sp.append(
        csaps.UnivariateCubicSmoothingSpline(device_results.frequency, device_results.powerLinearNormByFreq[:, 0], smooth=smoothing))
    device_results.sp.append(
        csaps.UnivariateCubicSmoothingSpline(device_results.frequency, device_results.powerLinearNormByFreq[:, 1], smooth=smoothing))
    device_results.sp.append(
        csaps.UnivariateCubicSmoothingSpline(device_results.frequency, device_results.powerLinearNormByFreq[:, 2], smooth=smoothing))
    device_results.sp.append(
        csaps.UnivariateCubicSmoothingSpline(device_results.frequency, device_results.powerLinearNormByFreq[:, 3], smooth=smoothing))

    N = 10000
    device_results.freqHighSamp = np.linspace(min(device_results.frequency), max(device_results.frequency), num=N)
    powerHighSamp0 = device_results.sp[0](device_results.freqHighSamp)
    powerHighSamp0 = powerHighSamp0 / max(powerHighSamp0)
    powerHighSamp1 = device_results.sp[1](device_results.freqHighSamp)
    powerHighSamp1 = powerHighSamp1 / max(powerHighSamp1)
    powerHighSamp2 = device_results.sp[2](device_results.freqHighSamp)
    powerHighSamp2 = powerHighSamp2 / max(powerHighSamp2)
    powerHighSamp3 = device_results.sp[3](device_results.freqHighSamp)
    powerHighSamp3 = powerHighSamp3 / max(powerHighSamp3)
    device_results.powerHighSampNorm = np.stack((powerHighSamp0, powerHighSamp1, powerHighSamp2, powerHighSamp3), axis=1)

    plt.figure()
    plt.suptitle(device_results.filename)
    for i in range(0, NUM_PORTS):
        plt.subplot(NUM_PORTS, 1, i + 1)
        plt.title('Port ' + str(i + 1))
        plt.plot(device_results.frequency, device_results.powerLinearNormByFreq[:, i], label='Normalized')
        plt.plot(device_results.freqHighSamp, device_results.powerHighSampNorm[:, i], label='Fit to Spline')
        plt.legend()
    if TEST:
        plt.show()

    cosine_fit, cosine_covariance = curve_fit(cos_squared, device_results.freqHighSamp, device_results.powerHighSampNorm[:, i])
    plt.plot(device_results.freqHighSamp, device_results.powerHighSampNorm[:, i])
    plt.plot(device_results.freqHighSamp, cos_squared(device_results.freqHighSamp, *cosine_fit))
    if TEST:
        plt.show()

    # print("====================")
    # print(one.filename)
    device_results.global_result = [None] * NUM_PORTS
    device_results.local_result = [None] * NUM_PORTS
    for port in range(NUM_PORTS):
        Jmod = lambda x: J(device_results.freqHighSamp, device_results.powerHighSampNorm[:, port], x)
        bounds = [(0, 40), (0, 2 * np.pi)]
        # Global optimization
        device_results.global_result[port] = differential_evolution(Jmod, bounds)
        # Convex optimization
        device_results.local_result[port] = minimize(Jmod, device_results.global_result[port].x, method='nelder-mead')
        # Verbose
        # print('Port ' + str(port+1))
        # print('Global:', one.global_result[port].x, one.global_result[port].fun)
        # print('Local:', one.local_result[port].x, one.local_result[port].fun)

    plt.figure()
    plt.suptitle(device_results.filename)

    for port in range(NUM_PORTS):
        plt.subplot(NUM_PORTS, 1, port + 1)
        device_results.plotPowerLinearNormByFreq(port)
        device_results.plotPowerHighSampNormByFreq(port)
        device_results.plotGlobalOptimizationCurve(port)
        device_results.plotLocalOptimizationCurve(port)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Amplitude (arbitrary units)")
        plt.xlim((freq(1620 * 1e-9) * 1e-12, freq(1560 * 1e-9) * 1e-12))
        plt.title("Power")
    if TEST:
        plt.show()

    phasediff = device_results.phase(device_results.frequency, 0) - device_results.phase(device_results.frequency, 2)
    middleindex = min(range(len(phasediff)), key=lambda i: abs(phasediff[i] - (np.pi / 2)))
    print("CROSSOVER FREQUENCY: ", device_results.frequency[middleindex], "THz")
    c0 = 299792458  # m/s
    crossover_wavelength = c0 / (device_results.frequency[middleindex] * 1e12)
    print("CROSSOVER WAVELENGTH:", crossover_wavelength * 1e9, "nm")

    middlefreq = device_results.frequency[middleindex]

    freq_array = [min(device_results.frequency), middlefreq, max(device_results.frequency)]
    fig = plt.figure(constrained_layout=True, figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
    gs = fig.add_gridspec(3, 3)
    ax = fig.add_subplot(gs[0, :])

    for i in range(NUM_PORTS):
        device_results.plotPowerLinearNormByFreq(i)

    for i in range(3):
        plt.axvline(freq_array[i], ls='--', color='0.0')
        plt.text(freq_array[i] - 0.02, -0.20, chr(i + 97) + ".")
    plt.ylabel("Power (a.u.)")
    plt.grid(True)

    ax = fig.add_subplot(gs[1, :])
    device_results.plotPhase()
    for i in range(3):
        plt.axvline(freq_array[i], ls='--', color='0.0')
    plt.ylabel("Phase (rad)")
    plt.xlabel("Frequency (THz)")
    plt.grid(True)

    mini = fig.add_subplot(gs[2, 0])
    device_results.plotPolarPhaseAtFreq(freq_array[0], mini, True)
    plt.title("a.")

    mini = fig.add_subplot(gs[2, 1])
    device_results.plotPolarPhaseAtFreq(freq_array[1], mini, True)
    plt.title("b.")

    mini = fig.add_subplot(gs[2, 2])
    device_results.plotPolarPhaseAtFreq(freq_array[2], mini, True)
    plt.title("c.")
    plt.show()


if __name__ == "__main__":
    
    file_names = ["X1_data.npz", "X2_data.npz", "P1_data.npz", "P2_data.npz"]
    # Read in the data
    for i in range(len(file_names)):
        file_names[i] = "C:\\Users\\camacho\\Desktop\\Data_Acquisition\\Measurement_Data" \
                "\\D21_00\\offset_finding\\" + file_names[i]
    find_phase_offset(file_names, True)
