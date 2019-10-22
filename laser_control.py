#Script for initializing laser and checking values on laser usage settings.

# ---------------------------------------------------------------------------- #
# Dependencies
# ---------------------------------------------------------------------------- #
from TSL550 import TSL550
import time
# ---------------------------------------------------------------------------- #
# Reference Data
# ---------------------------------------------------------------------------- #
class limits:
    """Constants for the laser settings."""
    class sweep_rate:
        type = "laser sweep"
        units = "nm/s"
        min = 1.0
        max = 100
    class wavelength:
        type = "wavelength"
        units = "nm"
        min = 1550
        max = 1630


# ---------------------------------------------------------------------------- #
# Functions
# ---------------------------------------------------------------------------- #
def initLaser(address="COM4"):
    return TSL550.TSL550(address)

#Generic function to check if a value is in range.
#Uses ranges from limits class.
def checkRange(value, range):
    if not (range.min <= value <= range.max):
        error_msg = "Invalid {0} of {1} {4}. Must be between {2} and {3} {4}."
        error_msg = error_msg.format(
            range.type,
            value,
            range.min,
            range.max,
            range.units
        )
        raise AttributeError(error_msg)

def checkSweepRate(sweep_rate):
    checkRange(sweep_rate, limits.sweep_rate)

def checkWavelength(wavelength):
    checkRange(wavelength, limits.wavelength)

# ---------------------------------------------------------------------------- #
# Unit Testing
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    print('-' * 20)
    print("Testing sweep rate limits.")
    print('-' * 20)
    for i in range(0, 105):
        try:
            checkSweepRate(i)
        except Exception as err:
            print(err)

    print('-' * 20)
    print("Testing wavelength limits.")
    print('-' * 20)
    for i in range(1545, 1635):
        try:
            checkWavelength(i)
        except Exception as err:
            print(err)
