# Data collection script utilizing oscilloscope or NIDAQ.
# ---------------------------------------------------------------------------- #
# Parameters
# ---------------------------------------------------------------------------- #
#Laser Sweep
wavelength_start    = 1550
wavelength_stop     = 1630
duration            = 5
trigger_step        = 0.01
power_dBm           = 7

# Measurement
device              = "SCOPE" # Currently using 'DAQ' or 'SCOPE'

# Saving Data
save_directory      = "measurements/"
use_sub_dir         = True # Use first command line argument as sub-directory.
force_sub_dir       = False # Require a sub-directory name to run.

# ---------------------------------------------------------------------------- #
# Reference
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# Import Libraries
# ---------------------------------------------------------------------------- #
# General
import sys                  # For getting command-line arguments.
import os                   # For making directories.
from pathlib import Path    # Also for directories.
from datetime import date   # For date stamp.
# Laser
from TSL550.TSL550 import TSL550
import serial.tools.list_ports
# Measurement
if device == "DAQ":
    from DAQinterface import NIDAQInterface
elif device == "SCOPE":
    from RTO.controller import RTO as connectScope
# Data Processing
from data_processing import WavelengthAnalyzer, VisualizeData

# ---------------------------------------------------------------------------- #
# Check Input
# ---------------------------------------------------------------------------- #
print("Checking inputs.")

# Get command line arguments.
args = sys.argv[1:] # First argument is this script. Discard it.

try:
    subDirName = args[0] # Extract first argument as subdirectory name.
except:
    subDirName = ""

if force_sub_dir:
    assert subDirName is not "",\
        "Save directory name required as first argment."

# Check laser settings.
laserSweepRate = (wavelength_stop - wavelength_start) / duration
# Todo: Check sweep rate against valid range.
assert \
    TSL550.MINIMUM_WAVELENGTH <= wavelength_start <= TSL550.MAXIMUM_WAVELENGTH,\
        "Starting wavelength out of range."
assert \
    TSL550.MINIMUM_WAVELENGTH <= wavelength_stop <= TSL550.MAXIMUM_WAVELENGTH,\
        "Ending wavelength out of range."

# ---------------------------------------------------------------------------- #
# Initialize Save Directory
# ---------------------------------------------------------------------------- #
today = date.today()
dateStamp = "{}_{}_{}_".format(today.year, today.month, today.day)
mainDirName = dateStamp + save_directory
mainDir = Path(Path.cwd(), mainDirName)
if not os.path.exists(mainDir):
    print("Creating " + mainDirName)
    os.makedirs(mainDir)
subDir = Path(mainDir, subDirName)
if not os.path.exists(subDir):
    print("Creating " + subDirName)
    os.makedirs(subDir)

# Todo: Ask for confirmation before overwriting existing data.
# ---------------------------------------------------------------------------- #
# Initialize Devices
# ---------------------------------------------------------------------------- #
print ("Initializing devices.")

#Laser
print("Connecting to laser...")
laser = TSL550(TSL550.LASER_PORT)
# ---------------------------------------------------------------------------- #
# Recheck Settings (from Device)
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# Collect Data
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# Process Data
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# Draw Visuals
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# Cleanup
# ---------------------------------------------------------------------------- #
