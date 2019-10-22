# DAQ-and-laser-API
For using the NI-DAQmx library and the DAQ cards with the TSL-550 laser for data acquisition

<b><u>Branches</b></u>

master - branch used for tested code that works 100%

main development branch - used to collect the changes from individual branches and integration test

Ben - mostly working on NI DAQ box


<b>Main development branch files:</b>

TSL550.py - Contains the interface functions for the TSL-550 laser

DAQinterface.py - Containst the NIDAQInterface class intended to be used with the NI DAQ cards.

measure_single_output.py - script for measuring the output from a single port

measure_device_multiple_outputs.py - script for measuring the output from multiple ports

Measurement.py - will contain different measurement processes and sweeps eventually

Dependencies: nidaqmx
