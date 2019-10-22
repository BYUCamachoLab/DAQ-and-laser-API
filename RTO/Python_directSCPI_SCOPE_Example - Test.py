# RTO/RTE Oscilloscope example for Python using PyVISA

acquisition_time = 10 #in seconds
timeout_buffer = acquisition_time * 2 #in seconds

import visa
import VISAresourceExtentions

# -----------------------------------------------------------
# Initialization
# -----------------------------------------------------------
rm = visa.ResourceManager()
scope = rm.open_resource('TCPIP::10.32.112.140::INSTR')

# The file VISAresourceExtentions.py must be in the same folder as this file
# It contains PyVISA Visa.Resource extension functions for and 2 new exception types

# try block to catch any InstrumentErrorException()
try:
    scope.write_termination = ''
    # Some instruments require LF at the end of each command. In that case, use:
    # scope.write_termination = '\n'
    scope.ext_clear_status()  # Clear instrument io buffers and status
    print(scope.query('*IDN?'))  # Query the Identification string
    scope.write('*RST;*CLS')  # Reset the instrument, clear the Error queue
    scope.write('SYST:DISP:UPD ON')  # Display update ON - switch OFF after debugging
    scope.ext_error_checking()  # Error Checking after Initialization block

    # -----------------------------------------------------------
    # Basic Settings:
    # -----------------------------------------------------------
    #Timebase & Acquisition Settings
    #scope.write('ACQ:POIN:AUTO RECL')  # Define Horizontal scale by number of points
    #scope.write('ACQ:POIN 4008')  # 4008 X points

    scope.write('ACQ:POIN:AUTO RES')
    scope.write('ACQ:RES 100e-12')

    scope.write('TIM:RANG {}'.format(acquisition_time))  # Set Acquisition time

    #Channel 1 Settings
    scope.write('CHAN1:RANG 4')  # Horizontal range 4V
    scope.write('CHAN1:POS 0')  # Offset 0
    scope.write('CHAN1:COUP DCL')  # Coupling DC 1MOhm
    scope.write('CHAN1:STAT ON')  # Switch Channel 1 ON

    #Channel 2 Settings
    scope.write('CHAN2:RANG 50')  # Horizontal range 50V
    scope.write('CHAN2:POS 0')  # Offset 0
    scope.write('CHAN2:COUP DCL')  # Coupling DC 1MOhm
    scope.write('CHAN2:STAT ON')  # Switch Channel 2 ON

    scope.ext_error_checking()  # Error Checking

    # -----------------------------------------------------------
    # Trigger Settings:
    # -----------------------------------------------------------
    scope.write('TRIG1:MODE AUTO')  # Trigger Auto mode in case of no signal is applied
    scope.write('TRIG1:SOUR CHAN2')  # Trigger source CH2
    scope.write('TRIG1:TYPE EDGE;:TRIG1:EDGE:SLOP POS')  # Trigger type Edge Positive
    scope.write('TRIG1:LEV1 2')  # Trigger level 2V
    scope.query('*OPC?')  # Using *OPC? query waits until all the instrument settings are finished
    scope.ext_error_checking()  # Error Checking

    # -----------------------------------------------------------
    # SyncPoint 'SettingsApplied' - all the settings were applied
    # -----------------------------------------------------------
    # Arming the scope
    # -----------------------------------------------------------
    scope.timeout = (acquisition_time + timeout_buffer)*1000  # Acquisition timeout in milliseconds - set it higher than the acquisition time
    scope.write('SING')
    # -----------------------------------------------------------
    # DUT_Generate_Signal() - in our case we use Probe compensation signal
    # where the trigger event (positive edge) is reoccurring
    # -----------------------------------------------------------
    scope.query('*OPC?')  # Using *OPC? query waits until the instrument finished the Acquisition
    scope.ext_error_checking()  # Error Checking
    # -----------------------------------------------------------
    # SyncPoint 'AcquisitionFinished' - the results are ready
    # -----------------------------------------------------------
    # Fetching the waveform in ASCII format
    # -----------------------------------------------------------
    print('Fetching waveform in ASCII format... ')
    waveformASCII = scope.query_ascii_values('FORM ASC;:CHAN1:DATA?')
    print('ASCII data samples read: {}'.format(len(waveformASCII)))

    print('Writing data samples to file.')
    with open(r'C:\Temp\Data.txt', 'w') as dataOut:
        for sample in waveformASCII:
            dataOut.write(str(sample) + '\n')

    scope.ext_error_checking()  # Error Checking after the data transfer
    # -----------------------------------------------------------
    # Fetching the trace in Binary format
    # Transfer of traces in binary format is faster.
    # The waveformBIN data and waveformASC data are however the same.
    # -----------------------------------------------------------
    print('Fetching waveform in binary format... ')
    waveformBIN = scope.query_binary_values('FORM REAL;:CHAN1:DATA?')
    print('Binary data samples read: {}'.format(len(waveformBIN)))
    scope.ext_error_checking()  # Error Checking after the data transfer
    # -----------------------------------------------------------
    # Making an instrument screenshot and transferring the file to the PC
    # -----------------------------------------------------------
    print('Taking instrument screenshot and saving it to the PC... ')
    scope.write('HCOP:DEV:LANG PNG')
    scope.write('MMEM:NAME \'c:\\temp\\Dev_Screenshot.png\'')
    scope.write('HCOP:IMM')  # Make the screenshot now
    scope.query('*OPC?')  # Wait for the screenshot to be saved
    scope.ext_error_checking()  # Error Checking after the screenshot creation
    scope.ext_query_bin_data_to_file('MMEM:DATA? \'c:\\temp\\Dev_Screenshot.png\'', r'c:\Temp\PC_Screenshot.png')
    print('saved to PC c:\\Temp\\PC_Screenshot.png\n')
    scope.ext_error_checking()  # Error Checking at the end of the program

except VISAresourceExtentions.InstrumentErrorException as e:
    # Catching instrument error exception and showing its content
    print('Instrument error(s) occurred:\n' + e.message)
