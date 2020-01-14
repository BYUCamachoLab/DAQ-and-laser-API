
from NIDAQ.NIDAQinterface import NIDAQInterface
from NIDAQ.NIDAQSweepMeasurement import NIDAQSweepMeasurement
from PhaseOffsetFinder import find_phase_offset

# Set sweep parameters
wavelength_startpoint = 1560
wavelength_endpoint = 1620
duration = 5
trigger_step = 0.01
sample_rate = NIDAQInterface.CARD_TWO_MAX_SAMPLE_RATE
power_dBm = 10
measurement_folder = "Measurement_Data/"
# For offset finding
output_channel_list = ["cDAQ1Mod1/ai1", "cDAQ1Mod1/ai2", "cDAQ1Mod1/ai3", "cDAQ1Mod2/ai0"]
# For S-parameter measurements
#output_channel_list = ["cDAQ1Mod1/ai1", "cDAQ1Mod1/ai2", "cDAQ1Mod1/ai3", "cDAQ1Mod2/ai0", "cDAQ1Mod2/ai1"]
device_type = "test device"
description = "test"
output_ports = ["test1", "test2", "test3", "test4"]

# Run the measurement
test_measurement = NIDAQSweepMeasurement(wavelength_startpoint,
                                         wavelength_endpoint,
                                         duration,
                                         trigger_step,
                                         sample_rate,
                                         power_dBm,
                                         measurement_folder,
                                         output_channel_list,
                                         device_type,
                                         description,
                                         output_ports)
test_measurement.perform_measurement()
test_measurement.visualize_data(save_figure=True)
test_measurement.save_data()

print(find_phase_offset(test_measurement.get_data()))
