
from NIDAQ.NIDAQ import NIDAQ
from NIDAQ.NIDAQSweepMeasurement import NIDAQSweepMeasurement
from PhaseOffsetFinder import find_phase_offset

# Set sweep parameters
wavelength_startpoint = 1560
wavelength_endpoint = 1620
duration = 5
trigger_step = 0.01
sample_rate = NIDAQ.CARD_TWO_MAX_SAMPLE_RATE
power_dBm = 10
measurement_folder = "Measurement_Data/"

output_channel_list = ["cDAQ1Mod1/ai1", "cDAQ1Mod1/ai2", "cDAQ1Mod1/ai3", "cDAQ1Mod2/ai0"]

device_type = "D21_00"
description = "offset_finding"
output_ports = ["X1", "X2", "P1", "P2"]

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
test_measurement.visualize_data(save_figure=True, show_figure=True)
test_measurement.save_data()

find_phase_offset(test_measurement.get_data(), TEST=True)
