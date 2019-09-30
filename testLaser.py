from TSL550 import TSL550
import time

address = "COM4"
laser = TSL550(address, baudrate=19200)
laser.on()
print(laser.print_status())

lambda_start = 1550
lambda_stop  = 1551
duration = 2
trigger_num_points = 1000
trigger_sweep_rate = 1000
trigger_step = (lambda_stop - lambda_start) / duration * trigger_sweep_rate * (1 / trigger_num_points)
print(trigger_step)

print(laser.print_status())
laser.closeShutter()
laser.sweep_set_mode(continuous=True, twoway=True, trigger=False, const_freq_step=False)
laser.trigger_enable_output()
print(laser.trigger_set_mode("Step"))
print(laser.trigger_set_step(0.001))



laser.sweep_wavelength(start=lambda_start,stop=lambda_stop,duration=duration,number=1)
time.sleep(5)
print(laser.wavelength_logging_number())
print(len(laser.wavelength_logging()))
