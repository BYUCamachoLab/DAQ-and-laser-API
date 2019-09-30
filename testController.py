import visa


rm = visa.ResourceManager()
print(rm.list_resources())
controller = rm.open_resource('GPIB0::11::INSTR')
print(controller.query('*IDN?'))

controller.write(':PADD1:POSITION 300')
