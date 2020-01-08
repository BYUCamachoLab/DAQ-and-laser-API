#------------------------------------------------------------------------------#
#
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Libraries
#------------------------------------------------------------------------------#
import visa
try:
    #Import when main.
    import VISAresourceExtentions
except ImportError:
    #Import when module.
    from . import VISAresourceExtentions

#------------------------------------------------------------------------------#
#Instrument Class
#------------------------------------------------------------------------------#
class RTO:
    """ Simple network controller class for R&S RTO oscilloscopes. """
    device = None
    def __init__(self, address):
        rm = visa.ResourceManager()
        self.device = rm.open_resource("TCPIP::{}::hislip0".format(address))
        #Pass VISA commands directly through.
        self.query = self.device.query
        self.write = self.device.write
        #Device Initialization
        self.device.write_termination = '' #Todo: Make this a setting.
        self.device.ext_clear_status()
        print("Connected: {}".format(self.device.query('*IDN?')))
        self.device.write('*RST;*CLS')
        self.device.write('SYST:DISP:UPD ON')
        self.device.ext_error_checking()

    def __send_command(self, command):
        self.device.write(command)
        self.wait_for_device()
        self.device.ext_error_checking()

    def wait_for_device(self):
        self.device.query('*OPC?') #Waits for the device until last action is complete.

    def acquisition_settings(self,
        sample_rate,
        duration,
        force_realtime = False
    ):
        #Set acquisition settings.
        short_command = 'ACQ:POIN:AUTO RES;:ACQ:SRAT {};:TIM:RANG {}'.format(
            sample_rate, duration
        )
        if force_realtime:
            self.__send_command('ACQ:MODE RTIM')
        self.__send_command(short_command)

    def add_channel(self, channel_num, range, position = 0, offset = 0, coupling = "DCL"):
        #Add a channel.
        short_command = 'CHAN{}:RANG {};POS {};OFFS {};COUP {};STAT ON'.format(
            channel_num, range, position, offset, coupling
        )
        self.__send_command(short_command)

    def __add_trigger(self,
        type,
        source,
        source_num,
        level,
        trigger_num = 1,
        mode = "NORM",
        settings: str = ""
    ):
        short_command = 'TRIG{}:MODE {};SOUR {};TYPE {};LEV{} {};'.format(
        trigger_num, mode, source, type, source_num, level
    )
        #Add a trigger.
        self.__send_command(short_command + settings)

    def edge_trigger(self, source_channel, trigger_level):
        #Todo: Edit to allow other trigger sources (ex. External Trigger).
        self.__add_trigger(
            type = "EDGE",
            source = "CHAN{}".format(source_channel),
            source_num = source_channel,
            level = trigger_level,
            settings = "EDGE:SLOP POS"
        )

    def start_acquisition(self, timeout, type = 'SING'):
        self.device.timeout = timeout*1000 #Translate seconds to ms.
        self.device.write(type)

    def get_data_ascii(self, channel):
        dataQuery = 'FORM ASC;:CHAN{}:DATA?'.format(channel)
        waveform = self.device.query_ascii_values(dataQuery)
        return waveform

    def get_data_binary(self, channel):
        dataQuery = 'FORM REAL;:CHAN{}:DATA?'.format(channel)
        waveform = self.device.query_binary_values(dataQuery)
        return waveform

    def take_screenshot(self, path):
        instrument_save_path = '\'C:\\temp\\Last_Screenshot.png\''
        self.device.write('HCOP:DEV:LANG PNG')
        self.device.write('MMEM:NAME {}'.format(instrument_save_path))
        self.device.write('HCOP:IMM')
        self.wait_for_device()
        self.device.ext_error_checking()
        self.device.ext_query_bin_data_to_file(
            'MMEM:DATA? {}'.format(instrument_save_path),
            str(path)
        )
        self.device.ext_error_checking()
