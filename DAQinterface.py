import nidaqmx

# TODO: Add documentation


class NIDAQInterface:

    """
    API class for interfacing with a DAQ card to read input.
    Works with a single analog voltage measurement.

    data members:
        task - The task object used for interfacing with the National Instruments DAQ device.
        CARD_ONE_MAX_SAMPLE_RATE - maximum sample rate for our lab's first daq card
        CARD_TWO_MAX_SAMPLE_RATE - maximum sample rate for our lab's second daq card
        DEFAULT_SAMPLES_PER_CHANNEL - number of samples to generate for each channel in the task,
                                      used to determine buffer size
    """

    CARD_ONE_MAX_SAMPLE_RATE = 10e4
    CARD_TWO_MAX_SAMPLE_RATE = 5e4
    DEFAULT_SAMPLES_PER_CHANNEL = 100

    def __init__(self):
        self.task = nidaqmx.Task()
        self.sample_rate = None
        self.samples_per_channel = None

    def add_channel(self, channel):
        """
        Adds a channel to the NIDAQ Task object.
        :param device: the device name
        :param channel: the channel name
        :return: n/a
        """
        self.task.ai_channels.add_ai_voltage_chan(channel)

    def add_channels(self, channels):
        """
        Adds several channels to the NIDAQ Task object.
        :param channels: a list of full channel names (device name + channel name)
        :return: n/a
        """
        for i in range(len(channels)):
            self.task.ai_channels.add_ai_voltage_chan(channels[i])

    def update_timing(self, sample_rate, samples_per_chan):
        """
        Sets the sample rate and number of samples to read for the Task
        :param sample_rate: the desired sample rate
        :param samples_per_chan: the desired number of data points to read
        :return: n/a
        """
        self.sample_rate = sample_rate
        self.samples_per_channel = samples_per_chan
        self.task.timing.cfg_samp_clk_timing(sample_rate, samps_per_chan=samples_per_chan)

    def initialize(self, channels=None, sample_rate=CARD_TWO_MAX_SAMPLE_RATE,
                   samples_per_chan=DEFAULT_SAMPLES_PER_CHANNEL):
        """
        Initialize the Task with the most commonly used parameters and the specified channels
        :param channels: a list of channels with full channel names
        :param sample_rate: a
        :param samples_per_chan:
        :return:
        """
        # Channels only added if specified
        if channels is not None:
            self.add_channels(channels)
        self.update_timing(sample_rate, samples_per_chan)

    def read(self, timeout_duration):
        return self.task.read(self.samples_per_channel, timeout_duration)

    def get100data(self, channel):
        with nidaqmx.Task() as task:
            # Configure task
            task.ai_channels.add_ai_voltage_chan(channel)
            task.timing.cfg_samp_clk_timing(self.CARD_TWO_MAX_SAMPLE_RATE,
                                            samps_per_chan=self.DEFAULT_SAMPLES_PER_CHANNEL)
            # Read the data
            data = task.read(number_of_samples_per_channel=self.DEFAULT_SAMPLES_PER_CHANNEL)
            return data

    def __del__(self):
        self.task.close()
