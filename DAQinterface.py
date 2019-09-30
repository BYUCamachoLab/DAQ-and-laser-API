import nidaqmx

# TODO: Add documentation


def add_channel(task, device, channel):
    task.ai_channels.add_ai_voltage_chan(device + "/" + channel)
    return task


def add_channels(task, channels):
    for i in range(len(channels)):
        task.ai_channels.add_ai_voltage_chan(channels[i])
    return task


def update_time(task, sample_rate, samples_per_chan):
    task.timing.cfg_samp_clk_timing(sample_rate, samps_per_chan=samples_per_chan)
    return task


def init_task(device="cDAQ1Mod1", channels=None, sample_rate=10000, samples_per_chan=100):
    if channels is None:
        channels = []
    task = nidaqmx.Task()
    # Channels only added if specified.
    for k in range(len(channels)):
        task = add_channel(task, device=device, channel=channels[k])
    task = update_time(task, sample_rate, samples_per_chan)
    return task


def get100data():
    with nidaqmx.Task() as task:
        # Configure task
        task.ai_channels.add_ai_voltage_chan("cDAQ1Mod1/ai0")
        task.timing.cfg_samp_clk_timing(10000, samps_per_chan=100)
        # Read the data
        data = task.read(number_of_samples_per_channel=100)
        return data


def test_daq():
    print("testDAQ")
    task = init_task()
    print(task.timing.samp_clk_rate)
    data = task.read(number_of_samples_per_channel=100)
    print(data)
    task.close()
