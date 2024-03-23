from Adafruit_MCP3008 import MCP3008
from Adafruit_GPIO.SPI import SpiDev
import numpy as np
import time
import pandas as pd
import pathlib
from soundfile import write

root_dir = pathlib.Path(__file__).resolve().parent

# Access ADC using hardware SPI (because its faster)
spi_port = 0
spi_device = 0
mcp = MCP3008(spi=SpiDev(spi_port, spi_device))


def mean(values):
    if not len(values):
        return mcp.read_adc(0)
    sum = np.sum(values)
    mean = sum / len(values)
    return mean


def remap_range(values, org_min, org_max, new_min, new_max):
    org_range = org_max - org_min
    new_range = new_max - new_min
    normalised_values = (values - org_min) / org_range
    remapped_values = (normalised_values * new_range) + new_min
    return remapped_values


def cycle_forward(new_val, t1, t2, t3, t4, t5):
    t5 = t4
    t4 = t3
    t3 = t2
    t2 = t1
    t1 = new_val
    return t1, t2, t3, t4, t5


def calculate_heart_rate(t1, t2, t3, t4, t5):
    bps = (5/t1 + 4/t2 + 4/t3 + 3/t4 + 2/t5) / 18
    bpm = bps * 60
    return bpm


values, binned_values, bpm_sample_times, bpm_values = [], [], [], []
sampling_rate = 1000
sample_period = 1 / sampling_rate
sampling_time = 180
bpm_sample_num = 11 + 2

print('Sampling...')
next_bpm_sample_time = 0
starttime = time.perf_counter()

# Iterates through bins
for bin in range(1, sampling_time * sampling_rate + 1):
    t_end = starttime + (bin * sample_period)
    while (time.perf_counter()) < t_end:
        # values.append(np.random.randint(0, 1024))
        values.append(mcp.read_adc(0))

    # Computes mean of data from 1ms of sampling and appends it binned_values
    binned_values.append(mean_val := mean(values))
    values = []

    # Creates list of timestamps for each new sample of duration sample_period
    current_time = sample_period * bin - (sample_period * 0.5)
    if mean_val >= 700 and current_time >= next_bpm_sample_time:
        bpm_sample_times.append(current_time)
        next_bpm_sample_time = current_time + 0.2

        if len(bpm_sample_times) == 11:
            t1 = bpm_sample_times[-1] - bpm_sample_times[-3]
            t2 = bpm_sample_times[-3] - bpm_sample_times[-5]
            t3 = bpm_sample_times[-5] - bpm_sample_times[-7]
            t4 = bpm_sample_times[-7] - bpm_sample_times[-9]
            t5 = bpm_sample_times[-9] - bpm_sample_times[-11]
            bpm_values.append(calculate_heart_rate(t1, t2, t3, t4, t5))

        elif len(bpm_sample_times) == bpm_sample_num:
            bpm_sample_num += 2
            t1, t2, t3, t4, t5 = cycle_forward(bpm_sample_times[-1] - bpm_sample_times[-3], t1, t2, t3, t4, t5)
            bpm_values.append(calculate_heart_rate(t1, t2, t3, t4, t5)) 

    if (bin+1) % 800 == 0:
        newest_bins = binned_values[-800:]
        rebinned_values = []
        for index in range(400):
            rebinned_values.append(sum(newest_bins[index*2:(index+1)*2]) / 2)
        if len(bpm_values) == 0:
            rebinned_values.append(0)
            pd.Series(rebinned_values).to_csv(root_dir / 'data.csv')
        else:
            bpm = round(bpm_values[-1])
            rebinned_values.append(bpm)
            pd.Series(rebinned_values).to_csv(root_dir / 'data.csv', index=False)

    if (bin+1) % 10000 == 0:
        data = remap_range(np.array(binned_values), 0, 1024, -1, 1)
        write(file=root_dir / 'audio_data.wav', data=data, samplerate=sampling_rate)
        binned_values = []

print('Sampling complete!')
