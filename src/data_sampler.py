from Adafruit_MCP3008 import MCP3008
from Adafruit_GPIO.SPI import SpiDev
import numpy as np
import time
import pandas as pd
import pathlib

root_dir = pathlib.Path(__file__).resolve().parent

# Access ADC using hardware SPI (because its faster)
spi_port = 0
spi_device = 0
mcp = MCP3008(spi=SpiDev(spi_port, spi_device))


# Computes mean of inputted NumPy array
def mean(values):
    if not len(values):
        print("Error")
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



values, binned_values, binned_timestamps, bpm_sample_times = [], [], [], []
sampling_rate = 10000
sample_period = 1 / sampling_rate
sampling_time = 20
bpm_sample_num = 11 + 2
bpm_values = []

print('Sampling...')
next_bpm_sample_time = 0
starttime = time.perf_counter()

# Iterates through bins
for bin in range(1, sampling_time * sampling_rate + 1):
    t_end = starttime + (bin * sample_period)
    while (time.perf_counter()) < t_end:
        values.append(mcp.read_adc(0))

    # Computes mean of data from 1ms of sampling and appends it binned_values
    binned_values.append(mean_val := mean(values))
    values = []

    # Creates list of timestamps for each new sample of duration sample_period
    binned_timestamps.append(current_time := sample_period * bin - (sample_period * 0.5))
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

    if (bin+1) % 1000 == 0:
        newest_bins = binned_values[-1000:]
        rebinned_values = []
        for index in range(50):
            rebinned_values.append(sum(newest_bins[index*20:(index+1)*20]) / 20)
        if len(bpm_values) == 0:
            rebinned_values.append('000')
            pd.Series(rebinned_values).to_csv(root_dir / 'data.csv')
        else:
            bpm = str(round(bpm_values[-1]))
            rebinned_values.append(bpm)
            pd.Series(rebinned_values).to_csv(root_dir / 'data.csv')
        
            

print('Sampling complete!')
