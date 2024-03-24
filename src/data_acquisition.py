from Adafruit_MCP3008 import MCP3008
from Adafruit_GPIO.SPI import SpiDev
import numpy as np
import time
import pandas as pd
import pathlib
from soundfile import write


ROOT_DIR = pathlib.Path(__file__).resolve().parent


# Access ADC using hardware SPI (because its faster)
SPI_PORT = 0
SPI_DEVICE = 0
mcp = MCP3008(spi=SpiDev(SPI_PORT, SPI_DEVICE))


# Returns mean value of an array
def mean(values):
    if not len(values):
        return mcp.read_adc(0)
    sum = np.sum(values)
    mean = sum / len(values)
    return mean


# Rescales values from one range to another
def remap_range(values, org_min, org_max, new_min, new_max):
    org_range = org_max - org_min
    new_range = new_max - new_min
    normalised_values = (values - org_min) / org_range
    remapped_values = (normalised_values * new_range) + new_min
    return remapped_values


# Cycles time period readings forward one time for live BPM calculation
def cycle_forward(new_val, t1, t2, t3, t4, t5):
    t5 = t4
    t4 = t3
    t3 = t2
    t2 = t1
    t1 = new_val
    return t1, t2, t3, t4, t5


# Calculates heart rate in beats per minute given last 5 time periods between full beats
def calculate_heart_rate(t1, t2, t3, t4, t5):
    # Determines beats per second
    # Recent readings are weighted more than older readings by the following formula: weight for tₙ = f(n) where f(n) = round(5/sqrt(6) * sqrt(-n+6))
    # Therefore bps = sum(f(n) / tₙ)) / sum(f(n))
    bps = (5/t1 + 4/t2 + 4/t3 + 3/t4 + 2/t5) / 18
    # Converts to beats per minute
    bpm = bps * 60
    return bpm


# Initialise variables
values, binned_values, bpm_sample_times, bpm_values = [], [], [], []
SAMPLING_RATE = 1000 # Sampling rate of 1000 to align with preferred Nyquist frequency of 500Hz
SAMPLE_PERIOD = 1 / SAMPLING_RATE
SAMPLING_TIME = 180 # Samples for 3 minutes before automatically stopping
bpm_sample_num = 11 + 2
next_bpm_sample_time = 0

print('Sampling...')
STARTTIME = time.perf_counter()

# Main loop to collect heart data and use equal width binning to go from a variable sampling rate to a sampling rate of 1000
# Iterates through bins
for bin in range(1, SAMPLING_TIME * SAMPLING_RATE + 1):
    t_end = STARTTIME + (bin * SAMPLE_PERIOD) # Time to sample upto for current bin

    # Samples for 1ms and saves data to values
    while (time.perf_counter()) < t_end:
        values.append(mcp.read_adc(0))

    # Computes mean of data from 1ms of sampling and appends it binned_values
    binned_values.append(mean_val := mean(values))
    values = [] # Empties values for the next iteration/bin


    # Calculates realtive current bin time
    current_time = SAMPLE_PERIOD * bin - (SAMPLE_PERIOD * 0.5)

    # Record beat if amplitude is greater than or equal to 700 and enough time has passed since the last recorded beat
    if mean_val >= 700 and current_time >= next_bpm_sample_time:
        bpm_sample_times.append(current_time)
        next_bpm_sample_time = current_time + 0.2

        # Initialise time period variables
        if len(bpm_sample_times) == 11:
            t1 = bpm_sample_times[-1] - bpm_sample_times[-3]
            t2 = bpm_sample_times[-3] - bpm_sample_times[-5]
            t3 = bpm_sample_times[-5] - bpm_sample_times[-7]
            t4 = bpm_sample_times[-7] - bpm_sample_times[-9]
            t5 = bpm_sample_times[-9] - bpm_sample_times[-11]
            # Append BPM vlaue to bpm_values
            bpm_values.append(calculate_heart_rate(t1, t2, t3, t4, t5))

        # Every 2 values (one full double beat) update the time period variables and append the updates BPM to bpm_values
        elif len(bpm_sample_times) == bpm_sample_num:
            bpm_sample_num += 2
            t1, t2, t3, t4, t5 = cycle_forward(bpm_sample_times[-1] - bpm_sample_times[-3], t1, t2, t3, t4, t5)
            bpm_values.append(calculate_heart_rate(t1, t2, t3, t4, t5)) 

    # Every 800ms calculate values to plot on GUI
    if (bin+1) % 800 == 0:
        newest_bins = binned_values[-800:] # Only take newest 800 binned values
        rebinned_values = [] 

        # Split values into 400 new equal width bins to display on GUI at a rate of 500Hz
        for index in range(400):
            rebinned_values.append(sum(newest_bins[index*2:(index+1)*2]) / 2)

        # Update data.csv with new 500 samples per second heart data and most recent BPM (if BPM hasn't been calculated yet, output BPM of 0)
        if len(bpm_values) == 0:
            rebinned_values.append(0)
            pd.Series(rebinned_values).to_csv(ROOT_DIR / 'data.csv')
        else:
            bpm = round(bpm_values[-1])
            rebinned_values.append(bpm)
            pd.Series(rebinned_values).to_csv(ROOT_DIR / 'data.csv', index=False)

    # Every 10 seconds save heart data to audio_data.wav
    if (bin+1) % 10000 == 0:
        # Rescale data to be in required range to save as .wav
        data = remap_range(np.array(binned_values), 0, 1024, -1, 1)
        
        write(file=ROOT_DIR / 'audio_data.wav', data=data, samplerate=SAMPLING_RATE)

        # Reset binned_values
        binned_values = []

print('Sampling complete!')
