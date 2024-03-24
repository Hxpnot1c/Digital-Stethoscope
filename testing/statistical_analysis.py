from Adafruit_MCP3008 import MCP3008
from Adafruit_GPIO.SPI import SpiDev
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import signal, fft
import time
from scipy.io.wavfile import write

# Access ADC using hardware SPI (because its faster)
SPI_PORT = 0
SPI_DEVICE = 0
mcp = MCP3008(spi=SpiDev(SPI_PORT, SPI_DEVICE))

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
SAMPLING_RATE = 1000
SAMPLE_PERIOD = 1 / SAMPLING_RATE
SAMPLING_TIME = 20
bpm_sample_num = 11 + 2

print('Sampling...')
STARTTIME = time.perf_counter()
next_bpm_sample_time = 0
# Iterates through bins
for bin in range(1, SAMPLING_TIME * SAMPLING_RATE + 1):
    # Collects data samples using sample_data() function for 1 sample_period seconds (1 millisecond)
    t_end = STARTTIME + (bin * SAMPLE_PERIOD)
    while (time.perf_counter()) < t_end:
        values.append(mcp.read_adc(0))
    # Computes mean of data from 1ms of sampling and appends it binned_values
    binned_values.append(mean_val := mean(values))
    values = []
    # Creates list of timestamps for each new sample of duration sample_period
    binned_timestamps.append(current_time := SAMPLE_PERIOD * bin - (SAMPLE_PERIOD * 0.5))
    if mean_val >= 700 and current_time >= next_bpm_sample_time:
        bpm_sample_times.append(current_time)
        next_bpm_sample_time = current_time + 0.2
        if len(bpm_sample_times) == 11:
            t1 = bpm_sample_times[-1] - bpm_sample_times[-3]
            t2 = bpm_sample_times[-3] - bpm_sample_times[-5]
            t3 = bpm_sample_times[-5] - bpm_sample_times[-7]
            t4 = bpm_sample_times[-7] - bpm_sample_times[-9]
            t5 = bpm_sample_times[-9] - bpm_sample_times[-11]
            print(calculate_heart_rate(t1, t2, t3, t4, t5))
        elif len(bpm_sample_times) == bpm_sample_num:
            bpm_sample_num += 2
            t1, t2, t3, t4, t5 = cycle_forward(bpm_sample_times[-1] - bpm_sample_times[-3], t1, t2, t3, t4, t5)
            print(calculate_heart_rate(t1, t2, t3, t4, t5))
print('Sampling complete!')
timestamps = binned_timestamps
input_signal = binned_values

# Plotting unfiltered vs fiiltered data
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(timestamps, input_signal, linestyle='solid', linewidth=0.4)
ax1.set_title('Input Signal')
CUTOFF_FREQUENCY = 200
ORDER = 4
sos = signal.butter(ORDER, CUTOFF_FREQUENCY, 'low', False, 'sos', SAMPLING_RATE)
low_pass_signal, _ = signal.sosfilt(sos, input_signal, zi=signal.sosfilt_zi(sos) * 387.5)
sos2 = signal.butter(ORDER, CUTOFF_FREQUENCY, 'low', False, 'sos', SAMPLING_RATE)
band_pass_signal, _ = signal.sosfilt(sos2, low_pass_signal, zi=signal.sosfilt_zi(sos2) * 387.5)
ax2.plot(timestamps, band_pass_signal, linestyle='solid', linewidth=0.4)
ax2.set_title('4th Order Butterworth Filter at 20Hz and 200Hz')
plt.savefig('Res/Archived_Data/Image Plots/Raw vs Filtered Data.png', bbox_inches='tight')
plt.show()

# Writing .wav files for unfiltered and filtered audio data
remapped_input_signal = np.int16(remap_range(np.array(input_signal), 0, 1023, -31768, 32767))
write('res/Archived_Data/Audio/Input_Signal.wav', SAMPLING_RATE, remapped_input_signal)
remapped_filtered_signal = np.int16(remap_range(np.array(band_pass_signal), 0, 1023, -31768, 32767))
write('res/Archived_Data/Audio/Filtered_Signal.wav', SAMPLING_RATE, remapped_filtered_signal)

# Plot frquency domain data for unfiltered and filtered data using fast fourier transform
NUM_SAMPLES = SAMPLING_RATE * SAMPLING_TIME
y1 = fft.fft(remapped_input_signal)
y2 = fft.fft(remapped_filtered_signal)
x = fft.fftfreq(NUM_SAMPLES, SAMPLE_PERIOD)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(x, y1, linestyle='solid', linewidth=0.4)
ax1.set_title('Frequency Domain Input Signal')
ax2.plot(x, y2, linestyle='solid', linewidth=0.4)
ax2.set_title('Frequency Domain Filtered Signal')
plt.xlim(0, 500)
plt.savefig('res/Archived_Data/Image Plots/Frequency Domain Raw vs Filtered Data.png', bbox_inches='tight')
plt.show()
