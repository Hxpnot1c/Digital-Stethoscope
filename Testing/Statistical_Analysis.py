from Adafruit_MCP3008 import MCP3008
from Adafruit_GPIO.SPI import SpiDev
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import signal, fft
import time
from scipy.io.wavfile import write

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

values, binned_values, binned_timestamps = [], [], []
samples = 0
sampling_rate = 1000
sample_period = 1 / sampling_rate
sampling_time = 10

print('Sampling...')
starttime = time.perf_counter()
# Iterates through bins
for bin in range(1, sampling_time * sampling_rate + 1):
    # Collects data samples using sample_data() function for 1 sample_period seconds (1 millisecond)
    t_end = starttime + (bin * sample_period)
    while time.perf_counter() < t_end:
        values.append(mcp.read_adc(0))
    # Computes mean of data from 1ms of sampling and appends it binned_values
    binned_values.append(mean(values))
    samples += len(values)
    values = []
    # Creates list of timestamps for each new sample of duration sample_period
    binned_timestamps.append(sample_period * bin - (sample_period * 0.5))
print('Sampling complete!')
timestamps = binned_timestamps
input_signal = binned_values

# Plotting unfiltered vs fiiltered data
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(timestamps, input_signal, linestyle='solid', linewidth=0.4)
ax1.set_title('Input Signal')
cutoff_frequency = 200
order = 4
sos = signal.butter(order, cutoff_frequency, 'low', False, 'sos', sampling_rate)
low_pass_signal, _ = signal.sosfilt(sos, input_signal, zi=signal.sosfilt_zi(sos) * 387.5)
sos2 = signal.butter(order, cutoff_frequency, 'low', False, 'sos', sampling_rate)
band_pass_signal, _ = signal.sosfilt(sos2, low_pass_signal, zi=signal.sosfilt_zi(sos2) * 387.5)
ax2.plot(timestamps, band_pass_signal, linestyle='solid', linewidth=0.4)
ax2.set_title('4th Order Butterworth Filter at 20Hz and 200Hz')
plt.savefig('Res/Archived_Data/Image Plots/Raw vs Filtered Data.png', bbox_inches='tight')
plt.show()

# Writing .wav files for unfiltered and filtered audio data
remapped_input_signal = np.int16(remap_range(np.array(input_signal), 0, 1023, -31768, 32767))
write('Res/Archived_Data/Audio/Input_Signal.wav', sampling_rate, remapped_input_signal)
remapped_filtered_signal = np.int16(remap_range(np.array(band_pass_signal), 0, 1023, -31768, 32767))
write('Res/Archived_Data/Audio/Filtered_Signal.wav', sampling_rate, remapped_filtered_signal)

# Plot frquency domain data for unfiltered and filtered data using fast fourier transform
samples = sampling_rate * sampling_time
y1 = fft.fft(remapped_input_signal)
y2 = fft.fft(remapped_filtered_signal)
x = fft.fftfreq(samples, sample_period)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(x, y1, linestyle='solid', linewidth=0.4)
ax1.set_title('Frequency Domain Input Signal')
ax2.plot(x, y2, linestyle='solid', linewidth=0.4)
ax2.set_title('Frequency Domain Filtered Signal')
plt.xlim(0, 500)
plt.savefig('Res/Archived_Data/Image Plots/Frequency Domain Raw vs Filtered Data.png', bbox_inches='tight')
plt.show()
