from Adafruit_MCP3008 import MCP3008
from Adafruit_GPIO.SPI import SpiDev
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import signal 
import time

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

values, binned_values, binned_timestamps = [], [], []
samples = 0
sampling_rate = 1000
sampling_period = 1 / sampling_rate
sample_time = 10


print('Sampling...')
starttime = time.perf_counter()
# Iterates through bins
for bin in range(1, sample_time * 1000 + 1):
    # Collects data samples using sample_data() function for 1 sample_period seconds (1 millisecond)
    t_end = starttime + (bin * sampling_period)
    while time.perf_counter() < t_end:
        values.append(mcp.read_adc(0))
    # Computes mean of data from 1ms of sampling and appends it binned_values
    binned_values.append(mean(values))
    samples += len(values)
    values = []
    # Creates list of timestamps for each new sample of duration sampling_period
    binned_timestamps.append(sampling_period * bin - (sampling_period * 0.5))
print('Sampling complete!')
timestamps = binned_timestamps
input_signal = binned_values



fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(timestamps, input_signal, linestyle='solid', linewidth=0.4)
ax1.set_title('Input Signal')

cutoff_frequency = 200
order = 4
sos = signal.butter(order, cutoff_frequency, 'low', False, 'sos', sampling_rate)
low_pass_signal, _ = signal.sosfilt(sos, input_signal, zi=signal.sosfilt_zi(sos) * 387.5)
sos2 = signal.butter(order, cutoff_frequency, 'low', False, 'sos', sampling_rate)
high_pass_signal, _ = signal.sosfilt(sos2, low_pass_signal, zi=signal.sosfilt_zi(sos2) * 387.5)
ax2.plot(timestamps, high_pass_signal, linestyle='solid', linewidth=0.4)
ax2.set_title('4th Order Butterworth Filter at 20Hz and 200Hz')
plt.savefig('Res/Raw vs Filtered Data.png', bbox_inches='tight')
plt.show()