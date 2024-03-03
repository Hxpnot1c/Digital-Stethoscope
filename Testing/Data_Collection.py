from Adafruit_MCP3008 import MCP3008
from Adafruit_GPIO.SPI import SpiDev
import time
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Access ADC using hardware SPI (because its faster)
spi_port = 0
spi_device = 0
mcp = MCP3008(spi=SpiDev(spi_port, spi_device, 5000))

# Data sampling function as a generator that outputs the current data value on a specified ADC channel and provides a timestamp of the data reading
def sample_data(channel=0):
    prev_value = 0
    while True:
        adc_value = mcp.read_adc(0)
        if abs(adc_value - prev_value) <= 3:
            adc_value = prev_value
        yield adc_value

def mean(values):
    sum = np.sum(values)
    mean = sum / values.size
    return mean

# Bins data using equal-width binning to achieve a regular sample rate of 1ksps (Using max heart sound freq of 400Hz, corresponding Nyquist rate is 0.8ksps. 0.2ksps is added to minimise data loss)
# def equal_width_bins(values, timestamps, sampling_rate = 1000):
#     sampling_period = 1 / sampling_rate
#     binned_values = []
#     binned_timestamps = []
#     for bin_number in range(int(np.ceil(np.max(timestamps) / sampling_period))):
#         # Computes mean amplitude value in the current bin_number (indexing from 0) with a width of sampling_period seconds
#         bin_values = [values[x] for x in range(len(timestamps)) if timestamps[x] > bin_number * sampling_period and timestamps[x] <= (bin_number + 1) * sampling_period]
#         bin = np.mean(bin_values)
#         binned_values.append(bin)
#         binned_timestamps.append(sampling_period * bin_number + (sampling_period * 0.5))
#         print(bin_number)
#     return binned_values, binned_timestamps

values = np.array([])
binned_values = np.array([])
binned_timestamps = np.array([])
sampling_rate = 1000
sampling_period = 1 / sampling_rate
sample_time = 10
starttime = time.time()
for bin in range(1, sample_time * 1000 + 1):
    t_end = starttime + (bin * sampling_period)
    # Iterates through generator function to create a list of values with their timestamps for a period of 10 seconds
    for adc_value in sample_data():
        if time.time() >= t_end:
            break
        values = np.append(values, adc_value)
    binned_values = np.append(binned_values, mean(values))
    binned_timestamps = np.append(binned_timestamps, sampling_period * bin - (sampling_period * 0.5))
print(binned_values)
plt.plot(binned_timestamps, binned_values, linestyle='solid', linewidth=0.4)
plt.savefig('Res/Data_Test.png', bbox_inches='tight')
plt.show()