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

# Computes mean of inputted NumPy array
def mean(values):
    sum = np.sum(values)
    mean = sum / values.size
    return mean

values = np.array([])
binned_values = np.array([])
binned_timestamps = np.array([])
sampling_rate = 1000
sampling_period = 1 / sampling_rate
sample_time = 10
starttime = time.time()

# Iterates through bins
for bin in range(1, sample_time * 1000 + 1):
    # Iterates through sample_data generator function for 1 sample_period seconds (1 millisecond)
    t_end = starttime + (bin * sampling_period)
    for adc_value in sample_data():
        if time.time() >= t_end:
            break
        values = np.append(values, adc_value)
    # Computes mean of data from 1ms of sampling and appends it binned_values
    binned_values = np.append(binned_values, mean(values))
    # Creates list of timestamps for each new sample of duration sampling_period
    binned_timestamps = np.append(binned_timestamps, sampling_period * bin - (sampling_period * 0.5))

# Plots the data and saves it to Digital-Stethoscope/Res/Data_Test
plt.plot(binned_timestamps, binned_values, linestyle='solid', linewidth=0.4)
plt.savefig('Res/Data_Test.png', bbox_inches='tight')
plt.show()