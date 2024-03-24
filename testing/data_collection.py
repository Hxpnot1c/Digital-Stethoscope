# Note that script should be run using 'sudo chrt 99 python Testing/Data_Collection.py' to ensure real time data collection
from Adafruit_MCP3008 import MCP3008
from Adafruit_GPIO.SPI import SpiDev
import time
import numpy as np
import matplotlib.pyplot as plt

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

values, binned_values, binned_timestamps = [], [], []
samples = 0
SAMPLING_RATE = 1000
SAMPLING_PERIOD = 1 / SAMPLING_RATE
SAMPLE_TIME = 10


print('Sampling...')
STARTTIME = time.perf_counter()
# Iterates through bins
for bin in range(1, SAMPLE_TIME * SAMPLING_RATE + 1):
    # Collects data samples using sample_data() function for 1 sample_period seconds (1 millisecond)
    t_end = STARTTIME + (bin * SAMPLING_PERIOD)
    while time.perf_counter() < t_end:
        values.append(mcp.read_adc(0))
    # Computes mean of data from 1ms of sampling and appends it binned_values
    binned_values.append(mean(values))
    samples += len(values)
    values = []
    # Creates list of timestamps for each new sample of duration sampling_period
    binned_timestamps.append(SAMPLING_PERIOD * bin - (SAMPLING_PERIOD * 0.5))

print('Sampling complete!')
print(f'No. of samples: {samples}')
print(f'Mean val: {mean(binned_values)}')
# Plots the data and saves it to Digital-Stethoscope/Res/Data_Test
plt.plot(binned_timestamps, binned_values, linestyle='solid', linewidth=0.4)
plt.savefig('res/Data_Test.png', bbox_inches='tight')
plt.show()
