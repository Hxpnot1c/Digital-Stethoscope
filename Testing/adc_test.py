from Adafruit_MCP3008 import MCP3008
from Adafruit_GPIO.SPI import SpiDev
import matplotlib.pyplot as plt
import time

# Access ADC using hardware SPI (because its faster)
spi_port = 0
spi_device = 0
mcp = MCP3008(spi=SpiDev(spi_port, spi_device, 5000))

# Data sampling function as a generator that outputs the current data value on a specified ADC channel and provides a timestamp of the data reading
def sample_data(channel=0):
    prev_value = 0
    start_time = time.time()
    while True:
        adc_value = mcp.read_adc(0)
        timestamp = time.time() - start_time
        if abs(adc_value - prev_value) <= 3:
            adc_value = prev_value
        yield adc_value, timestamp

values = []
timestamps = []
t_end = time.time() + 10
# Iterates through generator function to create a list of values with their timestamps for a period of 10 seconds
for adc_value, timestamp in sample_data():
    if time.time() >= t_end:
        break
    values.append(adc_value)
    timestamps.append(timestamp)
    time.sleep(1e-5)

# Plots the data and saves it to 'Data/ADC_Readings.png'
plt.plot (timestamps, values, linestyle='solid', linewidth=0.4)
plt.savefig('Res/ADC_Readings.png', bbox_inches='tight')
plt.show()

# Prints average samples per second
print(len(values) / 10)
