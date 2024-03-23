![Calday Robotics Logo](https://github.com/Hxpnot1c/Digital-Stethoscope/blob/main/docs/calday_robotics_logo.png)

--------------------------------------------------------------------------------

# Digital-Stethoscope

## Video Presentation

[![Presentation video by Calday Robotics](https://img.youtube.com/vi/tSeS5dvHLK4/0.jpg)](https://www.youtube.com/watch?v=tSeS5dvHLK4)

## Video Demonstration

[![Demonstration video by Calday Robotics](https://img.youtube.com/vi/pRtG8VmfHiQ/0.jpg)](https://www.youtube.com/watch?v=pRtG8VmfHiQ)

## Summary
<details>
    <summary>500 Word Summary</summary>
<br />
In regions where access to advanced medical equipment is limited, diagnosing cardiac irregularities such as arrhythmias, murmurs and extrasystoles poses significant challenges for healthcare providers. Our solution; a 3D printed, digital stethoscope with integrated AI. 

While analogue stethoscopes are still a vital and valued tool, it poses issues that need to be addressed. Namely, the heartbeat can be quiet, making it difficult for doctors to accurately diagnose. Moreover, they lack visual representation, which could enhance understanding.

Our device tackles these challenges with advanced sound capture. It amplifies heartbeats using a 3D printed stethoscope head, then records them with a microphone. This captured sound data is processed in real-time to create a live graph of the patient's heartbeat and rhythm. The device features a user-friendly interface that displays this live plot alongside fields for recording patient information and notes.

What sets it apart is its integration with advanced artificial intelligence, enabling discrete real-time analysis of recorded data. The integrated AI uses a deep convolutional neural network to diagnose heart irregularities such as murmurs and extrasystoles. First, the audio data is converted into features such as the zero crossing rate, a chroma short time Fourier transform, the Mel-frequency cepstrum coefficient and a Mel-Spectrogram. This allows for unnecessary information to be abstracted from the input data. The data is then fed into a deep neural network that consists of 5 layers and over 95,000 trainable parameters. This network consists of 3, 1-Dimensional convolutional layers and 2 Linear Layers. In order to train the model, we used a cross-entropy loss function and the AdamW optimiser. In order to prevent overfitting, we set the AdamW optimiser with an aggressive weight decay, upsampled minority classes and created synthetic data with added noise. This allowed it to achieve an accuracy of 98% on data that it had never seen before.

Healthcare providers will be able to receive immediate feedback and diagnostic suggestions, empowering them to make prompt and informed decisions regarding patient care.

One of the main benefits of this digital stethoscope is its low cost. With a total cost of only £40.94. Compared to traditional stethoscopes, our stethoscope presents itself as a key-choice in resource-constrained settings. In order to achieve this, we used a Raspberry Pi 3 Model B+ due to its sufficient processing power while minimising costs and an FDM 3D printer to create a majority of the stethoscope. The head and ring are made out of PLA and the connection piece for the microphone is made out of TPU to provide a sound isolating fit for the microphone. The cost allows for our design to be much more accessible in areas where medical resources are limited.

By facilitating early detection of cardiac irregularities, the Digital Stethoscope enables timely intervention and improved patient outcomes. Its accessibility extends to underserved communities, where access to specialised equipment is limited. With its potential to revolutionise cardiac diagnostics in developing countries, the Digital Stethoscope holds promise for saving lives and improving healthcare delivery worldwide.


</details>

 
## Build Guide
<details>
    <summary>Build guide</summary>
<br />
Print all 3D models in 'Digital-Stethoscope/3d_models':<br />
-  PLA_Stethoscope_Head.stl should be printed in PLA with 100% infil<br />
-  PLA_Stethoscope_Ring.stl should be printed in PLA with 100% infill<br />
-  TPU_Stethoscope_Connection.stl should be printed in TPU with 100% infill

Note: Our models were printed using a Creality CR10 Smart Pro 3D printer. Most FDM 3D printers will be sufficient.<br />

We also used the default setting on Cura slicer (with 100% infill).<br />
For Stethoscope Ring: Supports ON - Normal<br />

For the diaphragm we used a 40mm diameter silicone piece with a thickness of 0.35mm

See Assembly Video below for a 3D rendering of how to assemble the 3D printed parts and the diaphragm.

[![Substitutionary Rick Roll to be replaced with assembly video](https://img.youtube.com/vi/dQw4w9WgXcQ/0.jpg)](https://www.youtube.com/watch?v=dQw4w9WgXcQ)

All connections require no adhesive and rely on a friction fit.
In order to attach the Ring some pressure will be required this will then provide a tight fit.

Microphone will then also provide a tight fit into the TPU connection piece

The following connections may require soldering and the use of breadboards. Please see the below pinout images to demonstrate pin connections.<br />
Connect V<sub>DD</sub> on the MCP3008 ADC to 3.3V on the Raspberry Pi using pin 17<br />
Connect V<sub>REF</sub> on the MCP3008 ADC to 3.3V on the Raspberry Pi using pin 17<br />
Connect AGND on the MCP3008 ADC to GND on the Raspberry Pi using pin 25<br />
Connect CLK on the MCP3008 ADC to GPIO 11 (SCLK) on the Raspberry Pi using pin 23<br />
Connect D<sub>OUT</sub> on the MCP3008 ADC to GPIO 9 (MISO) on the Raspberry Pi using pin 21<br />
Connect D<sub>IN</sub> on the MCP3008 ADC to GPIO 10 (MOSI) on the Raspberry Pi using pin 19<br />
Connect CS/SHDN on the MCP3008 ADC to GPIO 8 (CE0) on the Raspberry Pi using pin 8<br />
Connect GND on the MAX9814 microphone to GND on the Raspberry Pi using pin 25<br />
Connect V+ on the MAX9814 microphone to 3.3V on the Raspberry Pi using pin 17<br />
Connect OUT on the MAX9814 microphone to CH0 on the MCP3008 ADC<br />

![Raspberry Pi pinout](https://www.raspberrypi.com/documentation/computers/images/GPIO-Pinout-Diagram-2.png)
![MCP3008 ADC pinout](https://cdn-learn.adafruit.com/assets/assets/000/001/222/medium800/raspberry_pi_mcp3008pin.gif)
![MAX9814 microphone pinout](https://pmdway.com/cdn/shop/products/Electret-Microphone-Amplifier-MAX9814-Auto-Gain-Control-pmdway-3_708x408.jpg)

The Raspberry Pi should be setup with Raspberry Pi OS (Legacy, 64 bit)<br />
Note: 64 bit is required for pytorch

Install the following dependencies using pip:<br />
-  adafruit_mcp3008
-  librosa
-  matplotlib
-  pandas
-  pyqt5
-  soundfile
-  torch

Download 'Digital-Stethoscope/src'

Run the following commands in the terminal to run the program:<br />
'sudo chrt 99 python src/model_prediction.py'
'sudo chrt 99 python src/gui.py'
'sudo chrt 99 python src/data_acquisition.py'
</details>
