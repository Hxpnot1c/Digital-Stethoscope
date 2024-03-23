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

For the diaphragm, we used a 40mm diameter silicone piece with a thickness of 0.35mm

See Assembly Video below for a 3D rendering of how to assemble the 3D printed parts and the diaphragm.

[![Substitutionary Rick Roll to be replaced with assembly video](https://img.youtube.com/vi/KgJvibv4-vc/0.jpg)](https://www.youtube.com/watch?v=KgJvibv4-vc)

All connections require no adhesive and rely on a friction fit.
In order to attach the Ring some pressure will be required, this will then provide a tight fit.

The microphone will then also provide a tight fit into the TPU connection piece

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

## Further Reading
<details>
    <summary>Definitions and useful links</summary>
<br />
Definitions:
Heart Arrhythmia:<br />
An arrhythmia is an abnormality of the heart's rhythm.

Heart Murmur:<br />
Heart murmurs are sounds — such as whooshing or swishing — made by rapid, choppy (turbulent) blood flow through the heart.

Extrasystoles/Premature ventricular contractions:<br />
Extra heartbeats that begin in one of the heart's two lower pumping chambers (ventricles).

Zero-Crossing Rate (ZCR):<br />
A measurement in audio processing that counts the number of times a signal crosses zero from positive to negative or vice versa within a specific time window. It's a simple way to characterise the "brightness" or "harshness" of a sound.
    
Chroma Short-Time Fourier Transform (Chroma STFT):<br />
A signal processing technique used to analyse audio. It breaks down the sound into its component frequencies over short periods of time. Chroma STFT focuses specifically on the pitch information, representing the signal in terms of musical notes (chroma).
    
Mel-Frequency Cepstrum Coefficient (MFCC):<br />
A feature extraction technique commonly used in speech and audio recognition. It mimics how the human auditory system perceives sound by converting the sound's frequency spectrum into a representation on the Mel scale, which approximates human hearing. MFCCs capture the spectral envelope of the sound, making them useful for tasks like speaker identification and speech recognition.
    
Mel-Spectrogram:<br />
A visual representation of a sound's frequency content over time. It uses colour to represent the intensity of different frequencies at different time points. Mel spectrograms are often used in conjunction with MFCCs, as they provide a more intuitive way to understand the spectral information captured by MFCCs.
    
Deep Neural Network (DNN):<br />
A type of artificial neural network with multiple layers of interconnected nodes. DNNs can learn complex patterns from data and are powerful tools for tasks like image recognition, natural language processing, and speech recognition.
    
Convolutional Neural Network (CNN):<br />
A powerful type of artificial neural network used to classify sounds. Similar to image processing, CNNs excel at finding patterns in audio data.
    
Trainable Parameters:<br />
The numerical values within a deep neural network that are adjusted during the training process. These parameters determine how the network maps input data to output predictions. By adjusting these parameters, the network learns to perform a specific task.
    
1-Dimensional Convolutional Layers:<br />
A specific type of layer in a deep neural network used for processing sequential data like audio or text. These layers apply a filter (like a small window) that slides across the input data, extracting features based on local patterns. By stacking multiple convolutional layers, the network can learn increasingly complex features.
    
Linear Layers:<br />
Layers in a deep neural network that perform a weighted sum of their inputs. These layers are often used at the end of a network to combine the learned features and produce the final output prediction.
    
ReLU Activation Function:<br />
A popular activation function used in artificial neural networks. It adds a non-linearity to the network, which is crucial for its ability to learn complex patterns from data.
    
Cross-Entropy Loss Function:<br />
A common function used to measure the error between a neural network's predictions and the true labels of the data. It's particularly useful for classification tasks where the network outputs probabilities for different categories. The loss function helps the network learn by indicating how much its predictions deviate from the desired outcome.
    
AdamW Optimizer:<br />
An optimization algorithm used to train deep neural networks. It efficiently adjusts the network's trainable parameters based on the calculated loss function. AdamW is a variant of the Adam optimizer that addresses certain stability issues.
    
Overfitting:<br />
A situation where a deep neural network performs well on the training data but poorly on unseen data. This occurs when the network memorises specific details of the training examples rather than learning generalisable patterns. Weight decay and other regularisation techniques can help prevent overfitting.
    
Weight Decay:<br />
A technique used during training to prevent a deep neural network from overfitting to the training data. It penalises the network for having large weights, encouraging it to learn more generalisable features.
    
Dropout Layers:<br />
A technique used in artificial neural networks to improve their performance, especially to address overfitting, by randomly deactivating a certain percentage of neurons in a layer during training. This forces the network to learn to use different combinations of neurons each time, preventing it from relying too heavily on any specific neuron or connection.
    
Upsampling:<br />
An operation that increases the resolution of an image or signal. In audio processing, it might involve interpolating new data points between existing ones to create a higher sampling rate.
    
Minority Classes:<br />
In a classification task with multiple categories, the classes with the fewest data points are referred to as minority classes.
    
Class Imbalance:<br />
A situation in a classification dataset where some classes have significantly fewer data points than others. This can pose challenges for training a deep neural network, as the model might prioritise learning patterns from the majority classes and perform poorly on the minority classes.
    
Noise (in reference to audio):<br />
Unwanted sound that disrupts the desired audio signal. Noise can come from various sources and can significantly impact the quality and clarity of audio recordings or playback.
    
MAX9814:<br />
An integrated circuit (IC) that combines a microphone amplifier with automatic gain control (AGC). It's commonly used in audio applications to boost weak microphone signals to a usable level. AGC helps ensure the amplified signal stays within a certain range, preventing distortion.
    
MCP3008:<br />
An analogue-to-digital converter (ADC) IC that converts analogue voltage signals from sensors or other circuits into digital data that can be processed by microcontrollers or computers. It has 8 channels, meaning it can convert signals from up to 8 analogue inputs simultaneously. The MCP3008 typically uses a serial communication protocol (SPI) to communicate with the microcontroller.
    



Links to helpful videos:
But what is a neural network? | Chapter 1, Deep learning by 3Blue1Brown<br />
But what is a convolution? by 3Blue1Brown<br />
But what is the Fourier Transform?  A visual introduction. by 3Blue1Brown<br />

</details>
