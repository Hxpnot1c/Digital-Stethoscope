![Calday Robotics Logo](https://github.com/Hxpnot1c/Digital-Stethoscope/blob/main/docs/calday_robotics_logo.png)

--------------------------------------------------------------------------------

# Digital-Stethoscope

## Video Presentation

[![Substitutionary Rick Roll to be replaced with presentation video](https://img.youtube.com/vi/dQw4w9WgXcQ/0.jpg)](https://www.youtube.com/watch?v=dQw4w9WgXcQ)

## Video Demonstration

[![Substitutionary Rick Roll to be replaced with demonstration video](https://img.youtube.com/vi/dQw4w9WgXcQ/0.jpg)](https://www.youtube.com/watch?v=dQw4w9WgXcQ)

## Summary
<details>
    <summary>500 word summary</summary>
    Insert 500 word summary here
</details>

 
## Build Guide

Print all 3D models in 'Digital-Stethoscope/3d_models':<br />
-  PLA_Stethoscope_Head.stl should be printed in PLA with 100% infil<br />
-  PLA_Stethoscope_Ring.stl should be printed in PLA with 100% infill<br />
-  TPU_Stethoscope_Connection.stl should be printed in TPU with 100% infill

Note: Our models were printed using a Creality CR10 Smart Pro 3D printer. Most FDM 3D printers will be sufficient.<br />

We also used the default setting on Cura slicer (with 100% infill).<br />
For Stethoscope Ring: Supports ON - Normal<br />

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
Connect V+ on the MAX9814 microphone to 3.3V on the Raspberry Pi using pin pin 17<br />
Connect OUT on the MAX9814 microphone to CH0 on the MCP3008 ADC<br />
![Raspberry Pi pinout](https://www.raspberrypi.com/documentation/computers/images/GPIO-Pinout-Diagram-2.png)
![MCP3008 ADC pinout](https://cdn-learn.adafruit.com/assets/assets/000/001/222/medium800/raspberry_pi_mcp3008pin.gif)
![MAX9814 microphone pinout](https://pmdway.com/cdn/shop/products/Electret-Microphone-Amplifier-MAX9814-Auto-Gain-Control-pmdway-3_708x408.jpg)
