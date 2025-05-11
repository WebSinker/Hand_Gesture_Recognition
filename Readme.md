# Hand Gesture Control System

A computer vision-based hand gesture recognition system that allows you to control your computer using hand gestures. This system uses a webcam to track your hand movements and converts them into mouse actions.

## Features

- **Gesture Recognition**: Detects and interprets various hand gestures
- **Mouse Control**: Translates hand gestures into mouse movements and clicks
- **Adaptable Sensitivity**: Adjusts tracking sensitivity based on movement speed
- **Calibration**: Can be personalized to your hand for improved accuracy
- **Comprehensive Dashboard**: Real-time display of system status and gesture information

## Supported Gestures

- **Peace Sign**: Toggle mouse control on/off
- **Point** (index finger extended): Move cursor
- **Pinch** (thumb and index finger together): Click
- **Pinch and Hold** (for 3 seconds): Activate drag mode
- **Open Hand**: Right-click

## Requirements

- Python 3.8 or newer
- Webcam with reasonable quality
- Decent lighting conditions
- See `requirements.txt` for Python package dependencies

## Installation

1. Clone this repository or download the source code:
   ```
   git clone https://github.com/yourusername/hand-gesture-control.git
   cd hand-gesture-control
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage Guide

### Step 1: Test Your Camera Capabilities

First, run the camera capabilities test to determine the optimal settings for your webcam:

```
python camera-capabilities.py
```

This will:
- Check your camera's supported resolutions
- Measure actual FPS performance
- Analyze image quality
- Provide recommended settings for hand tracking

Take note of the recommended resolution and other settings from the output.

### Step 2: Adjust Parameters (Optional)

If the camera test suggests different settings than the defaults, open `hand-gesture-controller.py` and modify the following sections:

1. In the `run` method, find and update the resolution settings:
   ```python
   # Set optimal camera resolution based on testing
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Change this value
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Change this value
   ```

2. Make the same changes in the `safe_run` method.

3. If needed, adjust the MediaPipe model complexity in the `__init__` method:
   ```python
   model_complexity=1  # 0=Fastest, 1=Balanced, 2=Most accurate
   ```

### Step 3: Run the Gesture Control System

Start the hand gesture control system:

```
python hand-gesture-controller.py
```

You'll be asked if you want to calibrate the system. Calibration is recommended for the best performance:

- If you choose 'y', follow the on-screen instructions to make each gesture when prompted
- If you choose 'n', the system will use generic gesture recognition

### Step 4: Using the System

1. Make a **Peace Sign** and hold it for 1.5 seconds to activate mouse control
2. Use your **Index Finger** to move the cursor
3. Make a **Pinch** gesture (touch thumb and index finger) to click
4. Hold a **Pinch** for 3 seconds to enter drag mode
5. Show an **Open Hand** (all fingers extended) to right-click
6. Press **Q** on your keyboard to exit the program

## Optimization Tips

- **Lighting**: Ensure your hand is well-lit with diffuse lighting
- **Background**: Use a plain background that contrasts with your skin tone
- **Distance**: Keep your hand 20-60 cm from the camera
- **Calibration**: For the best accuracy, calibrate the system to your hand
- **Practice**: It may take some time to get used to the gestures

## Troubleshooting

- **Poor Detection**: Try adjusting lighting, using a simpler background, or re-calibrating
- **Lag/Low FPS**: Lower the camera resolution or reduce the model complexity
- **Accidental Clicks**: Try keeping your hand more stable during movement
- **System Not Responding**: Ensure the webcam is properly connected and not being used by another application

## Advanced Customization

You can modify the code to:
- Add new gestures
- Change the gesture-to-action mapping
- Adjust sensitivity and smoothing parameters
- Create custom visualization options

## License

[MIT License](LICENSE)

## Acknowledgments

- [MediaPipe](https://github.com/google/mediapipe) for hand landmark detection
- [OpenCV](https://opencv.org/) for computer vision functionality
- [PyAutoGUI](https://pypi.org/project/PyAutoGUI/) for controlling the mouse