# Detailed Setup Guide

This guide provides step-by-step instructions for setting up and optimizing the Hand Gesture Control System.

## System Requirements

- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: Version 3.8 or newer
- **Hardware**: 
  - Webcam (built-in or external)
  - Processor: At least dual-core CPU
  - RAM: Minimum 4GB (8GB recommended)
  - Graphics: Basic integrated graphics are sufficient

## Detailed Installation Steps

### Windows

1. Install Python from [python.org](https://www.python.org/downloads/)
   - Make sure to check "Add Python to PATH" during installation

2. Open Command Prompt and navigate to your project directory:
   ```
   cd path\to\project\folder
   ```

3. Create a virtual environment (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### macOS

1. Install Python using Homebrew (if not already installed):
   ```
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   brew install python
   ```

2. Open Terminal and navigate to your project directory:
   ```
   cd path/to/project/folder
   ```

3. Create a virtual environment (recommended):
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Linux

1. Install Python and pip (if not already installed):
   ```
   sudo apt update
   sudo apt install python3 python3-pip python3-venv
   ```

2. Open Terminal and navigate to your project directory:
   ```
   cd path/to/project/folder
   ```

3. Create a virtual environment (recommended):
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Camera Calibration Process

1. Run the camera capabilities test:
   ```
   python camera-capabilities.py
   ```

2. Analyze the output and note the recommended settings:
   - **Recommended Resolution**: Use this for optimal performance
   - **Recommended FPS**: Indicates what your camera can handle
   - **Recommended Model Complexity**: Usually 1 (balanced) is a good choice

3. If needed, modify the settings in `hand-gesture-controller.py`:
   - Open the file in a text editor
   - Locate the `run` and `safe_run` methods (around lines 615 and 670)
   - Update the resolution settings to match the recommendations:
     ```python
     cap.set(cv2.CAP_PROP_FRAME_WIDTH, YOUR_RECOMMENDED_WIDTH)
     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, YOUR_RECOMMENDED_HEIGHT)
     ```

4. If your camera has poor performance (FPS < 15), consider:
   - Reducing the resolution further
   - Changing model complexity to 0 in the `__init__` method:
     ```python
     model_complexity=0  # Faster but less accurate
     ```

## Hand Gesture Calibration Guide

When you start the system, you'll be asked if you want to calibrate:

1. Choose 'y' to begin calibration
2. Follow the on-screen instructions for each gesture:
   - **Point**: Extend only your index finger, curl other fingers
   - **Open**: Extend all fingers
   - **Pinch**: Touch your thumb and index finger together
   - **Fist**: Close your hand
   - **Peace**: Extend index and middle fingers, curl others

Tips for good calibration:
- Hold each gesture steady
- Position your hand in the center of the frame
- Ensure good lighting
- Try different angles for each sample

## Customizing the System

### Adjusting Gesture Sensitivity

To make the system more or less sensitive to gesture changes, modify:

1. **Smoothing Window**: Higher values make gesture recognition more stable but less responsive
   ```python
   self.smoothing_window = 10  # Increase for more stability, decrease for faster response
   ```

2. **Threshold values**: For changing how gestures are detected
   ```python
   # For example, to make pinch detection more sensitive:
   thumb_index_distance = self.calculate_distance(thumb_tip, index_tip)
   # Change 0.05 to a smaller value for more sensitivity or larger for less
   elif thumb_index_distance < 0.05:
       gesture = "Pinch"
   ```

### Changing Click and Drag Behavior

To adjust how long to hold for drag:
```python
# If we've been holding the pinch for more than 3 seconds, start dragging
# Change 3.0 to a different value to make drag activation faster or slower
if pinch_duration > 3.0 and not self.is_dragging:
```

### Adding New Gestures

To add a new gesture:

1. Define the detection logic in the `recognize_gesture` method
2. Add the gesture to the calibration list in `calibrate_hand`
3. Implement the action in `map_gesture_to_action`
4. Update the dashboard display in `create_dashboard`

## Optimization for Different Environments

### Low Light Conditions
- Increase the brightness adjustment in `preprocess_frame`:
  ```python
  beta = 10  # Increase from 5 to brighten the image more
  ```

### High Contrast Backgrounds
- Adjust the skin detection thresholds:
  ```python
  lower_skin = np.array([0, 135, 85], dtype=np.uint8)  # Experiment with these values
  upper_skin = np.array([255, 180, 135], dtype=np.uint8)
  ```

### Reducing CPU Usage
- Optimize the processing pipeline:
  ```python
  # Use a smaller bilateral filter kernel
  smoothed = cv2.bilateralFilter(frame, 5, 75, 75)  # Changed from 9 to 5
  
  # Skip some frames
  if frame_count % 2 != 0:  # Process every other frame
      continue
  ```

## Troubleshooting Common Issues

### Unable to Detect Hand
- Check lighting conditions
- Make sure your hand is within the camera's field of view
- Try recalibrating the system
- Adjust the skin color thresholds

### System Running Slowly
- Lower the camera resolution
- Reduce the model complexity
- Close other CPU-intensive applications
- Check if your webcam supports hardware acceleration

### Gestures Not Being Recognized Correctly
- Try recalibrating with more deliberate gestures
- Increase the smoothing window for more stable recognition
- Check for interference in the background
- Make sure there's good contrast between your hand and the background

### Mouse Movements Too Sensitive/Not Sensitive Enough
- Adjust the sensitivity values in the code:
  ```python
  # For fine movements (increase from 0.6 for less sensitivity)
  sensitivity = 0.6
  
  # For fast movements (decrease from 1.4 for less sensitivity)
  sensitivity = 1.4
  ```