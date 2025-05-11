import cv2
import time
import numpy as np

def check_camera_capabilities(camera_index=0):
    """Check and display camera capabilities."""
    print("\n===== CAMERA CAPABILITIES CHECKER =====\n")
    
    # Try to open the camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera at index {camera_index}")
        return
    
    print(f"Successfully connected to camera at index {camera_index}\n")
    
    # Check available camera properties
    print("===== CAMERA PROPERTIES =====")
    properties = [
        (cv2.CAP_PROP_FRAME_WIDTH, "Width"),
        (cv2.CAP_PROP_FRAME_HEIGHT, "Height"),
        (cv2.CAP_PROP_FPS, "FPS"),
        (cv2.CAP_PROP_BRIGHTNESS, "Brightness"),
        (cv2.CAP_PROP_CONTRAST, "Contrast"),
        (cv2.CAP_PROP_SATURATION, "Saturation"),
        (cv2.CAP_PROP_HUE, "Hue"),
        (cv2.CAP_PROP_GAIN, "Gain"),
        (cv2.CAP_PROP_EXPOSURE, "Exposure"),
        (cv2.CAP_PROP_AUTOFOCUS, "Autofocus"),
        (cv2.CAP_PROP_FOCUS, "Focus"),
        (cv2.CAP_PROP_ZOOM, "Zoom"),
        (cv2.CAP_PROP_BACKLIGHT, "Backlight"),
        (cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, "White Balance Blue"),
        (cv2.CAP_PROP_WHITE_BALANCE_RED_V, "White Balance Red"),
        (cv2.CAP_PROP_BUFFERSIZE, "Buffer Size"),
        (cv2.CAP_PROP_FOURCC, "FOURCC")
    ]
    
    for prop_id, prop_name in properties:
        value = cap.get(prop_id)
        if prop_id == cv2.CAP_PROP_FOURCC:
            # Convert FOURCC code to string
            value = "".join([chr((int(value) >> 8 * i) & 0xFF) for i in range(4)])
        print(f"{prop_name}: {value}")
    
    # Test available resolutions
    print("\n===== TESTING COMMON RESOLUTIONS =====")
    
    resolutions = [
        (320, 240),
        (640, 480),
        (800, 600),
        (1024, 768),
        (1280, 720),
        (1920, 1080),
        (2560, 1440),
        (3840, 2160)
    ]
    
    available_resolutions = []
    
    original_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    original_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    for width, height in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        success = (abs(actual_width - width) < 1) and (abs(actual_height - height) < 1)
        
        if success:
            available_resolutions.append((int(actual_width), int(actual_height)))
            print(f"✅ {width}x{height} - Available")
        else:
            print(f"❌ {width}x{height} - Not available (Got {int(actual_width)}x{int(actual_height)})")
    
    # Restore original resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, original_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, original_height)
    
    # Test FPS
    print("\n===== FPS TEST =====")
    fps_test_duration = 5  # seconds
    
    # Set to highest available resolution
    if available_resolutions:
        best_resolution = max(available_resolutions, key=lambda x: x[0] * x[1])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, best_resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best_resolution[1])
        print(f"Testing at resolution: {best_resolution[0]}x{best_resolution[1]}")
    
    frame_count = 0
    start_time = time.time()
    
    print(f"Measuring FPS for {fps_test_duration} seconds...")
    
    while (time.time() - start_time) < fps_test_duration:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
    
    end_time = time.time()
    measured_fps = frame_count / (end_time - start_time)
    
    print(f"Measured FPS: {measured_fps:.2f}")
    
    # Test image quality
    print("\n===== IMAGE QUALITY ANALYSIS =====")
    ret, frame = cap.read()
    if ret:
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate basic statistics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Perform edge detection to assess sharpness
        edges = cv2.Canny(gray, 100, 200)
        edge_percentage = (np.count_nonzero(edges) / edges.size) * 100
        
        # Calculate contrast
        min_val, max_val, _, _ = cv2.minMaxLoc(gray)
        contrast_ratio = max_val / (min_val + 1e-6)  # Avoid division by zero
        
        print(f"Average Brightness: {mean_brightness:.2f}/255")
        print(f"Brightness Variation (StdDev): {std_brightness:.2f}")
        print(f"Contrast Ratio: {contrast_ratio:.2f}")
        print(f"Edge Detail: {edge_percentage:.2f}%")
        
        # Save sample image
        sample_filename = "camera_sample.jpg"
        cv2.imwrite(sample_filename, frame)
        print(f"Sample image saved as '{sample_filename}'")
        
        # Display live view with basic info
        print("\nShowing live view for 10 seconds (press 'q' to quit early)...")
        fps_font = cv2.FONT_HERSHEY_SIMPLEX
        start_time = time.time()
        
        while (time.time() - start_time) < 10:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add info overlay
            cv2.putText(frame, f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}", 
                       (10, 30), fps_font, 0.7, (0, 255, 0), 2)
            
            cv2.putText(frame, f"FPS: {measured_fps:.1f}", 
                       (10, 60), fps_font, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Camera Capabilities Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n===== TEST COMPLETE =====")
    print("Available Resolutions:", available_resolutions)
    
    # General recommendations
    print("\n===== RECOMMENDATIONS =====")
    if measured_fps < 15:
        print("⚠ Low FPS detected. Consider using a lower resolution for better performance.")
    else:
        print("✅ FPS is adequate for real-time applications.")
    
    if mean_brightness < 50:
        print("⚠ Low brightness detected. Improve lighting conditions for better hand detection.")
    elif mean_brightness > 200:
        print("⚠ High brightness detected. Reduce exposure or adjust lighting to prevent overexposure.")
    else:
        print("✅ Brightness levels appear good.")
    
    if std_brightness < 30:
        print("⚠ Low contrast detected. This may affect hand feature detection.")
    else:
        print("✅ Contrast appears adequate.")
    
    if edge_percentage < 5:
        print("⚠ Low level of detail detected. Check focus and lighting.")
    else:
        print("✅ Edge detail appears sufficient for hand tracking.")
    
    # Recommended settings
    print("\n===== RECOMMENDED SETTINGS FOR HAND TRACKING =====")
    
    # Find optimal resolution - prefer 720p if available, or next best option
    preferred_res = (1280, 720)
    if preferred_res in available_resolutions:
        recommended_res = preferred_res
    elif available_resolutions:
        # Choose highest resolution below 1080p, or lowest available if all are above
        suitable_resolutions = [r for r in available_resolutions if r[1] <= 1080]
        if suitable_resolutions:
            recommended_res = max(suitable_resolutions, key=lambda x: x[0] * x[1])
        else:
            recommended_res = min(available_resolutions, key=lambda x: x[0] * x[1])
    else:
        recommended_res = (640, 480)  # Fallback
    
    print(f"Recommended Resolution: {recommended_res[0]}x{recommended_res[1]}")
    print(f"Recommended FPS: {min(30, round(measured_fps))}")
    print("Recommended Model Complexity: 1 (Balanced)")
    
if __name__ == "__main__":
    # You can specify a different camera index if needed
    check_camera_capabilities(0)