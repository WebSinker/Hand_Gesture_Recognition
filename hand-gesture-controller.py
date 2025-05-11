import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math
import os
import logging
from datetime import datetime
import time


class HandGestureController:
    # Main Initialization Function
    def __init__(self):
        # Initialize MediaPipe Hands with enhanced parameters
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # False for video streams
            max_num_hands=1,          # Only track one hand for simplicity
            min_detection_confidence=0.7,  # Minimum confidence for initial detection
            min_tracking_confidence=0.7,   # Minimum confidence to continue tracking
            model_complexity=1        # Balance between speed and accuracy (0-2)
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.base_width = 640
        self.base_height = 480
        
        # Screen resolution for mouse control
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Gesture smoothing
        self.prev_gesture = None
        self.gesture_smoothing = []
        self.smoothing_window = 10  # Number of frames for temporal smoothing
        
        # Mouse control parameters
        self.prev_index_tip = None
        self.mouse_smoothing = []
        self.mouse_smooth_factor = 15
        
        # Virtual mouse mode
        self.virtual_mouse_active = False
        
        # Flag to check if calibration is done
        self.is_calibrated = False
        
        # Dictionary to store hand pose features for each gesture
        self.calibration_data = {}
        
        # Initialize classifier (will be trained during calibration)
        self.classifier = None

        # These lines for pinch gesture timing
        self.prev_mouse_x = None
        self.prev_mouse_y = None
        self.pinch_start_time = 0
        self.is_dragging = False

    # Setup logging for the system
    def setup_logging(self):
        """Set up logging to track system performance and errors."""
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # Set up logging with timestamp in filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f'logs/gesture_system_{timestamp}.log'
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()  # Also log to console
            ]
        )
        
        self.logger = logging.getLogger('HandGestureSystem')
        self.logger.info("Logging initialized")
        
        # Track performance metrics
        self.fps_values = []
        self.detection_times = []
        self.gesture_recognition_times = []
        
    #Image processing functions
    # Process frame for better hand detection
    def preprocess_frame(self, frame):
        """Preprocess the frame for better hand detection using full frame width."""
        # Original frame dimensions
        h, w = frame.shape[:2]
        
        # 1. Apply bilateral filter to smooth while preserving edges
        smoothed = cv2.bilateralFilter(frame, 9, 75, 75)
        
        # 2. Adjust contrast and brightness
        alpha = 1.3  # Contrast control (1.0 means no change)
        beta = 5     # Brightness control (0 means no change)
        adjusted = cv2.convertScaleAbs(smoothed, alpha=alpha, beta=beta)
        
        # 3. Color correction for better skin detection
        # Convert to YCrCb color space
        ycrcb = cv2.cvtColor(adjusted, cv2.COLOR_BGR2YCrCb)
        
        # 4. Apply skin color thresholding
        # Define skin color range in YCrCb space
        lower_skin = np.array([0, 135, 85], dtype=np.uint8)
        upper_skin = np.array([255, 180, 135], dtype=np.uint8)
        
        # Create a mask for skin color
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
        
        # Store the skin mask for visualization
        self.skin_mask = skin_mask
        
        # 5. Apply full-frame enhancement instead of using ROI
        # Apply adaptive histogram equalization for better contrast to the full adjusted frame
        # Convert to LAB color space
        lab = cv2.cvtColor(adjusted, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge channels
        merged = cv2.merge((cl, a, b))
        
        # Convert back to BGR
        enhanced_frame = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        
        # Store the processed frame for visualization
        self.processed_frame = enhanced_frame
        
        # Return the full enhanced frame for detection and the original adjusted frame
        return enhanced_frame, adjusted

    # Detect hands in the frame
    def detect_hands(self, frame):
        """Process frame and detect hands."""
        # Convert BGR to RGB (MediaPipe requires RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to find hands
        # Removing the Image class usage that's causing issues
        results = self.hands.process(rgb_frame)
        
        return results

    # Draw hand landmarks on the frame
    def draw_landmarks(self, frame, results):
        """Draw hand landmarks and connections on the frame."""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
        return frame

    # Create a mask of the hand region
    def segment_hand(self, frame, results):
        """Create a mask highlighting the hand region."""
        h, w, c = frame.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract coordinates of hand landmarks
                points = []
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    points.append((x, y))
                
                # Create convex hull around hand
                if len(points) > 3:  # Need at least 4 points for convex hull
                    hull = cv2.convexHull(np.array(points))
                    # Fill the hand region
                    cv2.fillConvexPoly(mask, hull, 255)
        
        # Apply mask to original frame
        segmented_hand = cv2.bitwise_and(frame, frame, mask=mask)
        return segmented_hand, mask
    
    #Gesture recognition functions
    # Recognize hand gestures based on landmarks
    def recognize_gesture(self, hand_landmarks):
        """Recognize hand gestures with improved accuracy for Point vs Fist distinction."""
        if not hand_landmarks:
            return "No Hand"
        
        # Extract keypoints
        landmarks = hand_landmarks.landmark
        
        # Get fingertip and base positions
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[self.mp_hands.HandLandmark.THUMB_IP]
        thumb_mcp = landmarks[self.mp_hands.HandLandmark.THUMB_MCP]
        
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_dip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_DIP]
        index_pip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        index_mcp = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        
        middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_dip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
        middle_pip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        
        ring_tip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_pip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_PIP]
        
        pinky_tip = landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = landmarks[self.mp_hands.HandLandmark.PINKY_PIP]
        
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        
        # For Point gesture, we need a more strict check to ensure index is clearly extended
        # and other fingers are clearly curled
        index_extended = (index_tip.y < index_pip.y) and (index_tip.y < index_dip.y - 0.02)
        
        # Other fingers should be more clearly curled
        middle_curled = middle_tip.y > middle_pip.y + 0.01
        ring_curled = ring_tip.y > ring_pip.y + 0.01
        pinky_curled = pinky_tip.y > pinky_pip.y + 0.01
        
        # Calculate distance for pinch detection
        thumb_index_distance = self.calculate_distance(thumb_tip, index_tip)
        
        # Point: Only index finger clearly extended, others clearly curled
        if index_extended and middle_curled and ring_curled and pinky_curled:
            gesture = "Point"
        
        # Open hand: All fingers extended
        elif (index_tip.y < index_pip.y and
            middle_tip.y < middle_pip.y and
            ring_tip.y < ring_pip.y and
            pinky_tip.y < pinky_pip.y):
            gesture = "Open"
        
        # Pinch: Thumb and index fingertips close
        elif thumb_index_distance < 0.05:
            gesture = "Pinch"
        
        # Peace: Index and middle fingers extended, others curled
        elif (index_tip.y < index_pip.y and
            middle_tip.y < middle_pip.y and
            ring_tip.y > ring_pip.y and
            pinky_tip.y > pinky_pip.y):
            gesture = "Peace"
        
        # We no longer use "Fist" as a specific gesture to avoid confusion with Point
        else:
            gesture = "Unknown"
        
        # Apply temporal smoothing
        self.gesture_smoothing.append(gesture)
        if len(self.gesture_smoothing) > self.smoothing_window:
            self.gesture_smoothing.pop(0)
        
        # Return the most common gesture in the window
        if self.gesture_smoothing:
            from collections import Counter
            return Counter(self.gesture_smoothing).most_common(1)[0][0]
        
        return gesture

    # Calculate distance between landmarks
    def calculate_distance(self, landmark1, landmark2):
        """Calculate normalized distance between two landmarks."""
        return math.sqrt((landmark1.x - landmark2.x)**2 + 
                        (landmark1.y - landmark2.y)**2)

    # Calculate angle between three points
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points in degrees."""
        # Convert landmarks to numpy arrays
        p1 = np.array([point1.x, point1.y])
        p2 = np.array([point2.x, point2.y])
        p3 = np.array([point3.x, point3.y])
        
        # Calculate vectors
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Calculate angle
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Ensure value is in valid range
        
        # Return angle in degrees
        return np.degrees(np.arccos(cosine_angle))

    # Extract features for gesture recognition
    def extract_gesture_features(self, landmarks):
        """Extract relevant features from hand landmarks for gesture classification."""
        # This function extracts features that will help distinguish between gestures
        features = []
        
        # 1. Extract relative positions of fingertips to wrist
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        
        for landmark_id in [
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]:
            tip = landmarks[landmark_id]
            # Add relative position (normalized)
            features.append(tip.x - wrist.x)
            features.append(tip.y - wrist.y)
            features.append(tip.z - wrist.z)
        
        # 2. Add angles between finger segments
        # Thumb angle
        features.append(self.calculate_angle(
            landmarks[self.mp_hands.HandLandmark.THUMB_MCP],
            landmarks[self.mp_hands.HandLandmark.THUMB_IP],
            landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        ))
        
        # Add angles for each finger (PIP joint angle)
        for mcp, pip, dip in [
            (self.mp_hands.HandLandmark.INDEX_FINGER_MCP, self.mp_hands.HandLandmark.INDEX_FINGER_PIP, self.mp_hands.HandLandmark.INDEX_FINGER_DIP),
            (self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP, self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP, self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP),
            (self.mp_hands.HandLandmark.RING_FINGER_MCP, self.mp_hands.HandLandmark.RING_FINGER_PIP, self.mp_hands.HandLandmark.RING_FINGER_DIP),
            (self.mp_hands.HandLandmark.PINKY_MCP, self.mp_hands.HandLandmark.PINKY_PIP, self.mp_hands.HandLandmark.PINKY_DIP)
        ]:
            features.append(self.calculate_angle(
                landmarks[mcp],
                landmarks[pip],
                landmarks[dip]
            ))
        
        # 3. Add distances between fingertips
        for i in range(4):
            for j in range(i + 1, 5):
                tip_landmarks = [
                    landmarks[self.mp_hands.HandLandmark.THUMB_TIP],
                    landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
                    landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                    landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP],
                    landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
                ]
                features.append(self.calculate_distance(tip_landmarks[i], tip_landmarks[j]))
        
        return features

    # Recognize gestures using ML model
    def recognize_gesture_with_ml(self, hand_landmarks):
        """Recognize gestures using the trained ML classifier."""
        if not hand_landmarks or not hasattr(self, 'classifier'):
            # Fall back to rule-based method if classifier not available
            return self.recognize_gesture(hand_landmarks)
        
        # Extract features from current hand landmarks
        landmarks = hand_landmarks.landmark
        features = self.extract_gesture_features(landmarks)
        
        if not features:
            return "Unknown"
        
        # Predict gesture using classifier
        prediction = self.classifier.predict([features])[0]
        
        # Apply temporal smoothing
        self.gesture_smoothing.append(prediction)
        if len(self.gesture_smoothing) > self.smoothing_window:
            self.gesture_smoothing.pop(0)
        
        # Return the most common gesture in the window
        if self.gesture_smoothing:
            from collections import Counter
            return Counter(self.gesture_smoothing).most_common(1)[0][0]
        
        return prediction
    
    #Callibration functions
    # Calibrate the system to user's hand
    def calibrate_hand(self):
        """Calibrate the system to the user's hand for more accurate gesture recognition."""
        print("Starting hand calibration...")
        print("Please follow the instructions on screen.")
        
        cap = cv2.VideoCapture(0)
        
        # Set camera properties for calibration
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Calibration data
        calibration_gestures = ["Point", "Open", "Pinch", "Peace"]  # Removed "Fist" as we're not using it
        calibration_data = {}
        
        # For each gesture, collect sample data
        for gesture in calibration_gestures:
            print(f"\nPlease make a {gesture} gesture and hold steady...")
            # Countdown
            for i in range(3, 0, -1):
                print(f"{i}...")
                time.sleep(1)
            
            samples = []
            sample_count = 0
            
            # Collect 20 samples
            while sample_count < 20:
                success, frame = cap.read()
                if not success:
                    continue
                    
                # Flip horizontally
                frame = cv2.flip(frame, 1)
                
                # Add text instruction
                cv2.putText(frame, f"Make {gesture} gesture ({sample_count}/20)", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Process the frame - use direct preprocessing
                enhanced_frame, _ = self.preprocess_frame(frame)
                
                # Process the frame to detect hands
                results = self.detect_hands(enhanced_frame)
                
                # Draw landmarks
                frame = self.draw_landmarks(frame, results)
                
                if results.multi_hand_landmarks:
                    # Extract feature vector from hand landmarks
                    landmarks = results.multi_hand_landmarks[0].landmark
                    features = self.extract_gesture_features(landmarks)
                    
                    if features:
                        samples.append(features)
                        sample_count += 1
                        print(f"Sample {sample_count}/20 captured")
                
                cv2.imshow("Calibration", frame)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                    
                time.sleep(0.1)  # Small delay between samples
            
            # Store calibration data for this gesture
            if samples:
                calibration_data[gesture] = samples
                print(f"Calibration data for {gesture} collected.")
        
        cv2.destroyWindow("Calibration")
        cap.release()
        
        # Save calibration data
        self.calibration_data = calibration_data
        
        # Train a classifier with calibration data
        self.train_gesture_classifier()
        
        print("Calibration complete!")
        return True

    # Train a machine learning classifier using calibration data
    def train_gesture_classifier(self):
        """Train a classifier using the collected calibration data."""
        if not hasattr(self, 'calibration_data') or not self.calibration_data:
            print("No calibration data available.")
            return False
        
        # Prepare training data
        X = []  # Features
        y = []  # Labels (gestures)
        
        for gesture, samples in self.calibration_data.items():
            for sample in samples:
                X.append(sample)
                y.append(gesture)
        
        # Use a Random Forest classifier
        from sklearn.ensemble import RandomForestClassifier
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X, y)
        
        print(f"Classifier trained with {len(X)} samples.")
        self.is_calibrated = True
        return True
    
    #Action Mapping functions
    # Map recognized gestures to actions
    def map_gesture_to_action(self, gesture, hand_landmarks, frame_shape):
        """Map recognized gestures to mouse actions with Pinch for click and long Pinch for drag."""
        if not hand_landmarks:
            return
        
        h, w = frame_shape[:2]
        landmarks = hand_landmarks.landmark
        
        # Extract key points
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
        
        # Initialize variables if they don't exist
        if not hasattr(self, 'prev_mouse_x') or self.prev_mouse_x is None:
            self.prev_mouse_x = np.interp(index_x, [0, w], [0, self.screen_width])
        
        if not hasattr(self, 'prev_mouse_y') or self.prev_mouse_y is None:
            self.prev_mouse_y = np.interp(index_y, [0, h], [0, self.screen_height])
        
        if not hasattr(self, 'is_dragging'):
            self.is_dragging = False
        
        if not hasattr(self, 'pinch_start_time'):
            self.pinch_start_time = 0
        
        if not hasattr(self, 'peace_toggled'):
            self.peace_toggled = False
        
        if not hasattr(self, 'peace_start_time'):
            self.peace_start_time = 0
        
        if not hasattr(self, 'click_debounce_time'):
            self.click_debounce_time = 0
        
        # Calculate hand velocity for dynamic sensitivity adjustment
        if self.prev_index_tip is not None:
            prev_x, prev_y = self.prev_index_tip
            velocity = math.sqrt((index_x - prev_x)**2 + (index_y - prev_y)**2)
            
            # Adjust mouse sensitivity based on velocity
            slow_threshold = 5
            fast_threshold = 30
            
            if velocity < slow_threshold:
                sensitivity = 0.6  # More precise for fine movements
            elif velocity > fast_threshold:
                sensitivity = 1.4  # Cover more distance for fast movements
            else:
                # Linear interpolation between slow and fast
                sensitivity = 0.6 + (1.4 - 0.6) * ((velocity - slow_threshold) / 
                                                (fast_threshold - slow_threshold))
        else:
            sensitivity = 1.0
        
        # Store current position for next frame
        self.prev_index_tip = (index_x, index_y)
        
        # Detect gesture transitions
        gesture_changed = (gesture != self.prev_gesture)
        
        # Toggle virtual mouse control with "Peace" gesture + hold timer
        if gesture == "Peace":
            if self.prev_gesture != "Peace":
                self.peace_start_time = time.time()
            
            # Toggle after holding peace sign for 1.5 seconds
            if time.time() - self.peace_start_time > 1.5:
                if not self.peace_toggled:
                    self.virtual_mouse_active = not self.virtual_mouse_active
                    self.peace_toggled = True
                    
                    # Haptic/visual feedback
                    if self.virtual_mouse_active:
                        self.logger.info("Virtual mouse activated")
                        try:
                            pyautogui.press('numlock')  # Toggle numlock as basic feedback
                            pyautogui.press('numlock')  # Toggle it back
                        except:
                            pass
                    else:
                        self.logger.info("Virtual mouse deactivated")
        else:
            # Reset peace sign timer when gesture changes
            self.peace_toggled = False
        
        # Virtual mouse control when active
        if self.virtual_mouse_active:
            # Point gesture: Move mouse with adaptive acceleration
            if gesture == "Point":
                # Convert hand position to screen coordinates with adaptive sensitivity
                raw_mouse_x = np.interp(index_x, [0, w], [0, self.screen_width])
                raw_mouse_y = np.interp(index_y, [0, h], [0, self.screen_height])
                
                # Apply sensitivity multiplier based on velocity
                mouse_x = self.prev_mouse_x + ((raw_mouse_x - self.prev_mouse_x) * sensitivity)
                mouse_y = self.prev_mouse_y + ((raw_mouse_y - self.prev_mouse_y) * sensitivity)
                
                # Store for next frame
                self.prev_mouse_x = mouse_x
                self.prev_mouse_y = mouse_y
                
                # Apply smoothing for more natural movement
                self.mouse_smoothing.append((mouse_x, mouse_y))
                if len(self.mouse_smoothing) > self.mouse_smooth_factor:
                    self.mouse_smoothing.pop(0)
                
                # Calculate average position with more weight to recent positions
                if self.mouse_smoothing:
                    # Weighted average (recent positions have more influence)
                    weights = np.linspace(0.5, 1.0, len(self.mouse_smoothing))
                    weights = weights / np.sum(weights)  # Normalize weights
                    
                    avg_x = sum(x * w for (x, _), w in zip(self.mouse_smoothing, weights))
                    avg_y = sum(y * w for (_, y), w in zip(self.mouse_smoothing, weights))
                    
                    # Move mouse
                    try:
                        pyautogui.moveTo(avg_x, avg_y)
                    except:
                        self.logger.error("Failed to move mouse")
            
            # Pinch gesture: Click immediately, drag after 3 seconds
            elif gesture == "Pinch":
                # When gesture first changes to Pinch
                if gesture_changed:
                    # Store the time we started the pinch
                    self.pinch_start_time = time.time()
                    
                    # Perform a click immediately when pinch is detected
                    # Only if not already in dragging mode
                    if not self.is_dragging and time.time() - self.click_debounce_time > 0.5:
                        try:
                            pyautogui.click()
                            self.logger.info("Click")
                            self.click_debounce_time = time.time()  # Prevent rapid clicks
                        except:
                            self.logger.error("Failed to click")
                
                # Calculate how long we've been holding the pinch
                pinch_duration = time.time() - self.pinch_start_time
                
                # If we've been holding the pinch for more than 3 seconds, start dragging
                if pinch_duration > 3.0 and not self.is_dragging:
                    try:
                        # Start drag operation
                        pyautogui.mouseDown()
                        self.logger.info("Mouse Down - Starting Drag after 3-second hold")
                        self.is_dragging = True
                    except:
                        self.logger.error("Failed to start drag")
                
                # If we're dragging, move the mouse
                if self.is_dragging:
                    # Continue moving while dragging with smoothing
                    raw_mouse_x = np.interp(index_x, [0, w], [0, self.screen_width])
                    raw_mouse_y = np.interp(index_y, [0, h], [0, self.screen_height])
                    
                    # Apply lower sensitivity for more precise dragging
                    drag_sensitivity = 0.7
                    
                    mouse_x = self.prev_mouse_x + ((raw_mouse_x - self.prev_mouse_x) * drag_sensitivity)
                    mouse_y = self.prev_mouse_y + ((raw_mouse_y - self.prev_mouse_y) * drag_sensitivity)
                    
                    try:
                        pyautogui.moveTo(mouse_x, mouse_y)
                        self.prev_mouse_x = mouse_x
                        self.prev_mouse_y = mouse_y
                    except:
                        self.logger.error("Failed to drag")
                
                # If not dragging, still allow movement during pinch for precise cursor positioning
                elif not self.is_dragging:
                    raw_mouse_x = np.interp(index_x, [0, w], [0, self.screen_width])
                    raw_mouse_y = np.interp(index_y, [0, h], [0, self.screen_height])
                    
                    # Apply lower sensitivity for more precise positioning
                    pinch_sensitivity = 0.5
                    
                    mouse_x = self.prev_mouse_x + ((raw_mouse_x - self.prev_mouse_x) * pinch_sensitivity)
                    mouse_y = self.prev_mouse_y + ((raw_mouse_y - self.prev_mouse_y) * pinch_sensitivity)
                    
                    self.prev_mouse_x = mouse_x
                    self.prev_mouse_y = mouse_y
            
            # If we were dragging and changed from Pinch to another gesture, release drag
            elif self.prev_gesture == "Pinch" and self.is_dragging:
                try:
                    pyautogui.mouseUp()
                    self.logger.info("Mouse Up - Ending Drag")
                    self.is_dragging = False
                except:
                    self.logger.error("Failed to release drag")
            
            # Open hand: Right click
            elif gesture == "Open" and gesture_changed:
                try:
                    pyautogui.rightClick()
                    self.logger.info("Right Click")
                except:
                    self.logger.error("Failed to right click")
        
        # Store gesture for next frame comparison
        self.prev_gesture = gesture
        
    #Diagnostic and Performance functions
    # Log performance metrics
    def log_performance(self, start_time, detection_time, recognition_time, current_fps):
        """Log performance metrics."""
        self.fps_values.append(current_fps)
        self.detection_times.append(detection_time)
        self.gesture_recognition_times.append(recognition_time)
        
        # Log every 100 frames
        if len(self.fps_values) % 100 == 0:
            avg_fps = sum(self.fps_values[-100:]) / 100
            avg_detection = sum(self.detection_times[-100:]) / 100
            avg_recognition = sum(self.gesture_recognition_times[-100:]) / 100
            
            self.logger.info(f"Performance: FPS={avg_fps:.2f}, " 
                            f"Detection={avg_detection:.4f}s, "
                            f"Recognition={avg_recognition:.4f}s")

    # Run diagnostic tests
    def diagnostic_mode(self):
        """Run the system in diagnostic mode to identify issues."""
        self.logger.info("Starting diagnostic mode")
        
        # Check if camera is working
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.logger.error("Failed to open camera. Check connection or permissions.")
            return False
        
        # Check camera resolution
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.logger.info(f"Camera resolution: {width}x{height}")
        
        # Check if MediaPipe is working
        success, frame = cap.read()
        if not success:
            self.logger.error("Failed to read frame from camera.")
            cap.release()
            return False
        
        # Test MediaPipe hand detection
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            self.logger.info(f"MediaPipe test: {'Successful' if results is not None else 'Failed'}")
        except Exception as e:
            self.logger.error(f"MediaPipe error: {str(e)}")
            cap.release()
            return False
        
        # Check lighting conditions
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        self.logger.info(f"Average frame brightness: {brightness:.2f}/255")
        
        if brightness < 50:
            self.logger.warning("Low lighting detected. This may reduce hand detection accuracy.")
        elif brightness > 200:
            self.logger.warning("Very bright lighting detected. This may cause overexposure.")
        
        # Test frame preprocessing
        try:
            enhanced_frame, roi = self.preprocess_frame(frame)
            self.logger.info("Frame preprocessing test: Successful")
        except Exception as e:
            self.logger.error(f"Preprocessing error: {str(e)}")
        
        # Test skin detection
        if hasattr(self, 'skin_mask'):
            skin_pixel_percentage = (np.count_nonzero(self.skin_mask) / (self.skin_mask.shape[0] * self.skin_mask.shape[1])) * 100
            self.logger.info(f"Skin detection: {skin_pixel_percentage:.2f}% of frame identified as skin")
            
            if skin_pixel_percentage < 5:
                self.logger.warning("Very little skin detected. Check camera positioning and lighting.")
            elif skin_pixel_percentage > 60:
                self.logger.warning("Too much skin detected. Background may be interfering with detection.")
        
        # Test ML classifier if available
        if self.is_calibrated and self.classifier is not None:
            try:
                # Check if we can extract features and make prediction
                if results.multi_hand_landmarks:
                    landmarks = results.multi_hand_landmarks[0].landmark
                    features = self.extract_gesture_features(landmarks)
                    prediction = self.classifier.predict([features])[0]
                    self.logger.info(f"ML classifier test: Successful (predicted '{prediction}')")
            except Exception as e:
                self.logger.error(f"ML classifier error: {str(e)}")
        
        # Clean up
        cap.release()
        self.logger.info("Diagnostic test completed")
        return True

    # Create dashboard UI
    def create_dashboard(self, frame, results, gesture, fps, confidence=None):
        """Create a comprehensive status dashboard overlay for real-time monitoring."""
        h, w = frame.shape[:2]
        dashboard_height = 180
        dashboard_width = w
        
        # Create a semi-transparent dashboard overlay
        dashboard = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8)
        dashboard[:, :] = (30, 30, 30)  # Dark background
        
        # Add system status section
        cv2.putText(dashboard, "Hand Gesture Control System", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
        
        # Display FPS with color coding
        fps_color = (0, 255, 0) if fps > 25 else (0, 165, 255) if fps > 15 else (0, 0, 255)
        cv2.putText(dashboard, f"FPS: {fps:.1f}", (10, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
        
        # Display current gesture with highlight
        gesture_color = (0, 255, 255)  # Yellow highlight for current gesture
        cv2.putText(dashboard, f"Gesture: {gesture}", (10, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, gesture_color, 2)
        
        # Display ML confidence if available
        if confidence is not None:
            conf_color = (0, 255, 0) if confidence > 85 else (0, 165, 255) if confidence > 70 else (0, 0, 255)
            cv2.putText(dashboard, f"Confidence: {confidence:.1f}%", (10, 115), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)
        
        # Display virtual mouse status
        mouse_status = "ON" if self.virtual_mouse_active else "OFF"
        mouse_color = (0, 255, 0) if self.virtual_mouse_active else (0, 0, 255)
        cv2.putText(dashboard, f"Virtual Mouse: {mouse_status}", (10, 145), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, mouse_color, 2)
        
        # Add hand detection status on right side
        hand_status = "Detected" if results.multi_hand_landmarks else "Not Detected"
        hand_color = (0, 255, 0) if results.multi_hand_landmarks else (0, 0, 255)
        cv2.putText(dashboard, f"Hand: {hand_status}", (w - 200, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)
        
        # Add gesture mapping guide
        cv2.putText(dashboard, "Gesture Map:", (w - 200, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(dashboard, "Peace: Toggle Mouse", (w - 190, 105), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150) 
                    if gesture != "Peace" else (0, 255, 255), 1)
        cv2.putText(dashboard, "Point: Move Cursor", (w - 190, 125), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150) 
                    if gesture != "Point" else (0, 255, 255), 1)
        cv2.putText(dashboard, "Pinch: Click", (w - 190, 145), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150) 
                    if gesture != "Pinch" or self.is_dragging else (0, 255, 255), 1)
        cv2.putText(dashboard, "Pinch (3s hold): Drag", (w - 190, 165), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150) 
                    if not (gesture == "Pinch" and self.is_dragging) else (0, 255, 255), 1)
        
        # Create final output by combining frame and dashboard
        output = frame.copy()
        
        # Add dashboard at the bottom of the frame
        if h > dashboard_height:
            output[h-dashboard_height:h, 0:dashboard_width] = dashboard
        
        return output
    
    #Main execution function
    # Main run loop
    def run(self):
        """Main loop to run the hand gesture controller."""
        cap = cv2.VideoCapture(0)
        
        # Set optimal camera resolution based on testing
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Offer calibration option
        print("Do you want to calibrate the system for your hand? (y/n)")
        choice = input().lower()
        if choice == 'y':
            self.calibrate_hand()
        
        # Main processing loop
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read from webcam")
                break
            
            # Flip horizontally for a more intuitive mirror view
            frame = cv2.flip(frame, 1)
            
            # Preprocess the frame
            enhanced_frame, roi = self.preprocess_frame(frame)
            
            # Detect hands
            results = self.detect_hands(enhanced_frame)
            
            # Draw landmarks
            frame = self.draw_landmarks(frame, results)
            
            # Segment hand
            segmented_hand, mask = self.segment_hand(frame, results)
            
            # Recognize gestures - use ML model if calibrated, otherwise rule-based
            gesture = "No Hand"
            if results.multi_hand_landmarks:
                if self.is_calibrated and self.classifier is not None:
                    gesture = self.recognize_gesture_with_ml(results.multi_hand_landmarks[0])
                else:
                    gesture = self.recognize_gesture(results.multi_hand_landmarks[0])
            
            # Map gestures to actions
            self.map_gesture_to_action(gesture, 
                                    results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None,
                                    frame.shape)
            
            # Display gesture information on frame
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display virtual mouse status
            mouse_status = "ON" if self.virtual_mouse_active else "OFF"
            cv2.putText(frame, f"Virtual Mouse: {mouse_status}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the processed frames
            cv2.imshow("Original", frame)
            cv2.imshow("Hand Segmentation", segmented_hand)
            
            # Exit on 'q' key
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

    # Run with error handling
    def safe_run(self):
        """Run the system with error handling."""
        try:
            self.setup_logging()
            self.logger.info("Starting Hand Gesture Recognition System")
            
            # Run diagnostic mode first
            self.diagnostic_mode()
            
            # Prompt for calibration before starting the main loop
            print("\nDo you want to calibrate the system for your hand? (y/n)")
            choice = input().lower()
            if choice == 'y':
                try:
                    self.calibrate_hand()
                except Exception as e:
                    self.logger.error(f"Calibration error: {str(e)}")
                    print(f"Error during calibration: {e}")
            
            # Main processing loop with error handling
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.logger.error("Could not open webcam. Exiting.")
                return
            
            # Set optimal camera resolution based on testing
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            frame_count = 0
            start_time = time.time()
            
            while cap.isOpened():
                try:
                    # Measure FPS
                    frame_count += 1
                    if frame_count % 30 == 0:
                        elapsed_time = time.time() - start_time
                        fps = frame_count / elapsed_time
                        # Reset for next measurement
                        frame_count = 0
                        start_time = time.time()
                    else:
                        fps = 0  # Not calculating this frame
                    
                    # Read frame with timeout
                    success, frame = cap.read()
                    if not success:
                        self.logger.warning("Failed to read frame from webcam")
                        # Try to recover
                        cap.release()
                        time.sleep(1)  # Wait before trying to reconnect
                        cap = cv2.VideoCapture(0)
                        continue
                    
                    # Flip horizontally
                    frame = cv2.flip(frame, 1)
                    
                    # Preprocess frame
                    try:
                        enhanced_frame, roi = self.preprocess_frame(frame)
                    except Exception as e:
                        self.logger.error(f"Frame preprocessing error: {str(e)}")
                        enhanced_frame = frame  # Fall back to original frame
                    
                    # Detect hands
                    detection_start = time.time()
                    results = self.detect_hands(enhanced_frame)
                    detection_time = time.time() - detection_start
                    
                    # Draw landmarks
                    try:
                        frame = self.draw_landmarks(frame, results)
                    except Exception as e:
                        self.logger.error(f"Landmark drawing error: {str(e)}")
                    
                    # Recognize gestures
                    recognition_start = time.time()
                    gesture = "No Hand"
                    confidence = None
                    try:
                        if results.multi_hand_landmarks:
                            if self.is_calibrated and self.classifier is not None:
                                landmarks = results.multi_hand_landmarks[0].landmark
                                features = self.extract_gesture_features(landmarks)
                                gesture = self.recognize_gesture_with_ml(results.multi_hand_landmarks[0])
                                
                                # Get prediction probability (confidence)
                                if hasattr(self, 'classifier') and self.classifier is not None:
                                    probs = self.classifier.predict_proba([features])[0]
                                    confidence = max(probs) * 100
                            else:
                                gesture = self.recognize_gesture(results.multi_hand_landmarks[0])
                    except Exception as e:
                        self.logger.error(f"Gesture recognition error: {str(e)}")
                    recognition_time = time.time() - recognition_start
                    
                    # Log performance metrics if calculating FPS this frame
                    if fps > 0:
                        self.log_performance(start_time, detection_time, recognition_time, fps)
                    
                    # Map gestures to actions
                    try:
                        self.map_gesture_to_action(gesture, 
                                                results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None,
                                                frame.shape)
                    except Exception as e:
                        self.logger.error(f"Action mapping error: {str(e)}")
                    
                    # Create dashboard
                    try:
                        dashboard_frame = self.create_dashboard(frame, results, gesture, 
                                                            fps if fps > 0 else 30,  # Default to 30 if not measured
                                                            confidence)
                        # Display dashboard frame
                        cv2.imshow("Hand Gesture System", dashboard_frame)
                        
                        # Also show segmented hand view
                        segmented_hand, mask = self.segment_hand(frame, results)
                        cv2.imshow("Hand Segmentation", cv2.resize(segmented_hand, (320, 240)))
                        
                        # Show skin mask if available
                        if hasattr(self, 'skin_mask'):
                            skin_mask_display = cv2.cvtColor(
                                cv2.resize(self.skin_mask, (320, 240)), 
                                cv2.COLOR_GRAY2BGR
                            )
                            cv2.imshow("Skin Detection", skin_mask_display)
                            
                    except Exception as e:
                        self.logger.error(f"Dashboard creation error: {str(e)}")
                        # Fall back to basic display
                        cv2.imshow("Hand Gesture System", frame)
                    
                    # Check for exit key
                    if cv2.waitKey(5) & 0xFF == ord('q'):
                        break
                    
                except Exception as e:
                    self.logger.error(f"Unexpected error in main loop: {str(e)}")
                    # Continue on error
            
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            self.logger.info("System shutdown gracefully")
            
        except Exception as e:
            self.logger.critical(f"Critical error: {str(e)}")
            # Ensure cleanup
            try:
                cap.release()
            except:
                pass
            cv2.destroyAllWindows()

# Main execution point
if __name__ == "__main__":
    # Initialize and run the improved hand gesture controller
    controller = HandGestureController()
    print("Starting Improved Hand Gesture Control System")
    print("\nFor optimal hand tracking:")
    print("- Keep your hand 20-60 cm from the camera")
    print("- Ensure good, even lighting on your hand")
    print("- Avoid having similar skin tones in the background")
    print("\nGesture Controls:")
    print("- Make a 'Peace' sign to toggle virtual mouse control")
    print("- Use 'Point' to move cursor")
    print("- Use 'Pinch' for an immediate click")
    print("- Hold 'Pinch' for 3 seconds to enter drag mode")
    print("- Use 'Open Hand' to right-click")
    print("Press 'q' to quit")
    # Then run the system
    controller.safe_run()  # Use safe_run with error handling