import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math

class HandGestureController:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Screen resolution for mouse control
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Gesture states
        self.prev_gesture = None
        self.gesture_smoothing = []
        self.smoothing_window = 5
        
        # Mouse control parameters
        self.prev_index_tip = None
        self.mouse_smoothing = []
        self.mouse_smooth_factor = 10
        
        # Virtual mouse mode
        self.virtual_mouse_active = False
        
    def detect_hands(self, frame):
        """Process frame and detect hands."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        return results
    
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
    
    def recognize_gesture(self, hand_landmarks):
        """Recognize hand gestures based on landmark positions."""
        if not hand_landmarks:
            return "No Hand"
        
        # Extract keypoints from the first hand
        landmarks = hand_landmarks.landmark
        
        # Get fingertip and base positions for gesture recognition
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
        
        index_pip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_pip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        ring_pip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_PIP]
        pinky_pip = landmarks[self.mp_hands.HandLandmark.PINKY_PIP]
        
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        
        # Define gestures based on finger positions
        # Pointing (index finger extended, others closed)
        if (index_tip.y < index_pip.y and  # Index finger up
            middle_tip.y > middle_pip.y and  # Middle finger down
            ring_tip.y > ring_pip.y and  # Ring finger down
            pinky_tip.y > pinky_pip.y):  # Pinky down
            gesture = "Point"
        
        # Open hand (all fingers extended)
        elif (index_tip.y < index_pip.y and
              middle_tip.y < middle_pip.y and
              ring_tip.y < ring_pip.y and
              pinky_tip.y < pinky_pip.y):
            gesture = "Open"
        
        # Pinch gesture (thumb and index fingertips close)
        elif self.calculate_distance(thumb_tip, index_tip) < 0.1:
            gesture = "Pinch"
        
        # Fist (all fingers closed)
        elif (index_tip.y > index_pip.y and
              middle_tip.y > middle_pip.y and
              ring_tip.y > ring_pip.y and
              pinky_tip.y > pinky_pip.y):
            gesture = "Fist"
        
        # Peace sign (index and middle extended, others closed)
        elif (index_tip.y < index_pip.y and
              middle_tip.y < middle_pip.y and
              ring_tip.y > ring_pip.y and
              pinky_tip.y > pinky_pip.y):
            gesture = "Peace"
        
        # Default if no other gesture matches
        else:
            gesture = "Unknown"
        
        # Smooth the gesture recognition with a simple voting system
        self.gesture_smoothing.append(gesture)
        if len(self.gesture_smoothing) > self.smoothing_window:
            self.gesture_smoothing.pop(0)
        
        # Return the most common gesture in the window
        if self.gesture_smoothing:
            from collections import Counter
            return Counter(self.gesture_smoothing).most_common(1)[0][0]
        
        return gesture
    
    def calculate_distance(self, landmark1, landmark2):
        """Calculate normalized distance between two landmarks."""
        return math.sqrt((landmark1.x - landmark2.x)**2 + 
                         (landmark1.y - landmark2.y)**2)
    
    def map_gesture_to_action(self, gesture, hand_landmarks, frame_shape):
        """Map recognized gestures to mouse/keyboard actions."""
        if not hand_landmarks:
            return
        
        h, w = frame_shape[:2]
        landmarks = hand_landmarks.landmark
        
        # Extract key points
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
        
        # Toggle virtual mouse control with "Peace" gesture
        if gesture == "Peace" and self.prev_gesture != "Peace":
            self.virtual_mouse_active = not self.virtual_mouse_active
            if self.virtual_mouse_active:
                print("Virtual mouse activated")
            else:
                print("Virtual mouse deactivated")
        
        # Virtual mouse control when active
        if self.virtual_mouse_active:
            # Point gesture: Move mouse
            if gesture == "Point":
                # Convert hand position to screen coordinates
                mouse_x = np.interp(index_x, [50, w-50], [0, self.screen_width])
                mouse_y = np.interp(index_y, [50, h-50], [0, self.screen_height])
                
                # Apply smoothing
                self.mouse_smoothing.append((mouse_x, mouse_y))
                if len(self.mouse_smoothing) > self.mouse_smooth_factor:
                    self.mouse_smoothing.pop(0)
                
                # Calculate average position
                if self.mouse_smoothing:
                    avg_x = sum(x for x, y in self.mouse_smoothing) / len(self.mouse_smoothing)
                    avg_y = sum(y for x, y in self.mouse_smoothing) / len(self.mouse_smoothing)
                    
                    # Move mouse
                    pyautogui.moveTo(avg_x, avg_y)
            
            # Pinch gesture: Click
            elif gesture == "Pinch" and self.prev_gesture != "Pinch":
                pyautogui.click()
                print("Click")
            
            # Open hand: Right click
            elif gesture == "Open" and self.prev_gesture != "Open":
                pyautogui.rightClick()
                print("Right Click")
            
            # Fist: Drag (hold left mouse button)
            elif gesture == "Fist":
                if self.prev_gesture != "Fist":
                    pyautogui.mouseDown()
                    print("Mouse Down")
                
                # Continue moving while dragging
                mouse_x = np.interp(index_x, [50, w-50], [0, self.screen_width])
                mouse_y = np.interp(index_y, [50, h-50], [0, self.screen_height])
                pyautogui.moveTo(mouse_x, mouse_y)
            
            # Release drag when changing from Fist to another gesture
            elif self.prev_gesture == "Fist":
                pyautogui.mouseUp()
                print("Mouse Up")
        
        # Update previous gesture
        self.prev_gesture = gesture
    
    def run(self):
        """Main loop to run the hand gesture controller."""
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read from webcam")
                break
            
            # Flip horizontally for a more intuitive mirror view
            frame = cv2.flip(frame, 1)
            
            # Detect hands
            results = self.detect_hands(frame)
            
            # Draw landmarks
            frame = self.draw_landmarks(frame, results)
            
            # Segment hand
            segmented_hand, mask = self.segment_hand(frame, results)
            
            # Recognize gestures
            gesture = "No Hand"
            if results.multi_hand_landmarks:
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


if __name__ == "__main__":
    # Initialize and run the hand gesture controller
    controller = HandGestureController()
    print("Starting Hand Gesture Control System")
    print("Make a 'Peace' sign to toggle virtual mouse control")
    print("Use 'Point' to move, 'Pinch' to click, 'Open Hand' to right-click, 'Fist' to drag")
    print("Press 'q' to quit")
    controller.run()