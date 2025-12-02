import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
import json
from datetime import datetime

class CustomLabelDataCollector:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # User-defined actions list - will be built dynamically
        self.user_actions = []
        self.collected_data = {}
        self.frames_per_action = 10
        self.input_mode = True  # True = getting label input, False = collecting poses
        self.current_action = ""
        self.frames_collected_for_current = 0
        
        # Create data directory
        os.makedirs('custom_label_data', exist_ok=True)
        
    def extract_keypoints(self, results):
        """Extract pose keypoints from mediapipe results"""
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in 
                        results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        
        lh = np.array([[res.x, res.y, res.z] for res in 
                      results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        
        rh = np.array([[res.x, res.y, res.z] for res in 
                      results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        
        return np.concatenate([pose, lh, rh])
    
    def get_action_label_from_user(self):
        """Get action label from user via console input"""
        print("\n" + "="*50)
        print("CUSTOM ACTION LABEL INPUT")
        print("="*50)
        
        if self.user_actions:
            print("Previously added actions:")
            for i, action in enumerate(self.user_actions, 1):
                frames_count = len(self.collected_data.get(action, []))
                status = f"({frames_count}/{self.frames_per_action} frames)" if action in self.collected_data else "(0 frames)"
                print(f"  {i}. {action} {status}")
            print()
        
        while True:
            action_input = input("Enter action label (or 'quit' to finish, 'list' to see actions): ").strip()
            
            if action_input.lower() == 'quit':
                return None
            elif action_input.lower() == 'list':
                if self.user_actions:
                    print("\nCurrent actions:")
                    for i, action in enumerate(self.user_actions, 1):
                        frames_count = len(self.collected_data.get(action, []))
                        print(f"  {i}. {action} ({frames_count}/{self.frames_per_action} frames)")
                else:
                    print("No actions added yet.")
                continue
            elif action_input == "":
                print("Please enter a valid action label.")
                continue
            
            # Clean the input
            action_label = action_input.lower().replace(" ", "_")
            
            if action_label not in self.user_actions:
                self.user_actions.append(action_label)
                self.collected_data[action_label] = []
                print(f"✓ New action '{action_label}' added!")
            else:
                frames_count = len(self.collected_data[action_label])
                print(f"✓ Action '{action_label}' selected. Current frames: {frames_count}/{self.frames_per_action}")
            
            return action_label
    
    def collect_custom_label_data(self):
        """Main data collection loop with user-defined labels"""
        cap = cv2.VideoCapture(2)
        
        print("=== CUSTOM ISL DATA COLLECTION ===")
        print("Instructions:")
        print("1. You'll be asked to enter an action label via keyboard")
        print("2. Then perform the pose and press SPACE to capture frames")
        print("3. Each action needs 10 frames")
        print("4. Press 'n' to enter next action label")
        print("5. Press ESC to exit")
        print("\nStarting data collection...")
        
        while True:
            # Get action label from user
            action_label = self.get_action_label_from_user()
            
            if action_label is None:  # User wants to quit
                break
            
            self.current_action = action_label
            self.frames_collected_for_current = len(self.collected_data[action_label])
            
            print(f"\n▶ Now collect poses for: '{action_label}'")
            print(f"▶ Frames needed: {self.frames_per_action - self.frames_collected_for_current}")
            print("▶ Camera feed will open - press SPACE to capture, 'n' for next action, ESC to exit")
            
            # Camera collection loop for current action
            while self.frames_collected_for_current < self.frames_per_action:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Make detection
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = self.holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Draw landmarks
                self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
                self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                
                # Display information [48][7]
                cv2.putText(image, f'Action: {self.current_action}', 
                           (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, f'Frames: {self.frames_collected_for_current}/{self.frames_per_action}', 
                           (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(image, 'SPACE: Capture | N: Next Action | ESC: Exit', 
                           (15, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Progress bar
                bar_width = 400
                bar_height = 20
                progress = self.frames_collected_for_current / self.frames_per_action
                cv2.rectangle(image, (15, 100), (15 + bar_width, 100 + bar_height), (50, 50, 50), -1)
                cv2.rectangle(image, (15, 100), (15 + int(bar_width * progress), 100 + bar_height), (0, 255, 0), -1)
                
                cv2.imshow('Custom ISL Data Collection', image)
                
                # Keyboard input handling [1][10]
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC to exit completely
                    cap.release()
                    cv2.destroyAllWindows()
                    self.save_data()
                    return
                elif key == ord(' '):  # SPACE to capture frame
                    if results.pose_landmarks:
                        keypoints = self.extract_keypoints(results)
                        self.collected_data[self.current_action].append(keypoints)
                        self.frames_collected_for_current += 1
                        print(f"✓ Captured frame {self.frames_collected_for_current} for '{self.current_action}'")
                        
                        # Check if enough frames collected
                        if self.frames_collected_for_current >= self.frames_per_action:
                            print(f"✓ Completed data collection for '{self.current_action}'!")
                            break
                    else:
                        print("❌ No pose detected! Please ensure you're visible in the frame.")
                        
                elif key == ord('n'):  # Next action
                    print(f"Skipping to next action (current: {self.frames_collected_for_current} frames)")
                    break
            
            cv2.destroyAllWindows()
        
        cap.release()
        self.save_data()
    
    def save_data(self):
        """Save collected data and action labels"""
        print("\n" + "="*50)
        print("SAVING DATA")
        print("="*50)
        
        # Save individual action data
        for action in self.user_actions:
            if self.collected_data[action]:
                action_data = np.array(self.collected_data[action])
                np.save(f'custom_label_data/{action}.npy', action_data)
                print(f"✓ Saved {len(self.collected_data[action])} frames for '{action}'")
            else:
                print(f"⚠ No data collected for '{action}'")
        
        # Save action labels list [6]
        with open('custom_label_data/user_actions.pkl', 'wb') as f:
            pickle.dump(self.user_actions, f)
        
        # Save metadata
        metadata = {
            'total_actions': len(self.user_actions),
            'actions': self.user_actions,
            'frames_per_action': self.frames_per_action,
            'collection_date': datetime.now().isoformat(),
            'data_summary': {action: len(self.collected_data[action]) for action in self.user_actions}
        }
        
        with open('custom_label_data/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Data collection completed!")
        print(f"✓ Total actions: {len(self.user_actions)}")
        print(f"✓ Metadata saved to 'custom_label_data/metadata.json'")

if __name__ == "__main__":
    collector = CustomLabelDataCollector()
    collector.collect_custom_label_data()
