import cv2
import numpy as np
import mediapipe as mp
import pickle
import time
import tempfile
import threading
import os
from gtts import gTTS
import pygame
from googletrans import Translator
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import queue


class FlexibleISLPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ISL Real-time Prediction System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2C3E50')

        # Initialize variables
        self.model = None
        self.scaler = None
        self.user_actions = []
        self.translator = Translator()
        self.is_running = False
        self.is_paused = False
        self.cap = None
        self.detection_thread = None
        self.audio_queue = queue.Queue()

        # Detection settings
        self.confidence_threshold = tk.DoubleVar(value=0.25)
        self.detection_interval = tk.DoubleVar(value=5.0)
        self.selected_language = tk.StringVar(value='Tamil')
        self.camera_index = tk.IntVar(value=0)
        self.auto_detect = tk.BooleanVar(value=True)

        # Language options
        self.lang_codes = {
            'Tamil': 'ta',
            'Telugu': 'te', 
            'Kannada': 'kn',
            'Malayalam': 'ml',
            'English': 'en',
            'Hindi': 'hi'
        }

        # MediaPipe setup
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # Initialize pygame
        try:
            pygame.mixer.init()
        except:
            print("Warning: Audio initialization failed")

        # Status variables
        self.last_prediction = ""
        self.last_confidence = 0.0
        self.pose_detected = False
        self.prediction_count = 0

        self.setup_gui()
        self.setup_audio_thread()

    def setup_gui(self):
        """Setup the GUI layout"""
        # Main container
        main_frame = tk.Frame(self.root, bg='#2C3E50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        title_label = tk.Label(main_frame, text="🤟 ISL Real-time Recognition System", 
                              font=('Arial', 20, 'bold'), fg='#ECF0F1', bg='#2C3E50')
        title_label.pack(pady=(0, 20))

        # Top control panel
        self.setup_control_panel(main_frame)

        # Main content area
        content_frame = tk.Frame(main_frame, bg='#2C3E50')
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Video display area
        self.setup_video_area(content_frame)

        # Right panel for controls and info
        self.setup_right_panel(content_frame)

        # Bottom status bar
        self.setup_status_bar(main_frame)

    def setup_control_panel(self, parent):
        """Setup top control panel"""
        control_frame = tk.Frame(parent, bg='#34495E', relief=tk.RAISED, bd=2)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Model loading section with individual file selection
        model_frame = tk.Frame(control_frame, bg='#34495E')
        model_frame.pack(side=tk.LEFT, padx=10, pady=5)

        tk.Label(model_frame, text="Load Your Model Files:", font=('Arial', 10, 'bold'), 
                fg='#ECF0F1', bg='#34495E').pack(anchor=tk.W)

        # Individual file buttons
        file_buttons_frame = tk.Frame(model_frame, bg='#34495E')
        file_buttons_frame.pack(pady=2)

        tk.Button(file_buttons_frame, text="📁 Model (.pkl)", command=self.load_model_file,
                 bg='#3498DB', fg='white', font=('Arial', 8), width=12).pack(side=tk.LEFT, padx=1)
        tk.Button(file_buttons_frame, text="📁 Scaler (.pkl)", command=self.load_scaler_file,
                 bg='#9B59B6', fg='white', font=('Arial', 8), width=12).pack(side=tk.LEFT, padx=1)
        tk.Button(file_buttons_frame, text="📁 Actions (.pkl)", command=self.load_actions_file,
                 bg='#E67E22', fg='white', font=('Arial', 8), width=12).pack(side=tk.LEFT, padx=1)

        # Status labels for each file
        self.model_file_status = tk.Label(model_frame, text="❌ Model", fg='#E74C3C', bg='#34495E', font=('Arial', 8))
        self.model_file_status.pack(anchor=tk.W)
        self.scaler_file_status = tk.Label(model_frame, text="❌ Scaler", fg='#E74C3C', bg='#34495E', font=('Arial', 8))
        self.scaler_file_status.pack(anchor=tk.W)
        self.actions_file_status = tk.Label(model_frame, text="❌ Actions", fg='#E74C3C', bg='#34495E', font=('Arial', 8))
        self.actions_file_status.pack(anchor=tk.W)

        # Camera controls
        camera_frame = tk.Frame(control_frame, bg='#34495E')
        camera_frame.pack(side=tk.LEFT, padx=20, pady=5)

        tk.Label(camera_frame, text="Camera:", font=('Arial', 10, 'bold'), 
                fg='#ECF0F1', bg='#34495E').pack(anchor=tk.W)

        cam_control_frame = tk.Frame(camera_frame, bg='#34495E')
        cam_control_frame.pack()

        tk.Label(cam_control_frame, text="Index:", fg='#ECF0F1', bg='#34495E').pack(side=tk.LEFT)
        tk.Spinbox(cam_control_frame, from_=0, to=5, width=5, textvariable=self.camera_index).pack(side=tk.LEFT, padx=2)

        tk.Button(cam_control_frame, text="Connect", command=self.connect_camera,
                 bg='#27AE60', fg='white', font=('Arial', 9)).pack(side=tk.LEFT, padx=2)

        # Main controls
        main_controls = tk.Frame(control_frame, bg='#34495E')
        main_controls.pack(side=tk.RIGHT, padx=10, pady=5)

        self.start_button = tk.Button(main_controls, text="▶ Start Detection", 
                                     command=self.start_detection, bg='#27AE60', fg='white',
                                     font=('Arial', 12, 'bold'), state=tk.DISABLED)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.pause_button = tk.Button(main_controls, text="⏸ Pause", 
                                     command=self.pause_detection, bg='#F39C12', fg='white',
                                     font=('Arial', 12, 'bold'), state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(main_controls, text="⏹ Stop", 
                                    command=self.stop_detection, bg='#E74C3C', fg='white',
                                    font=('Arial', 12, 'bold'), state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

    def setup_video_area(self, parent):
        """Setup video display area"""
        video_frame = tk.Frame(parent, bg='#2C3E50')
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Video display label
        self.video_label = tk.Label(video_frame, text="📷 Camera Feed\n\nConnect camera to start", 
                                   font=('Arial', 16), fg='#BDC3C7', bg='#34495E',
                                   width=50, height=20)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Prediction display
        pred_frame = tk.Frame(video_frame, bg='#34495E', relief=tk.RAISED, bd=2)
        pred_frame.pack(fill=tk.X, pady=(10, 0))

        tk.Label(pred_frame, text="Current Prediction:", font=('Arial', 12, 'bold'), 
                fg='#ECF0F1', bg='#34495E').pack(anchor=tk.W, padx=5, pady=2)

        self.prediction_label = tk.Label(pred_frame, text="No prediction yet", 
                                        font=('Arial', 16, 'bold'), fg='#3498DB', bg='#34495E')
        self.prediction_label.pack(pady=5)

        self.confidence_label = tk.Label(pred_frame, text="Confidence: 0%", 
                                        font=('Arial', 10), fg='#95A5A6', bg='#34495E')
        self.confidence_label.pack(pady=2)

        # Manual prediction button
        self.manual_predict_button = tk.Button(pred_frame, text="🔍 Predict Now", 
                                             command=self.manual_prediction, bg='#9B59B6', fg='white',
                                             font=('Arial', 10, 'bold'), state=tk.DISABLED)
        self.manual_predict_button.pack(pady=5)

    def setup_right_panel(self, parent):
        """Setup right control panel"""
        right_frame = tk.Frame(parent, bg='#34495E', relief=tk.RAISED, bd=2, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)

        # Settings section
        settings_frame = tk.LabelFrame(right_frame, text="⚙️ Settings", font=('Arial', 12, 'bold'),
                                     fg='#ECF0F1', bg='#34495E')
        settings_frame.pack(fill=tk.X, padx=10, pady=10)

        # Language selection
        tk.Label(settings_frame, text="Output Language:", fg='#ECF0F1', bg='#34495E').pack(anchor=tk.W, pady=2)
        lang_combo = ttk.Combobox(settings_frame, textvariable=self.selected_language, 
                                 values=list(self.lang_codes.keys()), state="readonly", width=25)
        lang_combo.pack(pady=2)

        # Confidence threshold
        tk.Label(settings_frame, text="Confidence Threshold:", fg='#ECF0F1', bg='#34495E').pack(anchor=tk.W, pady=(10,2))
        conf_frame = tk.Frame(settings_frame, bg='#34495E')
        conf_frame.pack(fill=tk.X, pady=2)
        tk.Scale(conf_frame, from_=0.1, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                variable=self.confidence_threshold, bg='#34495E', fg='#ECF0F1',
                highlightthickness=0).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.conf_value_label = tk.Label(conf_frame, text="0.25", fg='#ECF0F1', bg='#34495E', width=6)
        self.conf_value_label.pack(side=tk.RIGHT)

        # Detection interval
        tk.Label(settings_frame, text="Auto Detection Interval (sec):", fg='#ECF0F1', bg='#34495E').pack(anchor=tk.W, pady=(10,2))
        interval_frame = tk.Frame(settings_frame, bg='#34495E')
        interval_frame.pack(fill=tk.X, pady=2)
        tk.Scale(interval_frame, from_=1.0, to=10.0, resolution=0.5, orient=tk.HORIZONTAL,
                variable=self.detection_interval, bg='#34495E', fg='#ECF0F1',
                highlightthickness=0).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.interval_value_label = tk.Label(interval_frame, text="5.0", fg='#ECF0F1', bg='#34495E', width=6)
        self.interval_value_label.pack(side=tk.RIGHT)

        # Auto detection checkbox
        tk.Checkbutton(settings_frame, text="Auto Detection", 
                      variable=self.auto_detect, fg='#ECF0F1', bg='#34495E',
                      selectcolor='#34495E').pack(anchor=tk.W, pady=5)

        # Statistics section
        stats_frame = tk.LabelFrame(right_frame, text="📊 Statistics", font=('Arial', 12, 'bold'),
                                  fg='#ECF0F1', bg='#34495E')
        stats_frame.pack(fill=tk.X, padx=10, pady=10)

        self.stats_text = tk.Text(stats_frame, height=6, width=30, bg='#2C3E50', fg='#ECF0F1',
                                 font=('Courier', 9), state=tk.DISABLED)
        self.stats_text.pack(fill=tk.X, pady=5)

        # Action history
        history_frame = tk.LabelFrame(right_frame, text="📜 Recent Predictions", font=('Arial', 12, 'bold'),
                                    fg='#ECF0F1', bg='#34495E')
        history_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.history_listbox = tk.Listbox(history_frame, bg='#2C3E50', fg='#ECF0F1',
                                         font=('Courier', 9), selectbackground='#3498DB')
        self.history_listbox.pack(fill=tk.BOTH, expand=True, pady=5)

        # Clear history button
        tk.Button(history_frame, text="Clear History", command=self.clear_history,
                 bg='#95A5A6', fg='white', font=('Arial', 9)).pack(pady=2)

        # Update labels
        self.confidence_threshold.trace('w', self.update_conf_label)
        self.detection_interval.trace('w', self.update_interval_label)

    def setup_status_bar(self, parent):
        """Setup bottom status bar"""
        status_frame = tk.Frame(parent, bg='#34495E', relief=tk.SUNKEN, bd=1)
        status_frame.pack(fill=tk.X, pady=(10, 0))

        self.status_label = tk.Label(status_frame, text="Ready - Load your model files to start", 
                                   fg='#ECF0F1', bg='#34495E', anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=5, pady=2)

        self.fps_label = tk.Label(status_frame, text="FPS: 0", 
                                fg='#ECF0F1', bg='#34495E', anchor=tk.E)
        self.fps_label.pack(side=tk.RIGHT, padx=5, pady=2)

    def setup_audio_thread(self):
        """Setup audio processing thread"""
        def audio_worker():
            while True:
                try:
                    text, lang_code = self.audio_queue.get(timeout=1)
                    if text == "STOP":
                        break
                    self.speak_text(text, lang_code)
                    self.audio_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Audio error: {e}")

        self.audio_thread = threading.Thread(target=audio_worker, daemon=True)
        self.audio_thread.start()

    def update_conf_label(self, *args):
        self.conf_value_label.config(text=f"{self.confidence_threshold.get():.2f}")

    def update_interval_label(self, *args):
        self.interval_value_label.config(text=f"{self.detection_interval.get():.1f}")

    def load_model_file(self):
        """Load model file individually"""
        try:
            model_file = filedialog.askopenfilename(
                title="Select Model File (.pkl)",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            if model_file:
                with open(model_file, 'rb') as f:
                    self.model = pickle.load(f)
                self.model_file_status.config(text="✅ Model", fg='#27AE60')
                self.check_all_files_loaded()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model file:\n{str(e)}")

    def load_scaler_file(self):
        """Load scaler file individually"""
        try:
            scaler_file = filedialog.askopenfilename(
                title="Select Scaler File (.pkl)",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            if scaler_file:
                with open(scaler_file, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.scaler_file_status.config(text="✅ Scaler", fg='#27AE60')
                self.check_all_files_loaded()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load scaler file:\n{str(e)}")

    def load_actions_file(self):
        """Load actions file individually"""
        try:
            actions_file = filedialog.askopenfilename(
                title="Select Actions File (.pkl)",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            if actions_file:
                with open(actions_file, 'rb') as f:
                    self.user_actions = pickle.load(f)
                self.actions_file_status.config(text="✅ Actions", fg='#27AE60')
                self.check_all_files_loaded()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load actions file:\n{str(e)}")

    def check_all_files_loaded(self):
        """Check if all required files are loaded"""
        if self.model and self.scaler and self.user_actions:
            self.status_label.config(text=f"All files loaded - {len(self.user_actions)} actions available")
            self.connect_camera()

    def connect_camera(self):
        """Connect to camera"""
        try:
            if self.cap:
                self.cap.release()

            self.cap = cv2.VideoCapture(self.camera_index.get())
            if not self.cap.isOpened():
                raise Exception(f"Cannot open camera {self.camera_index.get()}")

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            self.status_label.config(text=f"Camera {self.camera_index.get()} connected")

            if self.model and self.scaler and self.user_actions:
                self.start_button.config(state=tk.NORMAL)
                self.manual_predict_button.config(state=tk.NORMAL)

            # Start video preview
            self.update_video_feed()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to connect camera:\n{str(e)}")
            self.status_label.config(text="Camera connection failed")

    def extract_keypoints(self, results):
        """Extract keypoints from MediaPipe results (original method)"""
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                       results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        lh = np.array([[res.x, res.y, res.z] for res in
                      results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in
                      results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, lh, rh])

    def make_prediction(self, frame):
        """Make prediction on current frame"""
        try:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(image_rgb)

            self.pose_detected = bool(results.pose_landmarks)

            if results.pose_landmarks:
                keypoints = self.extract_keypoints(results)
                keypoints_scaled = self.scaler.transform([keypoints])
                probabilities = self.model.predict_proba(keypoints_scaled)[0]
                max_prob = max(probabilities)
                predicted_index = np.argmax(probabilities)

                if max_prob > self.confidence_threshold.get():
                    raw_label = self.user_actions[predicted_index].replace('_', ' ')

                    # Translation
                    try:
                        if self.selected_language.get() != 'English':
                            translated = self.translator.translate(raw_label, dest=self.lang_codes[self.selected_language.get()])
                            translated_text = translated.text
                        else:
                            translated_text = raw_label
                    except Exception as e:
                        print(f"Translation error: {e}")
                        translated_text = raw_label

                    self.last_prediction = raw_label
                    self.last_confidence = max_prob

                    # Update GUI
                    self.root.after(0, self.update_prediction_display, raw_label, translated_text, max_prob)

                    # Add to history
                    timestamp = time.strftime("%H:%M:%S")
                    history_entry = f"{timestamp}: {raw_label} ({max_prob:.2f})"
                    self.root.after(0, self.add_to_history, history_entry)

                    # Queue audio
                    self.audio_queue.put((translated_text, self.lang_codes[self.selected_language.get()]))

                    self.prediction_count += 1

                    return raw_label, max_prob
                else:
                    self.root.after(0, self.update_prediction_display, "Low Confidence", "", max_prob)
            else:
                self.root.after(0, self.update_prediction_display, "No Pose Detected", "", 0.0)

        except Exception as e:
            print(f"Prediction error: {e}")

        return None, 0.0

    def update_prediction_display(self, prediction, translated, confidence):
        """Update prediction display"""
        self.prediction_label.config(text=prediction)
        self.confidence_label.config(text=f"Confidence: {confidence:.2%}")

        if translated and translated != prediction:
            self.prediction_label.config(text=f"{prediction}\n({translated})")

    def add_to_history(self, entry):
        """Add entry to history"""
        self.history_listbox.insert(0, entry)
        if self.history_listbox.size() > 50:  # Keep only last 50 entries
            self.history_listbox.delete(50)

    def clear_history(self):
        """Clear prediction history"""
        self.history_listbox.delete(0, tk.END)

    def speak_text(self, text, lang_code):
        """Text-to-speech function"""
        try:
            tts = gTTS(text=text, lang=lang_code, slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_name = fp.name
                tts.save(temp_name)

            pygame.mixer.music.load(temp_name)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            os.remove(temp_name)

        except Exception as e:
            print(f"Audio Error: {e}")

    def update_video_feed(self):
        """Update video feed display"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)

                # Add overlays
                if self.is_running and not self.is_paused:
                    cv2.putText(frame, "🔴 DETECTING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif self.is_paused:
                    cv2.putText(frame, "⏸ PAUSED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
                else:
                    cv2.putText(frame, "Show your gesture!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Pose status
                if self.pose_detected:
                    cv2.putText(frame, "✓ Pose Detected", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "✗ No Pose", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Convert for tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_tk = ImageTk.PhotoImage(frame_pil)

                self.video_label.config(image=frame_tk, text="")
                self.video_label.image = frame_tk

        # Schedule next update
        self.root.after(33, self.update_video_feed)  # ~30 FPS

    def detection_worker(self):
        """Detection worker thread"""
        last_detection_time = 0
        frame_count = 0
        fps_start_time = time.time()

        while self.is_running:
            if self.is_paused:
                time.sleep(0.1)
                continue

            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    frame_count += 1

                    # Calculate FPS
                    if frame_count % 30 == 0:
                        fps = 30 / (time.time() - fps_start_time)
                        self.root.after(0, self.fps_label.config, {"text": f"FPS: {fps:.1f}"})
                        fps_start_time = time.time()

                    # Auto detection
                    now = time.time()
                    if self.auto_detect.get() and (now - last_detection_time > self.detection_interval.get()):
                        self.make_prediction(frame)
                        last_detection_time = now

                    # Update stats
                    if frame_count % 60 == 0:  # Update every 2 seconds
                        self.update_stats()

            time.sleep(0.033)  # ~30 FPS

    def update_stats(self):
        """Update statistics display"""
        stats_text = f"""Predictions: {self.prediction_count}
Last Action: {self.last_prediction[:15]}
Confidence: {self.last_confidence:.2%}
Pose Status: {"✓" if self.pose_detected else "✗"}
Language: {self.selected_language.get()}
Actions: {len(self.user_actions)}"""

        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
        self.stats_text.config(state=tk.DISABLED)

    def start_detection(self):
        """Start detection process"""
        if not all([self.model, self.scaler, self.user_actions, self.cap]):
            messagebox.showerror("Error", "All files and camera must be loaded first")
            return

        self.is_running = True
        self.is_paused = False

        self.start_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL)

        self.status_label.config(text="Detection started")

        # Start detection thread
        self.detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
        self.detection_thread.start()

    def pause_detection(self):
        """Pause/resume detection"""
        if self.is_running:
            self.is_paused = not self.is_paused
            if self.is_paused:
                self.pause_button.config(text="▶ Resume")
                self.status_label.config(text="Detection paused")
            else:
                self.pause_button.config(text="⏸ Pause")
                self.status_label.config(text="Detection resumed")

    def stop_detection(self):
        """Stop detection process"""
        self.is_running = False
        self.is_paused = False

        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED, text="⏸ Pause")
        self.stop_button.config(state=tk.DISABLED)

        self.status_label.config(text="Detection stopped")

    def manual_prediction(self):
        """Manual prediction trigger"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                self.make_prediction(frame)

    def on_closing(self):
        """Handle window closing"""
        self.is_running = False

        if self.cap:
            self.cap.release()

        # Stop audio thread
        self.audio_queue.put(("STOP", "en"))

        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FlexibleISLPredictionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()