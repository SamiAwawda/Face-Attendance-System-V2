"""
CameraEngine - AI Core for Face Detection and Recognition
Uses MediaPipe for fast face detection and DeepFace for 512-d embeddings.
"""

import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
from threading import Thread, Lock
from datetime import datetime, timedelta
import time


class CameraEngine:
    """
    Thread-safe camera engine for face detection and recognition.
    Optimized for high FPS with frame skipping and separate recognition thread.
    """
    
    # Bounding box colors (BGR format)
    COLOR_GREEN = (0, 255, 0)      # Success - attendance marked
    COLOR_GOLD = (0, 215, 255)     # Already marked within 60 min
    COLOR_RED = (0, 0, 255)        # Unknown/unregistered face
    COLOR_WHITE = (255, 255, 255)  # Default detection
    
    def __init__(self, camera_index=0):
        """Initialize the camera engine."""
        self.camera_index = camera_index
        self.cap = None
        self.is_running = False
        
        # MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for short-range (2m), 1 for full-range (5m)
            min_detection_confidence=0.6
        )
        
        # Thread safety
        self.frame_lock = Lock()
        self.current_frame = None
        self.processed_frame = None
        
        # Recognition state
        self.recognition_lock = Lock()
        self.last_recognition_result = None
        self.recognition_thread = None
        self.recognition_running = False
        self.recognition_in_progress = False  # Prevent concurrent recognitions
        
        # Frame skipping for performance - process every Nth frame for AI
        self.frame_count = 0
        self.process_every_n_frames = 15  # Increased from 5 to reduce AI load
        
        # Processing frame size (smaller for faster AI detection)
        self.process_width = 320
        self.process_height = 240
        
        # AI processing frame size (even smaller for DeepFace)
        self.ai_process_width = 160
        self.ai_process_height = 120
        
        # Known faces database (loaded from Firebase)
        self.known_embeddings = []  # List of (student_id, name, embedding)
        self.embeddings_lock = Lock()
        
        # Recent recognition results for display
        self.face_results = {}  # {face_id: (color, name, status, timestamp)}
        self.face_results_lock = Lock()
        
        # Callback for attendance logging
        self.on_attendance_marked = None
        
        # Capture mode flag
        self.capture_requested = False
        self.captured_embedding = None
        self.capture_lock = Lock()
        
        # Last recognition time for global throttling
        self.last_recognition_time = datetime.now() - timedelta(seconds=10)
    
    def start(self):
        """Start the camera and processing threads."""
        if self.is_running:
            return True
            
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.is_running = True
        
        # Start frame capture thread
        Thread(target=self._capture_loop, daemon=True).start()
        
        return True
    
    def stop(self):
        """Stop the camera and all threads."""
        self.is_running = False
        self.recognition_running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def _capture_loop(self):
        """Continuous frame capture loop - optimized to always get latest frame."""
        while self.is_running:
            if self.cap is None:
                break
            
            # Flush the buffer - read multiple times to get the latest frame
            # This prevents lag buildup from buffered frames
            for _ in range(2):  # Discard 2 frames to clear buffer
                self.cap.grab()
            
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            
            # Store raw frame (always latest)
            with self.frame_lock:
                self.current_frame = frame  # Don't copy, just reference
            
            self.frame_count += 1
            time.sleep(0.016)  # ~60fps max capture rate, reduces CPU load
    
    def get_frame(self, mode='registration'):
        """
        Get the current processed frame with bounding boxes.
        Optimized for high FPS with resized processing.
        
        Args:
            mode: 'registration' or 'attendance'
            
        Returns:
            JPEG encoded frame bytes
        """
        with self.frame_lock:
            if self.current_frame is None:
                # Return blank frame if no camera feed
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                _, buffer = cv2.imencode('.jpg', blank)
                return buffer.tobytes()
            frame = self.current_frame.copy()  # Copy here for thread safety
        
        h, w = frame.shape[:2]
        
        # Resize for faster face detection (MediaPipe)
        small_frame = cv2.resize(frame, (self.process_width, self.process_height))
        small_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Scale factors to map back to original frame
        scale_x = w / self.process_width
        scale_y = h / self.process_height
        
        # Detect faces on smaller frame (faster)
        results = self.face_detection.process(small_rgb)
        
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                # Scale coordinates back to original frame size
                x = int(bbox.xmin * self.process_width * scale_x)
                y = int(bbox.ymin * self.process_height * scale_y)
                box_w = int(bbox.width * self.process_width * scale_x)
                box_h = int(bbox.height * self.process_height * scale_y)
                
                # Ensure bounds
                x = max(0, x)
                y = max(0, y)
                
                # Generate face key for tracking
                face_key = f"{x//50}_{y//50}"
                
                if mode == 'registration':
                    # Registration mode - check if face is already registered
                    with self.face_results_lock:
                        if face_key in self.face_results:
                            color, name, status, _ = self.face_results[face_key]
                            if status == "registered":
                                # Already registered - RED box
                                label = f"Already Registered: {name}"
                            elif status == "new":
                                # New face - GREEN box
                                color = self.COLOR_GREEN
                                label = "New Face - Ready"
                            else:
                                # Scanning
                                color = self.COLOR_WHITE
                                label = "Scanning..."
                        else:
                            color = self.COLOR_WHITE
                            label = "Scanning..."
                            status = "scanning"
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), color, 2)
                    
                    # Draw label with background
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x, y - 25), (x + label_size[0] + 10, y), color, -1)
                    # Use white text on colored background
                    text_color = (0, 0, 0) if color == self.COLOR_GREEN or color == self.COLOR_WHITE else (255, 255, 255)
                    cv2.putText(frame, label, (x + 5, y - 7),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                    
                    # Run recognition check for registration mode (less frequent)
                    should_process = (
                        self.frame_count % (self.process_every_n_frames * 2) == 0 and
                        not self.recognition_in_progress and
                        (datetime.now() - self.last_recognition_time).total_seconds() >= 3
                    )
                    
                    if should_process and self.known_embeddings:
                        # Use smaller frame for AI processing
                        ai_frame = cv2.resize(frame, (self.ai_process_width * 2, self.ai_process_height * 2))
                        ai_frame_rgb = cv2.cvtColor(ai_frame, cv2.COLOR_BGR2RGB)
                        
                        # Scale coordinates for AI frame
                        ai_scale = (self.ai_process_width * 2) / w
                        ai_x = int(x * ai_scale)
                        ai_y = int(y * ai_scale)
                        ai_w = int(box_w * ai_scale)
                        ai_h = int(box_h * ai_scale)
                        
                        self._process_registration_check(ai_frame_rgb, ai_x, ai_y, ai_w, ai_h, face_key)
                else:
                    # Attendance mode - get recognition result
                    with self.face_results_lock:
                        if face_key in self.face_results:
                            color, name, status, _ = self.face_results[face_key]
                        else:
                            color = self.COLOR_RED
                            name = "Scanning..."
                            status = "scanning"
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), color, 3)
                    
                    # Draw name label with background
                    label = name if name else "Unknown"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(frame, (x, y - 30), (x + label_size[0] + 10, y), color, -1)
                    cv2.putText(frame, label, (x + 5, y - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    
                    # Process recognition ONLY if:
                    # 1. On the Nth frame
                    # 2. No recognition currently in progress
                    # 3. At least 2 seconds since last recognition attempt
                    should_process = (
                        self.frame_count % self.process_every_n_frames == 0 and
                        not self.recognition_in_progress and
                        (datetime.now() - self.last_recognition_time).total_seconds() >= 2
                    )
                    
                    if should_process:
                        # Use smaller frame for AI processing
                        ai_frame = cv2.resize(frame, (self.ai_process_width * 2, self.ai_process_height * 2))
                        ai_frame_rgb = cv2.cvtColor(ai_frame, cv2.COLOR_BGR2RGB)
                        
                        # Scale coordinates for AI frame
                        ai_scale = (self.ai_process_width * 2) / w
                        ai_x = int(x * ai_scale)
                        ai_y = int(y * ai_scale)
                        ai_w = int(box_w * ai_scale)
                        ai_h = int(box_h * ai_scale)
                        
                        self._process_recognition_async(ai_frame_rgb, ai_x, ai_y, ai_w, ai_h, face_key)
        
        # Encode to JPEG (lower quality = faster encoding)
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buffer.tobytes()
    
    def _process_recognition_async(self, frame_rgb, x, y, w, h, face_key):
        """Process face recognition in background with cooldown."""
        try:
            if not self.known_embeddings:
                return
            
            # Prevent concurrent recognitions
            if self.recognition_in_progress:
                return
            
            # Mark recognition as in progress and update timestamp
            self.recognition_in_progress = True
            self.last_recognition_time = datetime.now()
            
            # Check if we recently processed this face position (cooldown)
            with self.face_results_lock:
                if face_key in self.face_results:
                    _, _, _, last_time = self.face_results[face_key]
                    # Skip if processed within last 3 seconds
                    if (datetime.now() - last_time).total_seconds() < 3:
                        self.recognition_in_progress = False
                        return
                
            # Extract face region with padding
            pad = 10  # Reduced padding for smaller AI frame
            y1 = max(0, y - pad)
            y2 = min(frame_rgb.shape[0], y + h + pad)
            x1 = max(0, x - pad)
            x2 = min(frame_rgb.shape[1], x + w + pad)
            
            # Validate region bounds
            if y2 <= y1 or x2 <= x1:
                self.recognition_in_progress = False
                return
            
            face_region = frame_rgb[y1:y2, x1:x2].copy()  # Make a copy to avoid threading issues
            
            if face_region is None or face_region.size == 0:
                self.recognition_in_progress = False
                return
            
            # Ensure minimum face size for DeepFace
            if face_region.shape[0] < 20 or face_region.shape[1] < 20:
                self.recognition_in_progress = False
                return
            
            # Run recognition in thread with full error protection
            def recognize():
                try:
                    embedding = self._get_embedding(face_region)
                    if embedding is not None:
                        result = self._match_embedding(embedding)
                        if result:
                            student_id, name, distance = result
                            print(f"Face matched: {name} (distance: {distance:.3f})")
                            
                            # Check attendance status via callback
                            try:
                                if self.on_attendance_marked:
                                    status, color = self.on_attendance_marked(student_id, name)
                                    print(f"Attendance result: {status}")
                                else:
                                    status = "success"
                                    color = self.COLOR_GREEN
                            except Exception as callback_error:
                                print(f"Attendance callback error: {callback_error}")
                                status = "error"
                                color = self.COLOR_RED
                            
                            with self.face_results_lock:
                                self.face_results[face_key] = (color, name, status, datetime.now())
                        else:
                            with self.face_results_lock:
                                self.face_results[face_key] = (self.COLOR_RED, "Unknown", "unregistered", datetime.now())
                except Exception as e:
                    print(f"Recognition thread error: {e}")
                    import traceback
                    traceback.print_exc()
                    # Don't crash - just mark as unknown
                    try:
                        with self.face_results_lock:
                            self.face_results[face_key] = (self.COLOR_RED, "Error", "error", datetime.now())
                    except:
                        pass
                finally:
                    # Always release the recognition lock
                    self.recognition_in_progress = False
            
            Thread(target=recognize, daemon=True).start()
        except Exception as e:
            print(f"Recognition setup error: {e}")
            import traceback
            traceback.print_exc()
            self.recognition_in_progress = False
    
    def _process_registration_check(self, frame_rgb, x, y, w, h, face_key):
        """Check if face is already registered (for registration mode duplicate prevention)."""
        try:
            # Prevent concurrent recognitions
            if self.recognition_in_progress:
                return
            
            # Mark as in progress
            self.recognition_in_progress = True
            self.last_recognition_time = datetime.now()
            
            # Extract face region with padding
            pad = 10
            y1 = max(0, y - pad)
            y2 = min(frame_rgb.shape[0], y + h + pad)
            x1 = max(0, x - pad)
            x2 = min(frame_rgb.shape[1], x + w + pad)
            
            # Validate region bounds
            if y2 <= y1 or x2 <= x1:
                self.recognition_in_progress = False
                return
            
            face_region = frame_rgb[y1:y2, x1:x2].copy()
            
            if face_region is None or face_region.size == 0:
                self.recognition_in_progress = False
                return
            
            # Ensure minimum face size
            if face_region.shape[0] < 20 or face_region.shape[1] < 20:
                self.recognition_in_progress = False
                return
            
            def check_registration():
                try:
                    embedding = self._get_embedding(face_region)
                    if embedding is not None:
                        result = self._match_embedding(embedding)
                        if result:
                            # Face is already registered
                            student_id, name, distance = result
                            print(f"Registration check: Found existing face - {name} (distance: {distance:.3f})")
                            with self.face_results_lock:
                                self.face_results[face_key] = (self.COLOR_RED, name, "registered", datetime.now())
                        else:
                            # New face - can be registered
                            print("Registration check: New face detected")
                            with self.face_results_lock:
                                self.face_results[face_key] = (self.COLOR_GREEN, "New", "new", datetime.now())
                    else:
                        # Could not get embedding - show as scanning
                        with self.face_results_lock:
                            self.face_results[face_key] = (self.COLOR_WHITE, "", "scanning", datetime.now())
                except Exception as e:
                    print(f"Registration check error: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    self.recognition_in_progress = False
            
            Thread(target=check_registration, daemon=True).start()
        except Exception as e:
            print(f"Registration check setup error: {e}")
            self.recognition_in_progress = False

    def _get_embedding(self, face_image):
        """Generate 512-d embedding from face image."""
        try:
            # Validate input
            if face_image is None or face_image.size == 0:
                print("Warning: Empty face image passed to _get_embedding")
                return None
            
            # Ensure image is valid numpy array
            if not isinstance(face_image, np.ndarray):
                print("Warning: face_image is not a numpy array")
                return None
            
            # DeepFace expects BGR
            face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            
            # Resize if too small (DeepFace minimum requirement)
            min_size = 48
            if face_bgr.shape[0] < min_size or face_bgr.shape[1] < min_size:
                face_bgr = cv2.resize(face_bgr, (max(min_size, face_bgr.shape[1]), max(min_size, face_bgr.shape[0])))
            
            # Get embedding using Facenet512 with full error protection
            result = DeepFace.represent(
                face_bgr,
                model_name="Facenet512",
                enforce_detection=False,
                detector_backend="skip"  # We already detected the face
            )
            
            if result and len(result) > 0 and 'embedding' in result[0]:
                return np.array(result[0]['embedding'])
        except ValueError as ve:
            # Common DeepFace error - face could not be detected/processed
            print(f"DeepFace ValueError: {ve}")
        except Exception as e:
            print(f"Embedding error: {e}")
            import traceback
            traceback.print_exc()
        return None
    
    def _match_embedding(self, embedding, threshold=0.6):
        """
        Match embedding against known faces.
        
        Returns:
            (student_id, name, distance) if match found, None otherwise
        """
        with self.embeddings_lock:
            if not self.known_embeddings:
                return None
            
            best_match = None
            best_distance = float('inf')
            
            for student_id, name, known_embedding in self.known_embeddings:
                # Cosine distance
                known_emb = np.array(known_embedding)
                distance = 1 - np.dot(embedding, known_emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(known_emb)
                )
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = (student_id, name, distance)
            
            if best_match and best_match[2] < threshold:
                return best_match
        
        return None
    
    def load_embeddings(self, embeddings_list):
        """
        Load known face embeddings from database.
        
        Args:
            embeddings_list: List of (student_id, name, embedding) tuples
        """
        with self.embeddings_lock:
            self.known_embeddings = embeddings_list
            print(f"Loaded {len(embeddings_list)} face embeddings")
    
    def capture_face(self):
        """
        Capture current frame and extract face embedding.
        
        Returns:
            dict with 'success', 'embedding', 'message'
        """
        with self.frame_lock:
            if self.current_frame is None:
                return {'success': False, 'embedding': None, 'message': 'No camera feed'}
            frame = self.current_frame.copy()
        
        # Detect face
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(frame_rgb)
        
        if not results.detections:
            return {'success': False, 'embedding': None, 'message': 'No face detected'}
        
        if len(results.detections) > 1:
            return {'success': False, 'embedding': None, 'message': 'Multiple faces detected. Please ensure only one face is visible.'}
        
        # Get face bounding box
        detection = results.detections[0]
        h, w = frame.shape[:2]
        bbox = detection.location_data.relative_bounding_box
        
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        box_w = int(bbox.width * w)
        box_h = int(bbox.height * h)
        
        # Extract face with padding
        pad = 30
        y1 = max(0, y - pad)
        y2 = min(h, y + box_h + pad)
        x1 = max(0, x - pad)
        x2 = min(w, x + box_w + pad)
        
        face_region = frame_rgb[y1:y2, x1:x2]
        
        # Get embedding
        embedding = self._get_embedding(face_region)
        
        if embedding is None:
            return {'success': False, 'embedding': None, 'message': 'Failed to generate face embedding'}
        
        return {
            'success': True,
            'embedding': embedding.tolist(),
            'message': 'Face captured successfully'
        }
    
    def clear_results(self):
        """Clear cached recognition results."""
        with self.face_results_lock:
            self.face_results.clear()


# Singleton instance
_engine_instance = None
_engine_lock = Lock()


def get_engine():
    """Get or create the singleton CameraEngine instance."""
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            _engine_instance = CameraEngine()
        return _engine_instance
