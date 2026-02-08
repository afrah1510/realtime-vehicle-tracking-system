"""
Real-Time Vehicle & License Plate Recognition System - ENHANCED
================================================================
Features:
- Dual YOLO detection (vehicle type + license plates)
- Best frame selection based on detection quality
- Tesseract OCR with two-line plate support
- Advanced preprocessing pipeline from second code
- Real-time webcam processing
- Video output saving
- MySQL database logging (PERSISTENT CONNECTION)
- Improved detection for stationary vehicles
- LICENSE PLATE DISPLAYED ON VIDEO
"""

import cv2
import numpy as np
import pytesseract
from datetime import datetime
import argparse
import re
from collections import defaultdict, deque
from ultralytics import YOLO
from scipy.spatial.distance import euclidean
import time
import mysql.connector
from mysql.connector import Error
import os
import platform


# ============================================
# TESSERACT CONFIGURATION (CRITICAL!)
# ============================================
# Auto-detect Tesseract path based on OS
if platform.system() == 'Windows':
    # Common Windows installation paths
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    ]
    
    tesseract_found = False
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            print(f"✓ Tesseract found at: {path}")
            tesseract_found = True
            break
    
    if not tesseract_found:
        print("=" * 60)
        print("ERROR: Tesseract OCR not found!")
        print("Searched paths:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nPlease install Tesseract from:")
        print("https://github.com/UB-Mannheim/tesseract/wiki")
        print("=" * 60)
        exit(1)

elif platform.system() == 'Linux':
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'
    
elif platform.system() == 'Darwin':  # macOS
    pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

# Verify Tesseract is accessible
try:
    tesseract_version = pytesseract.get_tesseract_version()
    print(f"✓ Tesseract OCR Version: {tesseract_version}")
except Exception as e:
    print(f"✗ Tesseract OCR verification failed: {e}")
    print("  Please ensure Tesseract is properly installed")
    exit(1)


# ============================================
# CONFIGURATION & CONSTANTS
# ============================================
INDIAN_STATE_CODES = {
    'AN': 'Andaman & Nicobar', 'AP': 'Andhra Pradesh', 'AR': 'Arunachal Pradesh',
    'AS': 'Assam', 'BR': 'Bihar', 'CG': 'Chhattisgarh', 'CH': 'Chandigarh',
    'DD': 'Daman & Diu', 'DL': 'Delhi', 'DN': 'Dadra & Nagar Haveli',
    'GA': 'Goa', 'GJ': 'Gujarat', 'HP': 'Himachal Pradesh', 'HR': 'Haryana',
    'JH': 'Jharkhand', 'JK': 'Jammu & Kashmir', 'KA': 'Karnataka', 'KL': 'Kerala',
    'LA': 'Ladakh', 'LD': 'Lakshadweep', 'MH': 'Maharashtra', 'ML': 'Meghalaya',
    'MN': 'Manipur', 'MP': 'Madhya Pradesh', 'MZ': 'Mizoram', 'NL': 'Nagaland',
    'OD': 'Odisha', 'OR': 'Odisha', 'PB': 'Punjab', 'PY': 'Puducherry',
    'RJ': 'Rajasthan', 'SK': 'Sikkim', 'TN': 'Tamil Nadu', 'TR': 'Tripura',
    'TS': 'Telangana', 'UK': 'Uttarakhand', 'UP': 'Uttar Pradesh', 'WB': 'West Bengal',
}

DIGIT_TO_LETTER = {
    '0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', 
    '6': 'G', '8': 'B', '9': 'Q'
}

LETTER_TO_DIGIT = {
    'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'A': '4', 
    'S': '5', 'G': '6', 'B': '8', 'D': '0', 'T': '7',
    'Q': '9', 'U': '0', 'J': '1'
}

NOISE_PATTERNS = [
    r'\bIND\b', r'\bINDIA\b', r'\bHSRP\b',
    r'[®™©\*\#\@]',
    r'^\s+', r'\s+$',
]

REJECT_WORDS = [
    'TYPE', 'PLATE', 'NUMBER', 'VEHICLE', 'PRIVATE', 'COMMERCIAL',
    'REGISTRATION', 'CERTIFICATE', 'TEMPORARY', 'TRANSPORT',
]

PLATE_PATTERNS = {
    "standard": r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$',
    "old_format": r'^[A-Z]{2}\d{1,2}\d{4}$',
    "bharat_series": r'^\d{2}BH\d{4}[A-Z]{2}$',
    "diplomatic": r'^\d{2,3}(CD|CC|UN)\d{1,4}$',
    "military": r'^\d{2,3}[A-Z]\d{5,6}[A-Z]?$',
    "special_number": r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4,5}$',
}


# ============================================
# DATABASE MANAGER - PERSISTENT CONNECTION
# ============================================
class DatabaseManager:
    """Manages MySQL database connections and logging with connection persistence"""
    
    def __init__(self, host='localhost', database='vehicle_recognition', 
                 user='root', password='12345'):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.connection = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3
        self.create_database_and_table()
    
    def create_connection(self):
        """Create database connection with reconnection logic"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                autocommit=True,
                connection_timeout=10
            )
            self.reconnect_attempts = 0
            return True
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            return False
    
    def ensure_connection(self):
        """Ensure database connection is alive, reconnect if needed"""
        try:
            if self.connection is None:
                return self.create_connection()
            
            self.connection.ping(reconnect=True, attempts=3, delay=1)
            return True
            
        except Error as e:
            print(f"! Connection lost, attempting to reconnect... ({e})")
            self.connection = None
            
            if self.reconnect_attempts < self.max_reconnect_attempts:
                self.reconnect_attempts += 1
                return self.create_connection()
            else:
                print(f"! Max reconnection attempts ({self.max_reconnect_attempts}) reached")
                return False
    
    def create_database_and_table(self):
        """Create database and table if they don't exist"""
        try:
            if not self.create_connection():
                print("! MySQL logging disabled - connection failed")
                return
            
            cursor = self.connection.cursor()
            
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            cursor.execute(f"USE {self.database}")
            
            create_table_query = """
            CREATE TABLE IF NOT EXISTS detected_vehicles (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME NOT NULL,
                vehicle_number VARCHAR(20) NOT NULL,
                vehicle_type VARCHAR(50) NOT NULL,
                plate_confidence FLOAT,
                vehicle_confidence FLOAT,
                quality_score FLOAT,
                state_code VARCHAR(5),
                state_name VARCHAR(100),
                detection_type VARCHAR(20),
                INDEX idx_vehicle_number (vehicle_number),
                INDEX idx_timestamp (timestamp)
            )
            """
            cursor.execute(create_table_query)
            
            cursor.close()
            print("✓ MySQL database initialized successfully")
            print("✓ Database connection will remain persistent")
            
        except Error as e:
            print(f"! MySQL initialization error: {e}")
            print("! MySQL logging disabled")
            self.connection = None
    
    def log_detection(self, vehicle_number, vehicle_type, plate_conf, 
                     vehicle_conf, quality_score, state_code=None, 
                     state_name=None, detection_type="EXIT"):
        """Log a vehicle detection to database with connection persistence"""
        if not self.ensure_connection():
            return False
        
        try:
            cursor = self.connection.cursor()
            
            insert_query = """
            INSERT INTO detected_vehicles 
            (timestamp, vehicle_number, vehicle_type, plate_confidence, 
             vehicle_confidence, quality_score, state_code, state_name, detection_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                datetime.now(),
                vehicle_number,
                vehicle_type,
                float(plate_conf),
                float(vehicle_conf),
                float(quality_score),
                state_code,
                state_name,
                detection_type
            )
            
            cursor.execute(insert_query, values)
            cursor.close()
            return True
            
        except Error as e:
            print(f"! Database logging error: {e}")
            self.connection = None
            return False
    
    def __del__(self):
        """Destructor - only close connection when object is destroyed"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("✓ MySQL connection closed (system shutdown)")


# ============================================
# PREPROCESSING FUNCTIONS
# ============================================
def preprocess_plate(image, debug=False):
    """Enhanced preprocessing pipeline for 2-line plates"""
    processed = {"original": image.copy()}
    
    h, w = image.shape[:2]
    
    if w < 400:
        scale = 400 / w
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    elif h < 100:
        scale = 100 / h
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    processed["resized"] = image.copy()
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    processed["gray"] = gray
    
    denoised = cv2.bilateralFilter(gray, 11, 17, 17)
    
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    processed["enhanced"] = enhanced
    
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed["otsu"] = otsu
    
    adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 31, 10)
    processed["adaptive"] = adaptive
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    processed["morph"] = morph
    
    kernel_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel_sharp)
    processed["sharpened"] = sharpened
    
    if np.mean(gray) < 100:
        gamma = 1.5
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(gray, table)
        gamma_enhanced = clahe.apply(gamma_corrected)
        processed["gamma"] = gamma_enhanced
    
    if np.mean(enhanced) < 127:
        processed["inverted"] = cv2.bitwise_not(enhanced)
    
    return processed


def deskew_plate(image):
    """Correct skewed plates"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                            minLineLength=image.shape[1]//4, maxLineGap=10)
    
    if lines is None:
        return image
    
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 45:
            angles.append(angle)
    
    if not angles:
        return image
    
    median_angle = np.median(angles)
    
    if abs(median_angle) > 1.0:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), 
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    return image


def is_two_line_plate(image):
    """Detect if plate is two-line format with improved logic"""
    h, w = image.shape[:2]
    aspect = w / h if h > 0 else 0
    
    return aspect < 3.2


def split_two_line_plate(image):
    """Split two-line plate into top and bottom halves with IMPROVED algorithm"""
    h, w = image.shape[:2]
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    white_pixels = cv2.countNonZero(binary)
    total_pixels = h * w
    
    horizontal_proj = np.sum(binary, axis=1)
    
    start = int(h * 0.35)
    end = int(h * 0.65)
    mid_proj = horizontal_proj[start:end]
    
    if len(mid_proj) == 0:
        return image[:h//2, :], image[h//2:, :]
        
    split_index = 0
    
    if white_pixels > total_pixels / 2:
        split_index = np.argmin(mid_proj)
    else:
        split_index = np.argmax(mid_proj)
    
    split_point = start + split_index
    
    if split_index < len(mid_proj) * 0.2 or split_index > len(mid_proj) * 0.8:
        split_point = h // 2
    
    overlap = 8
    split_point = max(overlap, min(h - overlap, split_point))
    
    top = image[:split_point + overlap, :]
    bottom = image[split_point - overlap:, :]
    
    return top, bottom


# ============================================
# FRAME CANDIDATE CLASS
# ============================================
class FrameCandidate:
    """Stores frame data for quality comparison"""
    
    def __init__(self, plate_number, vehicle_type, plate_conf, vehicle_conf, 
                 plate_roi, ocr_clarity, state_code=None, state_name=None):
        self.plate_number = plate_number
        self.vehicle_type = vehicle_type
        self.plate_conf = plate_conf
        self.vehicle_conf = vehicle_conf
        self.plate_roi = plate_roi
        self.ocr_clarity = ocr_clarity
        self.state_code = state_code
        self.state_name = state_name
        self.timestamp = datetime.now()
        
        self.quality_score = self._calculate_quality()
    
    def _calculate_quality(self):
        """Calculate overall quality score"""
        score = 0.0
        
        score += self.plate_conf * 0.4
        score += self.vehicle_conf * 0.2
        
        expected_length = 10
        length_diff = abs(len(self.plate_number) - expected_length)
        clarity_score = max(0, 1 - (length_diff / expected_length))
        score += clarity_score * 0.3
        
        if self.plate_roi.size > 0:
            h, w = self.plate_roi.shape[:2]
            area = h * w
            size_score = min(1.0, area / 50000)
            score += size_score * 0.1
        
        return score


# ============================================
# VEHICLE TRACKER
# ============================================
class VehicleTracker:
    """Enhanced tracker with periodic detection for stationary vehicles"""
    
    def __init__(self, max_disappeared=30, max_distance=150, min_frames_before_log=3,
                 quality_threshold=0.60, periodic_check_frames=30):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.vehicle_info = {}
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_frames_before_log = min_frames_before_log
        
        self.quality_threshold = quality_threshold
        self.periodic_check_frames = periodic_check_frames
        
        self.frame_candidates = defaultdict(list)
        self.frame_count = defaultdict(int)
        
        self.last_logged_frame = {}
        self.logged_plates = set()
        
    def register(self, centroid, vehicle_type):
        """Register new vehicle"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.vehicle_info[self.next_object_id] = {
            'type': vehicle_type,
            'plate': None,
            'confidence': 0.0
        }
        self.frame_count[self.next_object_id] = 0
        self.last_logged_frame[self.next_object_id] = -1
        self.next_object_id += 1
        
    def deregister(self, object_id):
        """Remove tracked vehicle and return best frame if available"""
        best_candidate = None
        
        if object_id in self.frame_candidates and len(self.frame_candidates[object_id]) > 0:
            if self.frame_count[object_id] >= self.min_frames_before_log:
                if self.last_logged_frame[object_id] == -1:
                    candidates = self.frame_candidates[object_id]
                    best_candidate = max(candidates, key=lambda x: x.quality_score)
        
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.vehicle_info[object_id]
        if object_id in self.frame_candidates:
            del self.frame_candidates[object_id]
        if object_id in self.frame_count:
            del self.frame_count[object_id]
        if object_id in self.last_logged_frame:
            del self.last_logged_frame[object_id]
        
        return best_candidate
    
    def check_for_detection(self, object_id):
        """Check if vehicle should be detected now (periodic or quality-based)"""
        if self.frame_count[object_id] < self.min_frames_before_log:
            return None
        
        if object_id not in self.frame_candidates or len(self.frame_candidates[object_id]) == 0:
            return None
        
        if self.last_logged_frame[object_id] != -1:
            return None
        
        candidates = self.frame_candidates[object_id]
        best_candidate = max(candidates, key=lambda x: x.quality_score)
        
        if best_candidate.plate_number in self.logged_plates:
            return None
        
        current_frame = self.frame_count[object_id]
        
        if best_candidate.quality_score >= self.quality_threshold:
            return best_candidate
        
        if current_frame % self.periodic_check_frames == 0:
            if best_candidate.quality_score >= 0.5:
                return best_candidate
        
        return None
        
    def update(self, detections):
        """Update tracker with new detections"""
        if len(detections) == 0:
            deregistered_candidates = []
            periodic_candidates = []
            
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    candidate = self.deregister(object_id)
                    if candidate:
                        deregistered_candidates.append((object_id, candidate))
            
            return self.get_tracked_objects(), deregistered_candidates, periodic_candidates
        
        deregistered_candidates = []
        periodic_candidates = []
        
        if len(self.objects) == 0:
            for centroid, vehicle_type, _, conf in detections:
                self.register(centroid, vehicle_type)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            detection_centroids = [d[0] for d in detections]
            
            D = np.zeros((len(object_centroids), len(detection_centroids)))
            for i, oc in enumerate(object_centroids):
                for j, dc in enumerate(detection_centroids):
                    D[i, j] = euclidean(oc, dc)
            
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                    
                if D[row, col] > self.max_distance:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id] = detection_centroids[col]
                self.disappeared[object_id] = 0
                
                self.vehicle_info[object_id]['type'] = detections[col][1]
                self.vehicle_info[object_id]['confidence'] = detections[col][3]
                
                self.frame_count[object_id] += 1
                
                candidate = self.check_for_detection(object_id)
                if candidate:
                    periodic_candidates.append((object_id, candidate))
                    self.last_logged_frame[object_id] = self.frame_count[object_id]
                    self.logged_plates.add(candidate.plate_number)
                
                used_rows.add(row)
                used_cols.add(col)
            
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    candidate = self.deregister(object_id)
                    if candidate:
                        deregistered_candidates.append((object_id, candidate))
            
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(detection_centroids[col], detections[col][1])
        
        return self.get_tracked_objects(), deregistered_candidates, periodic_candidates
    
    def get_tracked_objects(self):
        """Get all tracked objects"""
        result = {}
        
        for object_id, centroid in self.objects.items():
            logged_status = "✓" if self.last_logged_frame[object_id] != -1 else ""
            
            result[object_id] = {
                'centroid': centroid,
                'type': self.vehicle_info[object_id]['type'],
                'plate': self.vehicle_info[object_id]['plate'],
                'confidence': self.vehicle_info[object_id]['confidence'],
                'frames_tracked': self.frame_count[object_id],
                'logged': logged_status
            }
        
        return result
    
    def add_frame_candidate(self, object_id, plate_number, vehicle_type,
                           plate_conf, vehicle_conf, plate_roi, ocr_clarity,
                           state_code=None, state_name=None):
        """Add a frame candidate for this vehicle"""
        if object_id not in self.objects:
            return
        
        candidate = FrameCandidate(
            plate_number, vehicle_type,
            plate_conf, vehicle_conf, plate_roi, ocr_clarity,
            state_code, state_name
        )
        
        self.frame_candidates[object_id].append(candidate)
        self.vehicle_info[object_id]['plate'] = plate_number


# ============================================
# INDIAN PLATE VALIDATOR
# ============================================
class IndianPlateValidator:
    """Enhanced validator for Indian license plates"""
    
    @staticmethod
    def clean(text):
        """Remove noise and clean text"""
        if not text:
            return ""
        
        text = text.upper().strip()
        
        for pattern in NOISE_PATTERNS:
            text = re.sub(pattern, '', text)
        
        text = re.sub(r'[^A-Z0-9]', '', text)
        
        return text
    
    @staticmethod
    def should_reject(text):
        """Check if text should be rejected"""
        clean = IndianPlateValidator.clean(text)
        
        if len(clean) < 6:
            return True, f"Too short ({len(clean)} chars)"
        if len(clean) > 13:
            return True, f"Too long ({len(clean)} chars)"
        
        letters = sum(1 for c in clean if c.isalpha())
        digits = sum(1 for c in clean if c.isdigit())
        
        if letters > 7:
            return True, f"Too many letters ({letters})"
        if digits < 2:
            return True, f"Too few digits ({digits})"
        if letters == len(clean) or digits == len(clean):
            return True, "All same type"
        
        for word in REJECT_WORDS:
            if word in clean and len(word) > 3:
                return True, f"Contains '{word}'"
        
        return False, ""
    
    @staticmethod
    def smart_correct(text):
        """Smart OCR correction using pattern-based structure analysis"""
        clean_text = IndianPlateValidator.clean(text)
        if len(clean_text) < 6:
            return clean_text
        
        to_letter = {'0': 'O', '8': 'B', '1': 'I', '5': 'S', '6': 'G'}
        to_digit = {'O': '0', 'B': '8', 'I': '1', 'S': '5', 'G': '6', 'D': '0', 'L': '1', 'Z': '2'}
        
        best_result = clean_text
        best_score = 0
        
        for series_len in [1, 2, 3]:
            if len(clean_text) < 4 + series_len + 1:
                continue
            
            state_raw = clean_text[0:2]
            district_raw = clean_text[2:4]
            series_raw = clean_text[4:4+series_len]
            number_raw = clean_text[4+series_len:]
            
            if not number_raw:
                continue
            
            state = ''.join(to_letter.get(c, c) if c.isdigit() else c for c in state_raw)
            district = ''.join(to_digit.get(c, c) if c.isalpha() else c for c in district_raw)
            series = ''.join(to_letter.get(c, c) if c.isdigit() else c for c in series_raw)
            number = ''.join(to_digit.get(c, c) if c.isalpha() else c for c in number_raw)
            
            result = state + district + series + number
            
            score = 0
            
            if state in INDIAN_STATE_CODES:
                score += 10
            
            pattern = r'^[A-Z]{2}\d{2}[A-Z]{' + str(series_len) + r'}\d{' + str(len(number)) + r'}$'
            if re.match(pattern, result):
                score += 5
            
            for p in PLATE_PATTERNS.values():
                if re.match(p, result):
                    score += 5
                    break
            
            if len(number) == 4 and number.isdigit():
                score += 3
            
            if score > best_score:
                best_score = score
                best_result = result
        
        return best_result
    
    @staticmethod
    def extract_plate_from_noise(text):
        """Extract valid plate from noisy OCR output"""
        clean = IndianPlateValidator.clean(text)
        if len(clean) < 8:
            return clean
        
        STATE_CHAR_FIXES = {
            'N': 'H', '0': 'O', '1': 'I', '8': 'B', '5': 'S', '6': 'G',
        }
        COMMON_STATES = {'MH', 'DL', 'KA', 'TN', 'GJ', 'RJ', 'UP', 'HR', 'MP', 'AP', 'TS', 'WB', 'PB'}
        
        candidates = []
        for start in range(min(4, len(clean) - 7)):
            potential = clean[start:]
            state_code = potential[:2]
            
            if state_code == 'HH':
                fixed_state = 'MH'
                score = 15
                candidates.append((fixed_state + potential[2:], score, fixed_state))
            
            if state_code in INDIAN_STATE_CODES:
                score = 10 if state_code in COMMON_STATES else 5
                candidates.append((potential, score, state_code))
            
            chars = list(state_code)
            for i in range(2):
                if chars[i] in STATE_CHAR_FIXES:
                    fixed = list(chars)
                    fixed[i] = STATE_CHAR_FIXES[chars[i]]
                    fixed_state = ''.join(fixed)
                    if fixed_state in INDIAN_STATE_CODES:
                        score = 12 if fixed_state in COMMON_STATES else 6
                        candidates.append((fixed_state + potential[2:], score, fixed_state))
            
            fixed_both = ''.join(STATE_CHAR_FIXES.get(c, c) for c in chars)
            if fixed_both != state_code and fixed_both in INDIAN_STATE_CODES:
                score = 11 if fixed_both in COMMON_STATES else 5
                candidates.append((fixed_both + potential[2:], score, fixed_both))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return clean
    
    @staticmethod
    def get_state(text):
        """Get state from plate text with fuzzy matching"""
        clean = IndianPlateValidator.clean(text)
        if len(clean) < 2:
            return None, None
        
        code = clean[:2]
        
        if code in INDIAN_STATE_CODES:
            return code, INDIAN_STATE_CODES[code]
        
        corrected_code = ""
        for c in code:
            if c.isdigit() and c in DIGIT_TO_LETTER:
                corrected_code += DIGIT_TO_LETTER[c]
            else:
                corrected_code += c
        
        if corrected_code in INDIAN_STATE_CODES:
            return corrected_code, INDIAN_STATE_CODES[corrected_code]
        
        for state in INDIAN_STATE_CODES:
            diff = sum(c1 != c2 for c1, c2 in zip(code, state))
            if diff == 1:
                return state, INDIAN_STATE_CODES[state]
        
        return None, None
    
    @staticmethod
    def match_pattern(text):
        """Match against known plate patterns"""
        clean = IndianPlateValidator.clean(text)
        for name, pattern in PLATE_PATTERNS.items():
            if re.match(pattern, clean):
                return True, name
        return False, ""
    
    @staticmethod
    def format_plate(text):
        """Format plate for display: SS DD LLL NNNN"""
        clean = IndianPlateValidator.clean(text)
        if len(clean) < 6:
            return clean
        
        if 'BH' in clean:
            m = re.match(r'^(\d{2})(BH)(\d{4})([A-Z]{2})$', clean)
            if m:
                return f"{m.group(1)} {m.group(2)} {m.group(3)} {m.group(4)}"
        
        for code in ['CD', 'CC', 'UN']:
            if code in clean:
                m = re.match(rf'^(\d{{2,3}})({code})(\d{{1,4}})$', clean)
                if m:
                    return f"{m.group(1)} {m.group(2)} {m.group(3)}"
        
        state = clean[:2]
        rest = clean[2:]
        
        district = ""
        i = 0
        while i < len(rest) and i < 2 and rest[i].isdigit():
            district += rest[i]
            i += 1
        rest = rest[i:]
        
        series = ""
        i = 0
        while i < len(rest) and i < 3 and rest[i].isalpha():
            series += rest[i]
            i += 1
        number = rest[i:]
        
        parts = [p for p in [state, district, series, number] if p]
        return ' '.join(parts)
    
    @staticmethod
    def validate(text, debug=False):
        """Full validation pipeline"""
        result = {
            "is_valid": False,
            "original": text,
            "corrected": "",
            "formatted": "",
            "state_code": None,
            "state_name": None,
            "plate_type": None,
            "rejection_reason": None,
        }
        
        if not text:
            result["rejection_reason"] = "Empty"
            return result
        
        cleaned = IndianPlateValidator.clean(text)
        cleaned = IndianPlateValidator.extract_plate_from_noise(cleaned)
        
        reject, reason = IndianPlateValidator.should_reject(cleaned)
        if reject:
            result["rejection_reason"] = reason
            return result
        
        corrected = IndianPlateValidator.smart_correct(cleaned)
        result["corrected"] = corrected
        
        is_bharat = 'BH' in corrected and re.match(r'^\d{2}BH\d{4}[A-Z]{2}$', corrected)
        is_diplomatic = bool(re.match(r'^\d{2,3}(CD|CC|UN)\d{1,4}$', corrected))
        is_military = bool(re.match(r'^\d{2,3}[A-Z]\d{5,6}[A-Z]?$', corrected))
        is_special = is_bharat or is_diplomatic or is_military
        
        if not is_special:
            state_code, state_name = IndianPlateValidator.get_state(corrected)
            if not state_code:
                result["rejection_reason"] = f"Invalid state: {corrected[:2]}"
                return result
            result["state_code"] = state_code
            result["state_name"] = state_name
        
        matches, plate_type = IndianPlateValidator.match_pattern(corrected)
        if matches:
            result["is_valid"] = True
            result["plate_type"] = plate_type
            result["formatted"] = IndianPlateValidator.format_plate(corrected)
        else:
            if result.get("state_code"):
                digits = sum(1 for c in corrected if c.isdigit())
                letters = sum(1 for c in corrected if c.isalpha())
                if 3 <= digits <= 6 and 2 <= letters <= 5:
                    result["is_valid"] = True
                    result["plate_type"] = "non_standard"
                    result["formatted"] = IndianPlateValidator.format_plate(corrected)
                else:
                    result["rejection_reason"] = "Invalid format"
            else:
                result["rejection_reason"] = "Invalid format"
        
        return result
    
    @staticmethod
    def calculate_ocr_clarity(raw_text, validated_text):
        """Calculate OCR clarity score"""
        if not raw_text or not validated_text:
            return 1.0
        
        raw_clean = IndianPlateValidator.clean(raw_text)
        diff_count = sum(1 for a, b in zip(raw_clean, validated_text) if a != b)
        diff_count += abs(len(raw_clean) - len(validated_text))
        
        clarity = diff_count / max(len(validated_text), 1)
        return min(1.0, clarity)


# ============================================
# MAIN RECOGNITION SYSTEM
# ============================================
class VehicleRecognitionSystem:
    """Main system integrating all components"""
    
    def __init__(self, plate_model_path, vehicle_model_path, 
                 min_frames=3, quality_threshold=0.60, periodic_check=30,
                 db_host='localhost', db_name='vehicle_recognition', 
                 db_user='root', db_password='12345'):
        print("="*60)
        print("Initializing Vehicle Recognition System - ENHANCED")
        print("="*60)
        
        print(f"\nLoading plate detection model: {plate_model_path}")
        self.plate_detector = YOLO(plate_model_path)
        print("✓ Plate detector loaded!")
        
        print(f"\nLoading vehicle classification model: {vehicle_model_path}")
        self.vehicle_detector = YOLO(vehicle_model_path)
        print("✓ Vehicle detector loaded!")
        
        print("\nUsing Tesseract OCR (2-line support)...")
        self.tesseract_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        self.tesseract_config_single = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        self.tracker = VehicleTracker(
            max_disappeared=30,
            max_distance=150,
            min_frames_before_log=min_frames,
            quality_threshold=quality_threshold,
            periodic_check_frames=periodic_check
        )
        
        self.validator = IndianPlateValidator()
        
        try:
            self.db = DatabaseManager(db_host, db_name, db_user, db_password)
        except Exception as e:
            print(f"! Database initialization failed: {e}")
            self.db = None
        
        self.detected_plates = set()
        
        print("\n" + "="*60)
        print("✓ System ready!")
        print("="*60)
        print("OCR Engine: TESSERACT (ENHANCED 2-LINE SUPPORT)")
        print("Database: PERSISTENT CONNECTION (never closes)")
        print("Enhanced Detection Settings:")
        print(f"  - Minimum frames before detection: {min_frames}")
        print(f"  - Quality threshold (early detect): {quality_threshold}")
        print(f"  - Periodic check interval: {periodic_check} frames")
        print("  - Quality factors: Plate conf, Vehicle conf, OCR clarity, ROI size")
        print("\nDetection Modes:")
        print("  1. HIGH QUALITY: Immediate detection when quality >= threshold")
        print(f"  2. PERIODIC: Detection every {periodic_check} frames for stationary vehicles")
        print("  3. EXIT: Detection when vehicle leaves frame")
        print("\nPreprocessing Pipeline (2-LINE OPTIMIZED):")
        print("  - Deskewing, CLAHE, Bilateral Filter")
        print("  - Otsu + Adaptive Thresholding")
        print("  - Morphological Operations")
        print("  - Gamma Correction for low light")
        print("  - SMART two-line plate splitting")
        print("  - PSM 6 mode for 2-line plates")
        print("="*60 + "\n")
    
    def _read_single(self, plate_img, is_two_line=False):
        """Read single plate region with multiple preprocessing attempts"""
        processed = preprocess_plate(plate_img)
        
        candidates = []
        
        config = self.tesseract_config if is_two_line else self.tesseract_config_single
        
        versions_to_try = ["resized", "enhanced", "otsu", "adaptive", "morph", "sharpened"]
        if "gamma" in processed:
            versions_to_try.append("gamma")
        if "inverted" in processed:
            versions_to_try.append("inverted")
        
        for version in versions_to_try:
            if version not in processed:
                continue
            
            try:
                text = pytesseract.image_to_string(processed[version], config=config)
                text = text.strip()
                
                if text:
                    clean_text = IndianPlateValidator.clean(text)
                    validation = IndianPlateValidator.validate(text)
                    
                    candidates.append({
                        "text": text,
                        "clean": clean_text,
                        "valid": validation["is_valid"],
                        "corrected": validation.get("corrected", clean_text),
                        "formatted": validation.get("formatted", clean_text),
                        "validation": validation,
                        "version": version
                    })
            except Exception as e:
                continue
        
        valid_candidates = [c for c in candidates if c["valid"]]
        
        if valid_candidates:
            best = valid_candidates[0]
            return best["corrected"], best["validation"]
        
        if candidates:
            best = candidates[0]
            return best["clean"], best["validation"]
        
        return "", {"is_valid": False}
    
    def read_plate_multipass(self, plate_img):
        """Multi-pass OCR with preprocessing variants and ENHANCED 2-line support"""
        plate_img = deskew_plate(plate_img)
        
        if is_two_line_plate(plate_img):
            top, bottom = split_two_line_plate(plate_img)
            
            combined_text, combined_validation = self._read_single(plate_img, is_two_line=True)
            
            top_text, top_validation = self._read_single(top, is_two_line=False)
            bottom_text, bottom_validation = self._read_single(bottom, is_two_line=False)
            
            split_combined = top_text + bottom_text
            split_validation = self.validator.validate(split_combined)
            
            if combined_validation["is_valid"]:
                return combined_text, combined_validation
            elif split_validation["is_valid"]:
                return split_combined, split_validation
            elif combined_text:
                return combined_text, combined_validation
            else:
                return split_combined, split_validation
        
        return self._read_single(plate_img, is_two_line=False)
    
    def process_frame(self, frame):
        """Process single frame"""
        h, w = frame.shape[:2]
        
        vehicle_results = self.vehicle_detector(frame, conf=0.3, verbose=False)[0]
        
        detections = []
        vehicle_boxes = {}
        
        for box in vehicle_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            vehicle_type = vehicle_results.names[cls]
            
            if vehicle_type == 'motorcycle':
                vehicle_type = 'motorbike'
            
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            
            detections.append((centroid, vehicle_type, (x1, y1, x2, y2), conf))
            vehicle_boxes[centroid] = (x1, y1, x2, y2, conf)
        
        tracked_objects, deregistered_candidates, periodic_candidates = self.tracker.update(detections)
        
        for obj_id, candidate in deregistered_candidates:
            if candidate.plate_number not in self.detected_plates:
                self.detected_plates.add(candidate.plate_number)
                print(f"✓ EXIT DETECT (Q:{candidate.quality_score:.2f}): {candidate.plate_number} | {candidate.vehicle_type}")
                
                if self.db:
                    self.db.log_detection(
                        candidate.plate_number,
                        candidate.vehicle_type,
                        candidate.plate_conf,
                        candidate.vehicle_conf,
                        candidate.quality_score,
                        candidate.state_code,
                        candidate.state_name,
                        "EXIT"
                    )
        
        for obj_id, candidate in periodic_candidates:
            if candidate.plate_number not in self.detected_plates:
                self.detected_plates.add(candidate.plate_number)
                detection_type = "HIGH-Q" if candidate.quality_score >= self.tracker.quality_threshold else "PERIODIC"
                print(f"✓ {detection_type} DETECT (Q:{candidate.quality_score:.2f}): {candidate.plate_number} | {candidate.vehicle_type}")
                
                if self.db:
                    self.db.log_detection(
                        candidate.plate_number,
                        candidate.vehicle_type,
                        candidate.plate_conf,
                        candidate.vehicle_conf,
                        candidate.quality_score,
                        candidate.state_code,
                        candidate.state_name,
                        detection_type
                    )
        
        plate_results = self.plate_detector(frame, conf=0.25, verbose=False)[0]
        plate_boxes = []
        
        for box in plate_results.boxes:
            px1, py1, px2, py2 = map(int, box.xyxy[0])
            plate_conf = float(box.conf[0])
            
            plate_roi = frame[py1:py2, px1:px2]
            
            if plate_roi.size == 0:
                continue
            
            validated_plate = None  # FIXED: Initialize before validation
            state_code = None
            state_name = None
            
            raw_text, validation = self.read_plate_multipass(plate_roi)
            
            # DEBUG OUTPUT
            if raw_text or validation.get("corrected"):
                print(f"DEBUG OCR - Raw: '{raw_text}' | Valid: {validation['is_valid']} | Conf: {plate_conf:.2f}")
            
            if validation["is_valid"]:
                validated_plate = validation["formatted"]
                state_code = validation.get("state_code")
                state_name = validation.get("state_name")
                
                print(f"  → Validated: {validated_plate}")
                
                ocr_clarity = self.validator.calculate_ocr_clarity(raw_text, validation["corrected"])
                
                plate_center = (int((px1 + px2) / 2), int((py1 + py2) / 2))
                
                min_dist = float('inf')
                closest_id = None
                
                for obj_id, obj_data in tracked_objects.items():
                    dist = euclidean(plate_center, obj_data['centroid'])
                    if dist < min_dist:
                        min_dist = dist
                        closest_id = obj_id
                
                if closest_id is not None and min_dist < 200:
                    obj_data = tracked_objects[closest_id]
                    
                    self.tracker.add_frame_candidate(
                        closest_id,
                        validated_plate,
                        obj_data['type'],
                        plate_conf,
                        obj_data['confidence'],
                        plate_roi,
                        ocr_clarity,
                        state_code,
                        state_name
                    )
            
            plate_boxes.append({
                'bbox': (px1, py1, px2, py2),
                'conf': plate_conf,
                'text': validated_plate  # Will be None if validation failed
            })
        
        return tracked_objects, plate_boxes
    
    def draw_annotations(self, frame, tracked_objects, plate_boxes):
        """Draw tracking info and PROMINENTLY DISPLAY license plate numbers"""
        h, w = frame.shape[:2]
        
        for plate_info in plate_boxes:
            px1, py1, px2, py2 = plate_info['bbox']
            conf = plate_info['conf']
            text = plate_info['text']
            
            color = (0, 255, 0) if text else (0, 255, 255)
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
            
            if text:
                plate_label = f"{text}"
                
                (label_width, label_height), baseline = cv2.getTextSize(
                    plate_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                )
                
                label_y = py1 - 10
                if label_y < label_height + 10:
                    label_y = py2 + label_height + 10
                
                cv2.rectangle(frame,
                             (px1, label_y - label_height - 5),
                             (px1 + label_width + 10, label_y + 5),
                             (0, 255, 255), -1)
                
                cv2.putText(frame, plate_label,
                           (px1 + 5, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            conf_label = f"{conf:.2f}"
            cv2.putText(frame, conf_label, (px1, py2 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        for obj_id, data in tracked_objects.items():
            cx, cy = data['centroid']
            vehicle_type = data['type']
            plate = data['plate']
            frames_tracked = data['frames_tracked']
            logged = data['logged']
            
            color = (0, 255, 0) if logged else (255, 128, 0)
            
            cv2.circle(frame, (cx, cy), 5, color, -1)
            
            label = f"ID:{obj_id} {logged} | {vehicle_type} | F:{frames_tracked}"
            if plate:
                label += f" | {plate}"
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            cv2.rectangle(frame, 
                         (cx - 100, cy - 35), 
                         (cx - 100 + text_width + 10, cy - 35 + text_height + 10),
                         (255, 255, 255), -1)
            
            cv2.putText(frame, label, (cx - 95, cy - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        stats_label = f"Tracked: {len(tracked_objects)} | Detected: {len(self.detected_plates)}"
        (stats_width, stats_height), baseline = cv2.getTextSize(
            stats_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        
        cv2.rectangle(frame, (5, 5), (15 + stats_width, 15 + stats_height), (255, 255, 255), -1)
        
        cv2.putText(frame, stats_label, 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return frame
    
    def run_webcam(self, camera_id=0, video_path=None, display_width=1280, save_output=True):
        """Run real-time detection on webcam or video file"""
        if video_path:
            cap = cv2.VideoCapture(video_path)
            print(f"Loading video: {video_path}")
        else:
            cap = cv2.VideoCapture(camera_id)
            print(f"Opening camera: {camera_id}")
        
        if not cap.isOpened():
            print("✗ Cannot open camera")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = None
        if save_output:
            if video_path:
                filename = os.path.basename(video_path)
                name, ext = os.path.splitext(filename)
                os.makedirs("output", exist_ok=True)
                output_path = f"output/output_{name}.mp4"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs("output", exist_ok=True)
                output_path = f"output/output_webcam_{timestamp}.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            print(f"Saving output to: {output_path}")
        
        print("Press 'q' to quit")
        print("Database connection: PERSISTENT (will not close)")
        
        cv2.namedWindow('Vehicle Recognition - ENHANCED', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Vehicle Recognition - ENHANCED', display_width, int(display_width * 9/16))
        
        fps_queue = deque(maxlen=30)
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                h, w = frame.shape[:2]
                
                tracked_objects, plate_boxes = self.process_frame(frame)
                
                annotated_frame = self.draw_annotations(frame, tracked_objects, plate_boxes)
                
                current_fps = 1 / (time.time() - start_time)
                fps_queue.append(current_fps)
                avg_fps = sum(fps_queue) / len(fps_queue)
                
                fps_label = f"FPS: {avg_fps:.1f}"
                (fps_width, fps_height), baseline = cv2.getTextSize(
                    fps_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                
                cv2.rectangle(annotated_frame, 
                             (w - 160, 5), 
                             (w - 160 + fps_width + 10, 5 + fps_height + 10),
                             (255, 255, 255), -1)
                
                cv2.putText(annotated_frame, fps_label, (w - 155, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                if out is not None:
                    out.write(annotated_frame)
                
                cv2.imshow('Vehicle Recognition - ENHANCED', annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n✓ Interrupted by user")
        finally:
            cap.release()
            if out is not None:
                out.release()
                print(f"\n✓ Output video saved successfully!")
            
            print("✓ Database connection maintained (persistent)")
            
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Real-Time Vehicle Recognition System - ENHANCED')
    parser.add_argument('--plate-model', type=str, default='models/license_plate.pt', help='Path to plate detection model (.pt)')
    parser.add_argument('--vehicle-model', type=str, default='models/vehicle_model.pt', help='Path to vehicle classification model (.pt)')
    parser.add_argument('--camera', type=int, default=1, help='Camera ID (default: 1)')
    parser.add_argument('--video', type=str, help='Path to video file (overrides --camera)')
    parser.add_argument('--min-frames', type=int, default=3, help='Minimum frames before detection (default: 3)')
    parser.add_argument('--quality-threshold', type=float, default=0.60, 
                       help='Quality threshold for early detection (default: 0.60)')
    parser.add_argument('--periodic-check', type=int, default=30,
                       help='Periodic check interval in frames (default: 30)')
    parser.add_argument('--display-width', type=int, default=1280, help='Display window width (default: 1280)')
    parser.add_argument('--no-save', action='store_true', help='Do not save output video')
    
    parser.add_argument('--db-host', type=str, default='localhost', help='MySQL host (default: localhost)')
    parser.add_argument('--db-name', type=str, default='vehicle_recognition', 
                       help='Database name (default: vehicle_recognition)')
    parser.add_argument('--db-user', type=str, default='root', help='MySQL user (default: root)')
    parser.add_argument('--db-password', type=str, default='12345', help='MySQL password (default: 12345)')
    
    args = parser.parse_args()
    
    system = VehicleRecognitionSystem(
        args.plate_model,
        args.vehicle_model,
        min_frames=args.min_frames,
        quality_threshold=args.quality_threshold,
        periodic_check=args.periodic_check,
        db_host=args.db_host,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password
    )
    
    system.run_webcam(args.camera, args.video, args.display_width, save_output=not args.no_save)


if __name__ == "__main__":
    main()