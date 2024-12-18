import cv2
import face_recognition
import numpy as np
import sqlite3
from datetime import datetime
import time
import os
import pickle  # Ensure pickle is imported
from ultralytics import YOLO

# Configuration Variables
USE_IP_CAMERA = True
CAMERA_URL = "http://192.168.1.11:8080/video"
MODEL_PATH = "yolov8-face.pt"
DATABASE_PATH = "attendance.db"
CAPTURE_INTERVAL = 5  # seconds
ZOOM_LEVELS = [1.0, 1.3, 1.6, 2.5]

# Initialize Database
def initialize_database():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        subject TEXT,
                        room TEXT,
                        date TEXT,
                        time TEXT)''')
    conn.commit()
    conn.close()

# Check if attendance already exists
def attendance_exists(name, subject, room):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    today_date = datetime.now().strftime("%Y-%m-%d")
    cursor.execute('''SELECT 1 FROM attendance WHERE name = ? AND subject = ? AND room = ? AND date = ?''',
                   (name, subject, room, today_date))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

# Save Attendance
def save_attendance(name, subject, room):
    if attendance_exists(name, subject, room):
        print(f"Attendance already marked for {name}, Subject: {subject}, Room: {room}")
        return

    print(f"Saving attendance for: {name}, Subject: {subject}, Room: {room}")
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    now = datetime.now()
    cursor.execute('''INSERT INTO attendance (name, subject, room, date, time) 
                      VALUES (?, ?, ?, ?, ?)''',
                   (name, subject, room, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")))
    conn.commit()
    print("Attendance saved successfully!")
    conn.close()

# Initialize YOLO Model
def load_model():
    print("Loading YOLOv8 model...")
    model = YOLO(MODEL_PATH)
    return model

# Camera Selection
def get_video_capture():
    if USE_IP_CAMERA:
        capture = cv2.VideoCapture(CAMERA_URL)
    else:
        capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        print("Failed to open video source.")
        exit(1)

    return capture

# Zoom Logic
def apply_zoom(frame, zoom_factor):
    h, w = frame.shape[:2]
    new_w = int(w / zoom_factor)
    new_h = int(h / zoom_factor)
    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    zoomed_frame = frame[y1:y1 + new_h, x1:x1 + new_w]
    return cv2.resize(zoomed_frame, (w, h), interpolation=cv2.INTER_LINEAR)

# Main Function
def take_attendance(subject, room):
    initialize_database()
    model = load_model()
    video_capture = get_video_capture()

    known_encodings = []
    known_names = []
    if os.path.exists('face_encodings.pkl'):
        with open('face_encodings.pkl', 'rb') as f:
            data = pickle.load(f)
            known_encodings = data['encodings']
            known_names = data['names']
            print(f"Loaded encodings: {len(known_encodings)}, names: {known_names}")

    last_capture_time = time.time() - CAPTURE_INTERVAL
    start_time = time.time()

    cv2.namedWindow('Attendance System', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Attendance System', 1280, 720)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to retrieve frame.")
            break

        elapsed = time.time() - start_time
        zoom_factor = ZOOM_LEVELS[int(elapsed // 3 % len(ZOOM_LEVELS))]
        frame = apply_zoom(frame, zoom_factor)

        if time.time() - last_capture_time >= CAPTURE_INTERVAL:
            last_capture_time = time.time()

            # Convert frame to RGB and detect faces
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                name = "Unknown"

                if True in matches:
                    matched_idx = matches.index(True)
                    name = known_names[matched_idx]
                    print(f"Face recognized: {name}")
                    save_attendance(name, subject, room)

                # Draw a rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

        cv2.imshow('Attendance System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    take_attendance('Mathematics', 'Room 101')
