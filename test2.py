import cv2
import numpy as np
import pickle
import pandas as pd
from ultralytics import YOLO
import cvzone
import psycopg2
from datetime import datetime


# Load parking area data and class list
with open("parking_data", "rb") as f:
    data = pickle.load(f)
    polylines = data["polylines"]

with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Load YOLO model for object detection
model = YOLO('yolov8n.pt')

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="parking_db",
    user="tshering",
    password="software@321",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

parking_name = input("Enter the parking name: ")
parking_space = input("Enter the parking space: ")

# Initialize video capture
cap = cv2.VideoCapture("rtsp://DVR:admin_123@192.168.0.98:554/Streaming/Channel/1")

# Initialize variables
total_cars = 0
counted_polylines = set()  # To track polylines that have turned red

# Track previous values to detect changes
prev_parked_cars = 0
prev_free_space = len(polylines)
prev_total_cars = 0


def car_enters():
    global total_cars
    total_cars += 1

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Resize frame for consistent processing
    frame = cv2.resize(frame, (1680, 850))

    # YOLO model prediction
    results = model.predict(frame)
    detections = pd.DataFrame(results[0].boxes.data).astype("float")

    # List to hold car centroids
    car_centroids = []

    # Process each detection
    for _, row in detections.iterrows():
        x1, y1, x2, y2, confidence, class_id = map(int, row)
        class_name = class_list[class_id]
        if class_name in ['car', 'bus', 'motorcycle']:
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            car_centroids.append(centroid)
    
    # Analyze parking spots
    parked_cars = 0
    for i, polyline in enumerate(polylines):
        is_car_in_spot = False  # To track if a car is inside the polyline
        for centroid in car_centroids:
            if cv2.pointPolygonTest(polyline, centroid, False) >= 0:
                parked_cars += 1
                is_car_in_spot = True
                break  # No need to check further centroids for this polyline

        if is_car_in_spot:
            # If a car is in the polyline, turn it red
            cv2.polylines(frame, [polyline], True, (0, 0, 255), 2)
            # Increment total_cars only if this polyline hasn't been counted before
            if i not in counted_polylines:
                counted_polylines.add(i)  # Mark this polyline as counted
                car_enters()
        else:
            # If no car is in the polyline, keep it green and remove it from counted_polylines
            cv2.polylines(frame, [polyline], True, (0, 255, 0), 2)
            if i in counted_polylines:
                counted_polylines.remove(i)  # Remove polyline from counted set

    # Calculate free spaces
    free_space = len(polylines) - parked_cars

    # Display car count and free space
    cvzone.putTextRect(frame, f'Cars in Parking: {parked_cars}', (700, 25), scale=2, thickness=2, colorR=(99, 11, 142))
    cvzone.putTextRect(frame, f'Free Spaces: {free_space}', (700, 65), scale=2, thickness=2, colorR=(99, 11, 142))
    cvzone.putTextRect(frame, f'Total vehicle parked: {total_cars}', (30, 30), scale=2, thickness=2, colorR=(99, 142, 11))

    # check if values have changed before updating the database
    if prev_parked_cars != parked_cars or prev_free_space != free_space or prev_total_cars != total_cars:
        # upadate the previous values
        prev_parked_cars = parked_cars
        prev_free_space = free_space
        prev_total_cars = total_cars

        # Insert data into the database
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        sql = """
        INSERT INTO parking_details (parking_id, parking_name, parking_space, timestamp, vehicle_in_parking, free_space, total_vehicle_parked)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        try:
            cursor.execute(sql, (1, parking_name, parking_space, timestamp, parked_cars, free_space, total_cars))
            conn.commit()
        except Exception as e:
            print("Database error:", e)
            conn.rollback()

    # Show the frame with annotations
    cv2.imshow('Parking Space Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
