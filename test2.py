import cv2
import numpy as np
import pickle
import pandas as pd
from ultralytics import YOLO
import cvzone
import psycopg2
from datetime import datetime, timedelta

# Load parking area data and class list
with open("parking_data", "rb") as f:
    data = pickle.load(f)
    polylines = data.get("polylines", [])
    slot_ids = data.get("slot", [])  # Load slot_id values

# Load class list for YOLO detection
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

# Initialize video capture
cap = cv2.VideoCapture("rtsp://DVR:admin_123@192.168.0.98:554/Streaming/Channel/1")

# Variables to track parking data
counted_polylines = set()  # Track counted polylines
arrival_times = {}  # Store the arrival time of vehicles
departure_times = {}  # Store the departure time of vehicles

# Fetch the parkingArea_id for a specific parking area
parking_location = "norzin lam"
cursor.execute("SELECT parkingArea_id FROM parking_area WHERE parking_location = %s", (parking_location,))
parkingArea_id_result = cursor.fetchone()

if parkingArea_id_result:
    parkingArea_id = parkingArea_id_result[0]  # Extract the parkingArea_id from the result
else:
    print("Parking area not found in the database.")
    parkingArea_id = None

# Main loop
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
        is_car_in_spot = False
        for centroid in car_centroids:
            if cv2.pointPolygonTest(polyline, centroid, False) >= 0:
                parked_cars += 1
                is_car_in_spot = True
                # If the car enters the spot and wasn't there before, record the arrival time
                if i not in counted_polylines:
                    arrival_times[i] = datetime.now()
                    print(f"Car arrived in slot {slot_ids[i]} at {arrival_times[i]}")
                counted_polylines.add(i)
                break  # No need to check further centroids for this polyline

        # Draw polyline and manage color based on occupancy
        color = (0, 0, 255) if is_car_in_spot else (0, 255, 0)
        cv2.polylines(frame, [polyline], True, color, 2)

    # Calculate free spaces
    free_space = len(polylines) - parked_cars

    # Display car count and free space
    cvzone.putTextRect(frame, f'Cars in Parking: {parked_cars}', (1362, 84), scale=2, thickness=2, colorR=(99, 11, 142))
    cvzone.putTextRect(frame, f'Free Spaces: {free_space}', (1362, 117), scale=2, thickness=2, colorR=(99, 11, 142))

    # Insert into the database
    try:
        # Insert into parking detail
        cursor.execute("""
            INSERT INTO parking_detail (parkingArea_id, total_parking_slot, vehicleIn_parking, free_parking_slot, currentTime)
            VALUES (%s, %s, %s, %s, %s)
        """, (parkingArea_id, len(polylines), parked_cars, free_space, datetime.now()))
        conn.commit()

        # Insert into parking slots
        for j, polyline in enumerate(polylines):
            if j < len(slot_ids):  # Ensure that the slot_id index exists
                slot_id = slot_ids[j]
                parkingDetail_id = 2
                is_occupied = j in counted_polylines
                arrival_time = arrival_times.get(j, None)
                departure_time = departure_times.get(j, None)

                cursor.execute("""
                    INSERT INTO parking_slots (slot_id, parkingdetail_id, is_occupied, arrival_time, departure_time)
                    VALUES (%s, %s, %s, %s, %s)
                """, (slot_id, parkingDetail_id, is_occupied, arrival_time, departure_time))
                conn.commit()
    except Exception as e:
        print(f"Database error: {e}")
        conn.rollback()

    # Show the frame with annotations
    cv2.imshow('Parking Space Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
