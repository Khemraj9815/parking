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
    polylines = data.get("polylines", [])
    slot = data.get("slot", [])  # Use .get to ensure slot is initialized

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

# Initialize variables
counted_polylines = set()  # Track counted polylines

# Fetch the parkingArea_id for a specific parking area
parking_location = "norzin lam"  # You can adjust this as needed
cursor.execute("SELECT parkingArea_id FROM parking_area WHERE parking_location = %s", (parking_location,))
parkingArea_id_result = cursor.fetchone()

if parkingArea_id_result:
    parkingArea_id = parkingArea_id_result[0]  # Extract the parkingArea_id from the result
else:
    print("Parking area not found in the database.")
    parkingArea_id = None



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
                break  # No need to check further centroids for this polyline

        if is_car_in_spot:
            # If a car is in the polyline, turn it red
            cv2.polylines(frame, [polyline], True, (0, 0, 255), 2)
            # Add to counted_polylines if not counted already
            counted_polylines.add(i)
        else:
            # If no car is in the polyline, keep it green
            cv2.polylines(frame, [polyline], True, (0, 255, 0), 2)
            if i in counted_polylines:
                counted_polylines.remove(i)

    # Calculate free spaces
    free_space = len(polylines) - parked_cars

    # Display car count and free space
    cvzone.putTextRect(frame, f'Cars in Parking: {parked_cars}', (700, 25), scale=2, thickness=2, colorR=(99, 11, 142))
    cvzone.putTextRect(frame, f'Free Spaces: {free_space}', (700, 65), scale=2, thickness=2, colorR=(99, 11, 142))

    # Check if values have changed before updating the database
    

    # Insert database
    try:
        cursor.execute("""
            INSERT INTO parking_detail (parkingarea_id, total_parking_slot, vehiclein_parking, free_parking_slot)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (parkingDetail_id) DO NOTHING;
        """, (parkingArea_id, len(polylines), parked_cars, free_space))
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
