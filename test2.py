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
total_time_parked = {}  # Store the total time parked for each slot (in minutes)

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
                # If the car enters the spot and wasn't there before, record the arrival time
                if i not in counted_polylines:
                    arrival_times[i] = datetime.now()
                    print(f"Car arrived in slot {slot_ids[i]} at {arrival_times[i]}")
                counted_polylines.add(i)
                break  # No need to check further centroids for this polyline

        if not is_car_in_spot:
            if i in counted_polylines:  # Vehicle was there previously and now has left
                departure_times[i] = datetime.now()
                # Calculate total parked time as interval
                total_time_parked[i] = departure_times[i] - arrival_times[i]
                print(f"Car left slot {slot_ids[i]} at {departure_times[i]} after parking for {total_time_parked[i]}")
                counted_polylines.remove(i)

        # Draw polyline and manage color based on occupancy
        color = (0, 0, 255) if is_car_in_spot else (0, 255, 0)
        cv2.polylines(frame, [polyline], True, color, 2)

    # Calculate free spaces
    free_space = len(polylines) - parked_cars

    # Display car count and free space
    cvzone.putTextRect(frame, f'Cars in Parking: {parked_cars}', (700, 25), scale=2, thickness=2, colorR=(99, 11, 142))
    cvzone.putTextRect(frame, f'Free Spaces: {free_space}', (700, 65), scale=2, thickness=2, colorR=(99, 11, 142))

    # Insert into database with slot_id
    try:
        # Insert into parking detail
        cursor.execute("""
            INSERT INTO parking_detail (parkingarea_id, total_parking_slot, vehiclein_parking, free_parking_slot)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (parkingarea_id) DO UPDATE
            SET vehiclein_parking = EXCLUDED.vehiclein_parking, free_parking_slot = EXCLUDED.free_parking_slot;
        """, (parkingArea_id, len(polylines), parked_cars, free_space))

        # Insert into parking slots
        for j, polyline in enumerate(polylines):
            if j < len(slot_ids):  # Ensure that the slot_id index exists
                slot_id = slot_ids[j]
                is_occupied = j in counted_polylines
                arrival_time = arrival_times.get(j, None)
                departure_time = departure_times.get(j, None)
                total_parked_time = total_time_parked.get(j, timedelta(0))  # Use timedelta(0) if no time is available

                cursor.execute("""
                    INSERT INTO parking_slots (slot_id, parkingDetail_id, is_occupied, arrival_time, departureTime, total_time_parked)
                    VALUES (%s, (SELECT parkingDetail_id FROM parking_detail WHERE parkingarea_id = %s LIMIT 1), %s, %s, %s, %s)
                    ON CONFLICT (slot_id) DO UPDATE
                    SET is_occupied = EXCLUDED.is_occupied, 
                        arrival_time = COALESCE(EXCLUDED.arrival_time, parking_slots.arrival_time),
                        departureTime = COALESCE(EXCLUDED.departureTime, parking_slots.departureTime),
                        total_time_parked = COALESCE(EXCLUDED.total_time_parked, parking_slots.total_time_parked);
                """, (slot_id, parkingArea_id, is_occupied, arrival_time, departure_time, total_parked_time))
        
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
