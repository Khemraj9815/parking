import cv2
import numpy as np
import cvzone
import pickle

# RTSP stream setup (adjust as needed)
cap = cv2.VideoCapture("rtsp://DVR:admin_123@192.168.0.98:554/Streaming/Channel/1")

drawing = False
polylines = []
slot = []

# Attempt to load existing parking data
try:
    with open("parking_data", "rb") as f:
        data = pickle.load(f)
        polylines = data.get("polylines", [])  # Use .get() to provide default empty list
        slot = data.get("slot", [])  # Ensure slot is initialize
except FileNotFoundError:
    print("No existing parking data found, starting fresh.")
except Exception as e:
    print(f"An error occurred while loading the parking data: {e}")

points = []

def draw(event, x, y, flags, param):
    global points, drawing, slot_id
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:  # Only allow 4 points to be stored
            points.append((x, y))
        if len(points) == 4:  # Once 4 points are captured, create the polyline
            drawing = False
            slot_id = input("Enter the slot_id: ")
            if slot_id:
                slot.append(slot_id)
                # Create a closed polyline with exactly 4 points
                polyline = np.array(points, np.int32).reshape((-1, 1, 2))
                polylines.append(polyline)
                points = []  # Clear points for the next polyline
    elif event == cv2.EVENT_RBUTTONDOWN:
        if polylines:
            polylines.pop()
            if slot:
                slot.pop()

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame = cv2.resize(frame, (1680, 850))
    
    # Draw the polylines and slot IDs on the frame
    for i, polyline in enumerate(polylines):
        cv2.polylines(frame, [polyline], True, (0, 0, 255), 2)
        if i < len(slot):  # Ensure index is in range
            cvzone.putTextRect(frame, f'{slot[i]}', tuple(polyline[0][0]), 1, 1)  # Adjust for cvzone
    
    cv2.imshow('FRAME', frame)

    # Set mouse callback for drawing
    cv2.setMouseCallback('FRAME', draw)

    # Keyboard input handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # Save the current parking data to file
        with open("parking_data", "wb") as f:
            data = {
                "polylines": polylines,
                "slot": slot
            }
            pickle.dump(data, f)
        print("Parking data saved.")
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
