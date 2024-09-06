import cv2
import numpy as np
import cvzone
import pickle


cap = cv2.VideoCapture("rtsp://DVR:admin_123@192.168.0.98:554/Streaming/Channel/1")

drawing = False
area = []

try:
    with open("parking_data", "rb") as f:
        data = pickle.load(f)
        polylines = data["polylines"]
        area = data["area"]
except:
    polylines = []

points = []
parking_number = " "


def draw(event, x, y, flags, param):
    global points, drawing
    drawing = True
    if event == cv2.EVENT_LBUTTONDOWN:
        points = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        parking_number = input("Enter the parking number: ")
        if parking_number:
            area.append(parking_number)
            polylines.append(np.array(points, np.int32))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if polylines:
            polylines.pop()
            area.pop()

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    frame=cv2.resize(frame,(1680,850))
    for i, polyline in enumerate(polylines):
        cv2.polylines(frame, [polyline], True, (0, 0, 255), 2)
        #cvzone.putTextRect(frame, f'{area[i]}', tuple(polyline[0]), 1, 1)    
    cv2.imshow('FRAME', frame)

    cv2.setMouseCallback('FRAME', draw)

    Key = cv2.waitKey(30) & 0xFF
    if Key == ord('s'):
        with open("parking_data", "wb") as f:
            data = {"polylines": polylines, "area": area}
            pickle.dump(data, f)
    if Key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
