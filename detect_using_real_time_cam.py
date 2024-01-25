import cv2
import numpy as np

# Object detection parameters
thres = 0.6
classNames = []
classFile = 'coco.names'
with open(classFile, 'r') as f:
    classNames = f.read().rstrip('\n').split('\n')

model = cv2.CascadeClassifier("face_detector.xml")
configpath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightspath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightspath, configpath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Lane detection parameters
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Define the region of interest (ROI) polygon coordinates
polygon = np.array([[50, 270], [20, 100], [360, 160], [480, 270]])

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from camera.")
        break

    # Object detection
    classIds, confs, bbox = net.detect(frame, confThreshold=thres)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
            cv2.putText(frame, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Lane detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    stencil = np.zeros_like(gray)
    cv2.fillConvexPoly(stencil, polygon, 1)
    masked_frame = cv2.bitwise_and(gray, gray, mask=stencil)
    ret, thresh = cv2.threshold(masked_frame, 130, 145, cv2.THRESH_BINARY)
    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 30, maxLineGap=200)

    try:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    except TypeError:
        pass

    # Display the processed frame
    cv2.imshow('Object and Lane Detection', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
