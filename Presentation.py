import cv2
import os
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time

# Parameters
screen_width, screen_height = 1580, 850  # Adjust this to your screen resolution
folderPath = "presentation"

# Camera Setup
cap = cv2.VideoCapture(0)
cam_width, cam_height = 640, 480  # Standard webcam resolution
cap.set(3, cam_width)
cap.set(4, cam_height)

# Hand Detector
detectorHand = HandDetector(detectionCon=0.8, maxHands=1)

# Variables
imgList = []
delay = 10
buttonPressed = False
counter = 0
drawMode = False
imgNumber = 0
delayCounter = 0
annotations = [[]]
annotationNumber = -1
annotationStart = False
camWidth, camHeight = 160, 120  # Smaller size for camera feed
camOffset = 100  # Offset from the right edge

# Gesture hold parameters
gestureHoldTime = 3  # Time in seconds to hold the gesture
gestureStartTime = None

# Get list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
print(pathImages)

# EMA parameters
alpha = 0.2  # Smoothing factor
prevX, prevY = 0, 0

def smooth_point(new_point, prev_point, alpha=0.2):
    new_x = alpha * new_point[0] + (1 - alpha) * prev_point[0]
    new_y = alpha * new_point[1] + (1 - alpha) * prev_point[1]
    return new_x, new_y

while True:
    # Get image frame
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    # Resize imgCurrent to match full screen dimensions
    imgCurrent = cv2.resize(imgCurrent, (screen_width, screen_height))

    # Find the hand and its landmarks
    hands, img = detectorHand.findHands(img)  # with draw

    if hands and buttonPressed is False:  # If hand is detected

        hand = hands[0]
        cx, cy = hand["center"]
        lmList = hand["lmList"]  # List of 21 Landmark points
        fingers = detectorHand.fingersUp(hand)  # List of which fingers are up

        # Constrain values for easier drawing
        xVal = int(np.interp(lmList[8][0], [0, cam_width], [0, screen_width]))
        yVal = int(np.interp(lmList[8][1], [0, cam_height], [0, screen_height]))
        indexFinger = (xVal, yVal)

        # Smooth the index finger position
        smoothedX, smoothedY = smooth_point(indexFinger, (prevX, prevY), alpha)
        indexFinger = (int(smoothedX), int(smoothedY))
        prevX, prevY = indexFinger

        if fingers == [1, 0, 0, 0, 0]:
            print("Right")
            buttonPressed = True
            if imgNumber > 0:
                imgNumber -= 1
                annotations = [[]]
                annotationNumber = -1
                annotationStart = False
        if fingers == [0, 0, 0, 0, 1]:
            print("Left")
            buttonPressed = True
            if imgNumber < len(pathImages) - 1:
                imgNumber += 1
                annotations = [[]]
                annotationNumber = -1
                annotationStart = False

        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

        if fingers == [0, 1, 0, 0, 0]:
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            print(annotationNumber)
            annotations[annotationNumber].append(indexFinger)
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

        else:
            annotationStart = False

        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                annotations.pop(-1)
                annotationNumber -= 1
                buttonPressed = True
        
        if fingers == [1, 1, 1, 1, 1]:
            if gestureStartTime is None:
                gestureStartTime = time.time()
            elif time.time() - gestureStartTime > gestureHoldTime:
                print("Four fingers detected for 3 seconds, ending the program.")
                break
        else:
            gestureStartTime = None

    else:
        annotationStart = False

    if buttonPressed:
        counter += 1
        if counter > delay:
            counter = 0
            buttonPressed = False

    for i, annotation in enumerate(annotations):
        for j in range(len(annotation)):
            if j != 0:
                cv2.line(imgCurrent, annotation[j - 1], annotation[j], (0, 0, 200), 12)

    # Resize the camera feed and place it slightly left from the corner of imgCurrent
    imgSmall = cv2.resize(img, (camWidth, camHeight))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:camHeight, w - camWidth - camOffset: w - camOffset] = imgSmall

    # Display the current presentation slide
    cv2.imshow("Slides", imgCurrent)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
