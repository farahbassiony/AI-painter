import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

folderPath= "header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
header = overlayList[0]
drawColor = (0, 0, 255)
brushThickness = 15
eraserThickness = 50
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = htm.handDetector(detectionCon=0.75)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
while True:
    fingers = [0, 0, 0, 0, 0]
    success, img = cap.read()
    img = cv2.flip(img, 1)  
    detector.findHands(img)
    lmList = detector.findPositions(img, draw=False)
    if len(lmList)!=0:
        # Get the index finger tip position
        x1, y1 = lmList[8][1], lmList[8][2]
        # Get the middle finger tip position
        x2, y2 = lmList[12][1], lmList[12][2]
        fingers=detector.fingersUp()
        # print(fingers)
    if fingers[1] and fingers[2]:
        xp, yp = 0, 0
        print("Selection Mode")
        cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
        if y1 < 125:  
            print(x1)
            time.sleep(0.1)
            if 150 < x1 < 250:
                header = overlayList[2]
                drawColor = (0, 255, 0)
                print("Color: green")

            elif 250 < x1 < 350:
                print("Color: same as before")
            elif 350 < x1 < 550:
                header = overlayList[0]
                drawColor = (255, 0, 0)
                print("Color: blue")
            
            elif 550 < x1 < 670:
                print("Color: same as before")
            elif 670 < x1 < 800:
                header = overlayList[3]
                drawColor = (0, 0, 255)
                print("Color: red")
            elif 800 < x1 < 950:
                print("Color: same as before")
            elif 950 < x1 < 1150:
                header = overlayList[1]
                drawColor = (0, 0, 0)
                print("Color: eraser")                                    
            
    if fingers[1] and fingers[2] == False:
        print("Drawing Mode")
        if xp == 0 and yp == 0:  
            xp, yp = x1, y1

        cv2.circle(img, (xp, yp), 15, drawColor, cv2.FILLED)

        if drawColor == (0, 0, 0):  
            cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
        else:
            cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
        xp, yp = x1, y1

      

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img= cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)
 
    img[0:125, 0:1280] = header
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == 27:  
        break
