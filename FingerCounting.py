import cv2
import HandTrackingModule as htm
import time
import os
import math
import random

blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
turquoise = (255, 255, 0)
purple = (255, 0, 255)
brown = (0, 255, 255)

colors = [blue, green, red, turquoise, purple, brown]
color = blue

folderPath = "FingerImages"
myList = os.listdir(folderPath)
overlayList = []
pTime = 0
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    image = cv2.resize(image, (200, 200))
    overlayList.append(image)

cap = cv2.VideoCapture(0)
detector = htm.HandDetector()
tipIds = [4, 8, 12, 16, 20]
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)
    if len(lmlist) != 0:

        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        length = math.hypot(x2 - x1, y2 - y1)

        if length < 25:
            color = random.choice(colors)
            cv2.circle(img, (cx, cy), 10, color, cv2.FILLED)

        fingers = []
        # Thumb
        if lmlist[tipIds[0]][1] > lmlist[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 fingers
        for id in range(1, 5):
            if lmlist[tipIds[id]][2] < lmlist[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        totalFingers = fingers.count(1)
        print(totalFingers)

        img[0:200, 0:200] = overlayList[totalFingers-1]

        cv2.rectangle(img, (20,300), (160, 420), color, cv2.FILLED)
        cv2.putText(img, str(totalFingers), (50, 400), cv2.FONT_HERSHEY_PLAIN,
                    8, (0, 0, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (480, 30), cv2.FONT_HERSHEY_PLAIN,
                2, (255,0,0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break