'''
changes:
- instead of comparing current frame with previous, current is compared with a continuously updating background frame
- background frame constantly updates, so there is some learning
- compares motion with long-term learnt background
- background gradually adapts with changes in the scene

limitations:
- assumes mostly stationary camera
- objects that are stationary for a long time may not be detected since they will be considered part of the bg
- cannot track objects
- no understanding of what the object is
'''

import cv2
import numpy as np


def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

cap = cv2.VideoCapture("resources/video.mp4")
cap.set(3,480)
cap.set(4,640)

success, frame1 = cap.read()
success, frame2 = cap.read()
frame1 = cv2.resize(frame1, (480,640))
frame2 = cv2.resize(frame2, (480,640))
background = frame1.astype(np.float32)

while True:

    background = 0.95 * background + 0.05 * frame2
    background_int = background.astype(np.uint8)

    diff = cv2.absdiff(frame2, background_int) #if pixel value same, diff = 0 (black)
    imgGray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 0)  #high blur better since lesser noise, only motion of a blob detected
    _, thresh = cv2.threshold(imgBlur, 50, 255, cv2.THRESH_BINARY) #if pixel above threshold, assign it as white, else 0 (black)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < 20: #or area >400:
            continue
        cv2.rectangle(frame2, (x,y),(x+w,y+h), (0,255,0), 2)

    #cv2.drawContours(frame1, contours, -1, (0,255,0), 2)

    cv2.imshow("Result", frame2)

    frame1 = frame2
    success, frame2 = cap.read()
    frame2 = cv2.resize(frame2, (480,640))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
