'''
step 1: find hsv limits for required colours
step 2: create a mask with given colours
step 3: find contours from the mask
step 4: get bounding rectangle for the contour
step 5: find top point of bounding rectange as point to draw circle 
step 6: draw the circle at mentioned point
step 7: store every new point in a list, so every point can be redrawn for every new frame

limitations:
- checking only for colours, no understanding of what the object is
- can be subject to lighting changes and occlusions
- for each new colour to be added, hsv limits have to be manually found with trial and error
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


#setting resolution
frameWidth = 640
frameHeight = 480

#cap is an object that that can read frames from the camera. 0 -> default camera of device
cap = cv2.VideoCapture(0)

#for setting resolution of the object. first parameter is property number. 
cap.set(3, frameWidth)
cap.set(4, frameHeight)

#colors: blue and orange
myColors = [[7,191,83,11,255,255], [93,143,29,106,255,255]]
myColorValues = [[0,128,255], [255,153,51]]

myPoints = []

def findColor(img, myColors, myColorValues):
    #converting each frame of the video to hsv
    #it is easier to detect colour in hsv that bgr
    #the actual colour itself mainly depends on hue, so it is easier to detect in different lighting
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masks = []
    masks.append(imgHSV)
    count = 0
    newPoints = []
    for color in myColors:
        #lower and upper limits of hsv
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        #takes every pixel from hsv image, converts to white if within range, else black
        mask = cv2.inRange(imgHSV, lower, upper)
        #gets relevant points from the image blob
        x,y = getContours(mask)
        #draws circle at points
        masks.append(mask)
        if(x!=0 and y!=0):
            newPoints.append([x,y,count])
        count+=1
    return masks, newPoints

def getContours(img):
    #contours stores each relevant object as a white blob, and bg is black 
    contours, heirarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    x,y,w,h = 0,0,0,0
    for cnt in contours: #cnt: list of points
        area = cv2.contourArea(cnt)
        if area>100: #to remove noise
            #calculates perimeter of object. this is then used to calculate number of important points
            peri = cv2.arcLength(cnt, True) 
            #finds number of important points in the contour
            #perimeter is used for tolerance of error. better to use a number related to the size of the object
            #last parameter indicates a closed object
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            x,y,w,h = cv2.boundingRect(approx)
    return int(x+(w/2)), int(y) #top centre corner

def draw(myPoints, myColorValues):
    for point in myPoints:
        cv2.circle(imgResult, (point[0],point[1]), 10, myColorValues[point[2]], -1) 
        #all old points are redrawn again since new frame is fresh

while True:
    success, img = cap.read()
    #img is each frame from the video stream
    img = cv2.flip(img, 1) #second param:  0 -> flip vertically, 1 -> flip horizontally(mirror), -1 -> both
    imgResult = img.copy()
    if not success:
        print("Failed to capture frame")
        break
    #returns masks and point where to draw for each colour
    masks, newPoints = findColor(img, myColors, myColorValues)
    if len(newPoints)!=0:
        for newP in newPoints:
            myPoints.append(newP)
    if len(myPoints)!=0:
        draw(myPoints, myColorValues)
            
    imgStack = stackImages(
        0.6,
        ([img, img, masks[0]],
         [masks[1], masks[2], imgResult])
    )

    cv2.imshow("Stacked Image", imgStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
