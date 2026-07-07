'''
step 1: convert image to gray for easier detection
step 2: apply gaussian blur to reduce image noise
step 3: detect edges using canny 
step 4: use image dilation to fill in gaps in edges
step 5: use image erosion to restore edges to original thickness
step 6: find largest edge, check if it is a rectangle
step 7: find corners of the object, warp into required points

limitations:
- does not understand what the object is
- assumes largest rectangular contour is object
- may not work if there is a low contrast between object and background
- may not work if edges are not clearly visible
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


def reorderPoints(pts):
    pts = pts.reshape((4,2))
    reordered = np.zeros((4,2), dtype=np.float32)
    minSum = float('inf')
    maxSum = float('-inf')
    minDiff = float('inf')
    maxDiff = float('-inf')
    for point in pts:
        x = point[0]
        y = point[1]
        pointSum = x + y
        pointDiff = y - x
        if pointSum < minSum:
            minSum = pointSum
            reordered[0] = point      # Top Left
        if pointSum > maxSum:
            maxSum = pointSum
            reordered[2] = point      # Bottom Right
        if pointDiff < minDiff:
            minDiff = pointDiff
            reordered[1] = point      # Top Right
        if pointDiff > maxDiff:
            maxDiff = pointDiff
            reordered[3] = point      # Bottom Left
    return reordered


cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture frame.")
        break

    imgResult = np.zeros((480, 640, 3), np.uint8)
    imgContour = img.copy()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #clip limit: how stretched out to make the values. higher -> more noise
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgClahe = clahe.apply(imgGray)
    imgBlur = cv2.GaussianBlur(imgClahe, (7,7), 0)
    imgCanny = cv2.Canny(imgBlur, 30, 150)
    kernel = np.ones((7, 7), np.uint8)
    imgDil = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThresh = cv2.erode(imgDil, kernel, iterations = 1)

    

    contours, hierarchy = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        maxArea = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area>maxArea:
                maxCnt = cnt
                maxArea = area
        peri = cv2.arcLength(maxCnt, True)
        approx = cv2.approxPolyDP(maxCnt, 0.03 * peri, True)

        
        cv2.drawContours(imgContour, [maxCnt], -1, (0,255,0), 5)

        if len(approx) == 4:
            pts1 = reorderPoints(np.float32(approx))
            pts2 = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgResult = cv2.warpPerspective(img, matrix, (640, 480))

    imgStack = stackImages(0.5,
                       [[img, imgGray, imgCanny],
                        [imgDil, imgContour, imgResult]])
    cv2.imshow("Result", imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

