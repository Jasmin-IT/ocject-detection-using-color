import cv2
import imutils

redLower = (27,47, 150) #redLOwer is the lower bound of the HSV color space
redUpper = (179,255, 255) #redUpper is the upper bound of the HSV color space

camera = cv2.VideoCapture(1)
if not camera.isOpened():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise SystemExit("Unable to open the camera. Check your camera index and connection.")

while True:
    (grabbed, frame) = camera.read()  # grabbed is a boolean that indicates whether the frame was successfully grabbed, and frame is the actual frame that was captured
    if not grabbed or frame is None:
        continue

    frame = imutils.resize(frame, width=600)  # resize the frame 
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)  # blur the frame to reduce noise
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  # convert to the HSV color space
    
    mask = cv2.inRange(hsv, redLower, redUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)  # find the minimum enclosing circle of the contour
        M = cv2.moments(c)
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))  # calculate the center of the contour
        else:
            center = None
        if radius > 10 and center is not None:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2) #draw a circle around the contour
            cv2.circle(frame, center, 5, (0, 0, 255), -1) #draw a dot at the center of the contour
            if radius > 250:
                print("stop")
            else:
                if (center[0] < 150):
                    print("left")
                elif (center[0] > 450):
                    print("right")
                elif (radius < 250):
                    print("front")
                else:
                    print("stop")
    cv2.imshow("Frame", frame) #display the frame
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
camera.release()
cv2.destroyAllWindows()