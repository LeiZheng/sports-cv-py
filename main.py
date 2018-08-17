import cv2
from detectai.Player import Player

history = 20
bsKnn = cv2.createBackgroundSubtractorKNN(detectShadows=True)
bsKnn.setHistory(history)

cv2.namedWindow("tracing players")
pedestrians = []
firstFrame = True
frames = 0


cap = cv2.VideoCapture("C:/vsc/data/hockey-tracking/test2.avi")

while cap.isOpened():
    ret, frame = cap.read()
    if ret is False:
        print("exit the video play")
        break

    fgmask = bsKnn.apply(frame)

    if frames < history:
        frames += 1
        continue

    th = cv2.threshold(fgmask.copy(), 127, 255, cv2.THRESH_BINARY)[1]
    th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
    image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    counter = 0
    for c in contours:
        if cv2.contourArea(c) > 500:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x,y), (x +w, y + h), (0,255, 0), 1)
            if firstFrame is True:
                pedestrians.append(Player("player " + str(counter), frame, (x, y, w,h)))
                # pedestrians[counter] =
                counter += 1


    for p in pedestrians:
        p.update(frame)

    firstFrame = False

    frames += 1
    cv2.imshow("tracing players", frame)
    cv2.waitKey(10)



