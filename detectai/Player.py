import cv2
import numpy as np
def center(points):
    """calculates centroid of a given matrix"""
    x = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4
    y = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4
    return np.array([np.float32(x), np.float32(y)], np.float32)

class Player():
    def buildKanmanFilter(self):
        filter = cv2.KalmanFilter(4,2)
        filter.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        filter.transitionMatrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32)
        filter.processNoiseCov =filter.transitionMatrix *0.03
        return filter

    def __init__(self, name, frame, windows):
        self.name = name;
        x, y, w, h = windows;
        self.roi = cv2.cvtColor(frame[y:y + h, x:x+w], cv2.COLOR_BGR2HSV)
        roi_hist = cv2.calcHist([self.roi], [0], None, [16],[0, 180])
        self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # set up the kalman for forecast
        self.kalman = self.buildKanmanFilter()
        self.measurement = np.array((2,1), np.float32)
        self.prediction = np.zeros((2,1), np.float32)
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.center = None
        self.track_window = windows
        self.update(frame)

    def __del__(self):
        print("Player %s destroyed" % self.name)


    def update(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back_project = cv2.calcBackProject([hsv],[0],self.roi_hist,  [0, 180],1)
        ret, self.track_window = cv2.CamShift(back_project, self.track_window, self.term_crit)
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        self.center = center(pts)
        cv2.polylines(frame, [pts], True, 255,1)
        self.kalman.correct(self.center)

        prediction = self.kalman.predict()
        cv2.circle(frame, (int(prediction[0]), int(prediction[1])), 4, (0,255,0), -1)

        cv2.putText(frame, self.name, (11,300), cv2.QT_FONT_NORMAL, 0.6, (0,0,0),1, cv2.LINE_AA )

        cv2.putText(frame, self.name, (12,322), cv2.QT_FONT_NORMAL, 0.6, (0, 255, 0), 1, cv2.LINE_AA)






