# coding: utf-8
#Frederick Müller 
#Campus Velbert/Heiligenhaus, Hochschule Bochum, 2018/2019

from threading import Thread
from cvSettings import CVSettings
cvSettings = CVSettings(True)
import cv2

class CameraVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.grabbed = self.stream.set(cv2.CAP_PROP_FRAME_WIDTH,cvSettings.cXResolution)
        self.grabbed = self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT,cvSettings.cYResolution)
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        print "Webcam Thread started"
        return self
 
    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()
 
    def read(self):
        return self.frame
 
    def stop(self):
        self.stopped = True

    def release(self):
        self.stream.release()
        self.stopped = True