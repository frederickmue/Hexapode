# coding: utf-8
#Frederick Müller 
#Campus Velbert/Heiligenhaus, Hochschule Bochum, 2018/2019

import cv2
import numpy as np
import traceback
from cvSettings import CVSettings
cvSettings = CVSettings()
import time
from cameraThread import CameraVideoStream

try:
    from tastfunktion import *
    from IKfunction import *
    from IK import *
    IKImport = True
except:
    traceback.print_exc()
    IKImport = False
    print "Inverse Kinematik und Zusatzfunktionen konnten nicht geladen werden."
    print "Fortfahren mit eingeschränktem Funktionsumfang."


class IKcvfunction:
    def __init__(self):
        if CVSettings.useStereoVision:
            self.capR = CameraVideoStream(0)
            self.capL = CameraVideoStream(1)
            self.capR.start()
            self.capL.start()
        else:
            self.cap = CameraVideoStream(0)
            self.cap.start()
        try:
            self.qr = cv2.QRCodeDetector()
        except:
            print "QRDecoder konnte nicht initialisiert werden. Wird OpenCV >4.0.0 genutzt?"
        self.foundObjects = []

    def findObjectFromColor(self, image, lowerIntervals = 0, upperIntervals = 0, preset = ''):
        #Sucht nach einem Objekt anhand bekannter Farbe/Geometrie

        if (preset != '' or lowerIntervals == 0 or upperIntervals == 0):
            if (preset == 'yellow'):
                print 'preset yellow'
                lowerIntervals=cvSettings.lowerYellow
                upperIntervals=cvSettings.upperYellow
            elif (preset == 'blue'):
                print 'preset blue'
                lowerIntervals=cvSettings.lowerBlue
                upperIntervals=cvSettings.upperBlue
            elif (preset == 'red'):
                print 'preset red'
                lowerIntervals=cvSettings.lowerRed
                upperIntervals=cvSettings.upperRed
            else:
                raise ValueError('Unknown preset or bad Intervals')
        blurred = cv2.medianBlur(image,3) # Für Distanzberechnung ggf. wenig hilfreich
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (255,255,255),(255,255,255))#Generieren einer 'Nullmaske'
        for i in xrange(0,len(lowerIntervals)):
            try:
                tempmask = cv2.inRange(hsv, lowerIntervals[i], upperIntervals[i])
                mask = cv2.bitwise_or(tempmask, mask)
            except IndexError:
                print "bad Intervals"
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2] #TODO: Argumente so i.o.?
        center = None

        if len(cnts) > 0:
            cnts.sort(key=cv2.contourArea, reverse=True)
            for c in cnts:
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                if radius > 20: #TODO: experimenteller Wert
                    distance = self.calculateDistance(radius, cvSettings.cRealObjectRadius) #TODO: Sucht explizit nach bekanntem Ball. Mehr Objekte?
                    valiMask = np.zeros((cvSettings.cYResolution,cvSettings.cXResolution), np.uint8)
                    cv2.circle(valiMask, (int(x), int(y)), int(radius), (255, 255, 255), thickness=-1)
                    valiImg = cv2.bitwise_and(image,image,mask=valiMask)
                    mean = cv2.mean(valiImg, valiMask) # TODO: Arbeiten mit stddev um Objekte zu filtern?
                    mean = np.asarray(mean[0:3])
                    mean = np.around(mean).astype("uint8")
                    mean = np.reshape(mean,(1,1,3))
                    mean = cv2.cvtColor(np.asarray(mean[0:3]), cv2.COLOR_BGR2HSV)
                    cv2.circle(image, (int(x), int(y)), int(radius), (255, 255, 255), thickness=1)
                    cv2.imwrite("test2.jpg", image) #TODO:DEBUG
                    if CVSettings.useDebugGUI:
                        cv2.imshow("Frame", image)
                        cv2.imshow("Mask", mask)
                        cv2.imshow("valImg",valiImg)
                        cv2.imshow("valiMask", valiMask)
                        print mean
                        key = cv2.waitKey(0) & 0xFF
                    
                    for c in xrange(0, len(lowerIntervals)):
                        comp1 = np.greater_equal(mean,lowerIntervals[c]) # mean > lowerIntervals
                        comp2 = np.less_equal(mean,upperIntervals[c]) # mean < upperIntervals
                        if (comp1.all() and comp2.all()): # vergleich einzelner Komponenten
                            return np.round((x,y,radius,distance),decimals=1)
                        else:
                            continue
                else:
                    continue 
            print "No object found."
            return (0,0,0,0)
        
    def cropImgToObject(self, image, foundObject):
        #TODO: Bild wird auf gefundenes Obejekt zugeschnitten.
        minX = int(np.round(foundObject[0]-foundObject[2]-10))
        maxX = int(np.round(foundObject[0]+foundObject[2]+10))
        minY = int(np.round(foundObject[1]-foundObject[2]-10))
        maxY = int(np.round(foundObject[1]+foundObject[2]+10))
        ROI = imcrop(image, (minX,minY,maxX,maxY))
        if CVSettings.useDebugGUI:
            cv2.imshow("ROI", ROI)
            key = cv2.waitKey(1) & 0xFF
        return ROI

    def imcrop(self, image, bbox): #TODO: Umschreiben
        bbox = np.round(bbox).astype('int')
        x1, y1, x2, y2 = bbox
        if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
            image, x1, x2, y1, y2 = self.pad_img_to_fit_bbox(image, x1, x2, y1, y2)
        return image[y1:y2, x1:x2, :]

    def pad_img_to_fit_bbox(self, image, x1, x2, y1, y2): #TODO: Umschreiben
        image = cv2.copyMakeBorder(image, - min(0, y1), max(y2 - image.shape[0], 0), -min(0, x1), max(x2 - image.shape[1], 0),cv2.BORDER_REPLICATE)
        y2 += -min(0, y1)
        y1 += -min(0, y1)
        x2 += -min(0, x1)
        x1 += -min(0, x1)
        return image, x1, x2, y1, y2

    def undistortImage(self, image):
        '''
        h,  w = image.shape[:2]
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cvSettings.cameraMatrix,cvSettings.distCoeffs,(w,h),1,(w,h))
        mapx,mapy = cv2.initUndistortRectifyMap(cvSettings.cameraMatrix,cvSettings.distCoeffs,None,newCameraMatrix,(w,h),5)
        dst = cv2.remap(image,mapx,mapy,cv2.INTER_LINEAR)
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]'''

        dst = cv2.undistort(image, cvSettings.cameraMatrix, cvSettings.distCoeffs) #TODO: Unterschied der zwei Methoden?
        return dst #Bild hat schwarzen Rahmen.

    def identifyCircle(self, img):
        #HoughCircles um mit geschnittenen Bildern zu erkennen ob es sich um einen Ball handelt.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=300, param2=30, minRadius=15, maxRadius=45)
        if circles is not None:
            circles = np.round(circles[0,:]).astype("int")
            if CVSettings.useDebugGUI:
                try:
                    for (x, y, r) in circles:
                        cv2.circle(img, (x, y), r, (0, 255, 0), 4)
                        cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                        cv2.imshow("output", img)
                        cv2.waitKey(1)
                except:
                    traceback.print_exc()
            return circles # (x, y, r)
        else:
            return None

    def calculateDistance(self, length, reallength):
        f_x = np.round(cvSettings.cameraMatrix[0,0]).astype("int")
        f_y = np.round(cvSettings.cameraMatrix[1,1]).astype("int")
        m_x = np.round(f_x / cvSettings.cFocalLength).astype("int")
        m_y = np.round(f_y / cvSettings.cFocalLength).astype("int")
        #TODO nur für runde Objekte:
        m = np.round((m_x+m_y)/2).astype("int")
        ss = length / m #size on sensor
        distance = np.round((reallength * cvSettings.cFocalLength) / ss).astype("int")
        return distance

    def readQRCode(self, img):
        data, bbox, rec = self.qr.detectAndDecode(img)
        if len(data) > 0:
            print bbox
            return data
        else:
            print("no QR Code found")

    def destroyCV(self): #experimentell
        self.cap.stop()
        self.cap.release()
        cv2.destroyAllWindows()

    if CVSettings.useStereoVision:
        def initStereoSGBM(self):
            self.stereo = cv2.StereoSGBM_create(
                minDisparity = cvSettings.minDisparity,
                numDisparities = cvSettings.numDisparities,
                blockSize = cvSettings.blockSize,
                uniquenessRatio = cvSettings.uniquenessRatio,
                speckleWindowSize = cvSettings.speckleWindowSize,
                speckleRange = cvSettings.speckleRange,
                disp12MaxDiff = cvSettings.disp12MaxDiff,
                P1 = cvSettings.P1,
                P2 = cvSettings.P2
                )
            
            self.stereoR=cv2.ximgproc.createRightMatcher(self.stereo)

        def initWLSFilter(self):
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.stereo)
            self.wls_filter.setLambda(cvSettings.lmbda)
            self.wls_filter.setSigmaColor(cvSettings.sigma)
            self.kernel = np.ones((1,1),np.uint8)

        def generateDistanceHeatmap(self, imgR, imgL):
            imgReR = cv2.remap(imgR,cvSettings.Right_Stereo_Map[0],cvSettings.Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
            imgReL = cv2.remap(imgL,cvSettings.Left_Stereo_Map[0],cvSettings.Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
            grayR = cv2.cvtColor(imgReR,cv2.COLOR_BGR2GRAY)
            grayL = cv2.cvtColor(imgReL,cv2.COLOR_BGR2GRAY)
            dispL = self.stereo.compute(grayL,grayR)
            dispR = self.stereoR.compute(grayR,grayL)
            dispL = np.int16(dispL)
            dispR = np.int16(dispR)
            filteredImg = self.wls_filter.filter(dispL,grayL,None,dispR)
            filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
            filteredImg = np.uint8(filteredImg)
            disp = ((dispL.astype(np.float32) / 16)-cvSettings.minDisparity) / cvSettings.numDisparities
            
            closing = cv2.morphologyEx(disp,cv2.MORPH_CLOSE, self.kernel)

            # Colors map
            dispc = (closing-closing.min())*255
            dispC = dispc.astype(np.uint8)
            disp_Color = cv2.applyColorMap(dispC,cv2.COLORMAP_JET)
            filt_Color = cv2.applyColorMap(filteredImg,cv2.COLORMAP_JET)

            if CVSettings.useDebugGUI:
                #cv2.imshow('Disparity', disp)
                cv2.imshow("imgReR",imgReR)
                cv2.imshow("imgReL",imgReL)
                #cv2.imshow('Closing',closing)
                #cv2.imshow('Color Depth',disp_Color)
                #cv2.imshow('Disparity Map', filteredImg)
                cv2.imshow('Filtered Color Depth',filt_Color)
                cv2.waitKey(1) & 0xFF
            return filteredImg, disp_Color #ggf. nicht beide von Bedeutung

    if IKImport:
        def faceObject(self, targetObject):
            #Objekt wird durch Bewegung im Zentrum des Bildes gehalten.
            #Berechnung des Winekls zum Objekt. Horizontales FOV abhängig von Auflösung und FPS?
            phiThreshold = 18
            dx= targetObject[0] - cvSettings.cXResolution/2
            degPerX= cvSettings.hFOV/cvSettings.cXResolution
            dphi = degPerX * dx
            dphi = np.round(dphi).astype('int')
            print dx
            print degPerX
            print dphi
            if (abs(dphi) >= phiThreshold):
                if(dphi < 0):
                    turn(dphi)
                elif(dphi > 0):
                    turn(dphi)
                return True
            else:
                return False

        def packenUndWerfen(self, targetObject):
            print targetObject[3]
            self.faceObject(targetObject)
            if targetObject[3] <= 240:
                packen()
                werfen()
            else:
                for i in xrange(1,int(targetObject[3]/4),1):
                    Gait(phi=0)
                    MoveIK(x,y,z,rotx,roty,rotz,cWalkingSpeed,cSyncSpeed)

        def ballSpielen(self, preset='red'):
            while True:
                try:
                    img = self.cap.read()
                    #cv2.imwrite("debug.jpg", img)
                    obj = self.findObjectFromColor(img,preset=preset)
                    self.packenUndWerfen(obj)
                except KeyboardInterrupt:
                    break

'''
#init
    if CVSettings.useStereoVision:
        IKcvfunction = IKcvfunction()
        imgR = IKcvfunction.capR.read() #benötigt um Zielauflösung anzunehmen
        imgL = IKcvfunction.capL.read() 
        time.sleep(5)        
        IKcvfunction.initStereoSGBM()
        IKcvfunction.initWLSFilter()
    else:
        IKcvfunction = IKcvfunction()
        img = IKcvfunction.cap.read()
        time.sleep(5)   
'''
'''
while(1):
    try:
        img = IKcvfunction.cap.read()
        cv2.imwrite("test.jpg", img)
        
    except:
        IKcvfunction.cap.stop()
        IKcvfunction.cap.release()
        traceback.print_exc()
        cv2.destroyAllWindows()
        break
'''