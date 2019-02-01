# coding: utf-8
#Frederick Müller 
#Campus Velbert/Heiligenhaus, Hochschule Bochum, 2018/2019

import os
import glob
import cv2
import numpy as np
import copy
import traceback

class CVSettings:
    useDebugGUI = False #Legt fest ob eine GUI genutzt werden soll (nicht möglich auf dem Beaglebone)
    cXResolution = 1280 #Zielauflösung X-Achse
    cYResolution = 720 #Zielauflösung Y-Achse

    #### HoughCircle Variabeln (aktuell nicht aktiv genutzt)
    dp=2 #Inverse ratio of the accumulator resolution to the image resolution
    p1=100 #The higher threshold of the two passed to the Canny() edge detector 
    p2=170 #The accumulator threshold for the circle centers at the detection stage

    #### Variabeln zur Distanzbestimmung
    #siehe https://support.logitech.com/en_us/product/hd-pro-webcam-c920/specs
    cFocalLength = 3.67 #Brennweite in mm
    cRealObjectRadius = 35 #Realgröße des Objekts in mm
    cSensorHeight = 3.6 #Realhöhe des Bildsensors in mm (aktuell nicht genutzt) Quelle:https://www.techtravels.org/modifying-logitech-c920/
    hFOV = 70.42 #horizontales Field of View in Grad

    #### Standardfarben (abgestimmt auf drei bestimmte Softbälle)
    lowerRed=[(0, 140, 100), (168, 150, 100)] #zwei Intervalle, da H=180=0 ist
    upperRed=[(10, 255, 250), (180, 255, 175)] #zwei Intervalle, da H=180=0 ist
    lowerBlue = [(100, 65, 36)]
    upperBlue = [(120, 244, 212)]
    lowerYellow = [(13, 101, 77)]
    upperYellow = [(28, 240, 255)]

    #### Kalibrierung
    calibrationSuccess = False
    useStereoVision = False #Legt fest ob zwei Kameras genutzt werden sollen. (/dev/video0 muss die rechte Kamera sein; video1 links)
    #genutzes Schachbrettmuster: 9x7 (BxH), 20mm Seitenlänge
    nCols = 9 #Zeilenmenge des Schachbretts
    nRows = 7 #Spaltenmenge des Schachbretts
    sLength = 20 #Seitenlänge der Felder in mm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, sLength, 0.001) #Abbruchkriterium der Kalibrierung
    criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, sLength, 0.001)#Abbruchkriterium der Stereo-Kalibrierung

    #### SGBM Matcher (Semi-global block matcher; benötigt um die Bilder beider Kameras zusammenzufügen)
    #die Werte entsprechen größtenteils dem offiziellen Beispiel: https://github.com/opencv/opencv/blob/master/samples/cpp/stereo_match.cpp
    window_size = 3 #SADWindowSize (Sum of Absolute Differences)
    minDisparity = 0 #Minimum possible disparity value.
    numDisparities = 160 #Maximum disparity minus minimum disparity
    blockSize = 3 #Matched block size.
    uniquenessRatio = 15 #Margin in % by which the best computed cost function value should "win" the second best value to consider the found match correct
    speckleWindowSize = 200 #Maximum size of smooth disparity regions to consider their noise speckles and invalidate. 
    speckleRange = 32 #Maximum disparity variation within each connected component.
    disp12MaxDiff = 1 # Maximum allowed difference (in integer pixel units) in the left-right disparity check.
    P1 = 8*3*window_size**2 #The first parameter controlling the disparity smoothness.
    P2 = 32*3*window_size**2 #The second parameter controlling the disparity smoothness.

    #### WLS Filter
    lmbda = 8000 #Lambda is a parameter defining the amount of regularization during filtering.
    sigma = 1.4 #Sigma is a parameter defining how sensitive the filtering process is to source image edges.


    def __init__(self, skipCalibration=False):
        if not skipCalibration:
            self.loadCalibrationMatrix()

    def loadCalibrationMatrix(self, dualCamera = True):
        if (self.useStereoVision and not self.calibrationSuccess):
            try:
                self.cameraMatrixR = np.loadtxt("./cameraMatrixR.txt", delimiter=',')
                self.cameraMatrixL = np.loadtxt("./cameraMatrixL.txt", delimiter=',')
                self.distCoeffsR = np.loadtxt("./cameraDistortionR.txt", delimiter=',')
                self.distCoeffsL = np.loadtxt("./cameraDistortionL.txt", delimiter=',')
                

                raise Exception
                #TODO: Aktuell bewusster Error; die gespeicherten Stereo Maps nicht die benötigte Form haben. Kalibrierung muss jedes mal neu gemacht werden.  
                '''           
                mapRX = np.loadtxt("./Right_Stereo_MapX.txt", delimiter=',')#, dtype='uint16')
                mapRY = np.loadtxt("./Right_Stereo_MapY.txt", delimiter=',')#, dtype='uint16')
                mapLX = np.loadtxt("./Left_Stereo_MapX.txt", delimiter=',')#, dtype='uint16')
                mapLY = np.loadtxt("./Left_Stereo_MapY.txt", delimiter=',')#, dtype='uint16')

                
                #Wiederherstellen der ursprünglichen Tuple-Form
                self.Right_Stereo_Map = np.stack((mapRX, mapRY), axis=2)
                self.Left_Stereo_Map = np.stack((mapLX, mapLY), axis=2)
                self.Right_Stereo_Map = tuple(map(tuple, self.Right_Stereo_Map))
                self.Left_Stereo_Map = tuple(map(tuple, self.Left_Stereo_Map))
                '''

                print "Kalibrierungsmatrix erfolgreich geladen:"
                print self.cameraMatrixR
                print self.cameraMatrixL
                print "Distortion: "
                print self.distCoeffsR
                print self.distCoeffsL
                self.calibrationSuccess = True

            except:
                traceback.print_exc()
                print 'Keine Kalibrierungsmatrix vorliegend. Starte Kalibrierung...'
                self.saveSnapshotsStereo()
                self.cameraCalibrationStereo()

        if (not self.useStereoVision and not self.calibrationSuccess):
            try:
                self.cameraMatrix = np.loadtxt("./cameraMatrix.txt", delimiter=',')
                self.distCoeffs = np.loadtxt("./cameraDistortion.txt", delimiter=',')
                print 'Kalibrierungsmatrix erfolgreich geladen:'
                print self.cameraMatrix
                print "Distortion: ", self.distCoeffs
                self.calibrationSuccess = True

            except:
                print 'Keine Kalibrierungsmatrix vorliegend. Starte Kalibrierung...'
                self.saveSnapshots()
                self.cameraCalibration()


    def saveSnapshots(self):
        folder = "./images" #TODO: Funktion nur mit GUI ausführbar!
        print "Press Space to save a Snapshot, press 'q' to quit. (while camera image is the active window)"
        print "Images will be saved to the specified folder with current set resolution"
        try:
            if not os.path.exists(folder):
                os.makedirs(folder)
                try:
                    os.stat(folder)
                except:
                    os.mkdir(folder)
        except:
            print "Fehler beim erstellen des Snapshot Ordners"

        count = 0
        fileName = "%s/" %folder
        cap = cv2.VideoCapture(0)
        cap.open(0,cv2.CAP_V4L2)
        ret = cap.set(cv2.CAP_PROP_FRAME_WIDTH,self.cXResolution)
        ret = cap.set(cv2.CAP_PROP_FRAME_HEIGHT,self.cYResolution)

        while True:
            ret, image = cap.read()
            imageSave = copy.deepcopy(image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #Positionen der inneren Ecken des Schachbretts bestimmen.
            ret, corners = cv2.findChessboardCorners(gray, (self.nRows,self.nCols),None)
            if ret:
                #Verbesserung der Genauigkeit der Positionen
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self.criteria)
                #Ecken anzeigen
                image = cv2.drawChessboardCorners(image, (self.nRows,self.nCols), corners2,ret)
                cv2.imshow('image', image) 
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord(' '):
                    print ("Saving image ", count)
                    cv2.imwrite("%s%d.jpg"%(fileName, count), imageSave)
                    count += 1
        cap.release()

    def saveSnapshotsStereo(self):
        folder = "./imagesDual" #TODO: Funktion nur mit GUI ausführbar!
        print "Press Space to save a Snapshot, press 'q' to quit. (while camera image is the active window)"
        print "Images will be saved to the specified folder with current set resolution"
        try:
            if not os.path.exists(folder):
                os.makedirs(folder)
                try:
                    os.stat(folder)
                except:
                    os.mkdir(folder)
        except:
            print "Fehler beim erstellen des Snapshot Ordners"

        count = 0
        fileName    = "%s/" %folder
        capL = cv2.VideoCapture(1)
        capR = cv2.VideoCapture(0)
        capL.open(1,cv2.CAP_V4L2)
        capR.open(0,cv2.CAP_V4L2)
        ret = capL.set(cv2.CAP_PROP_FRAME_WIDTH,self.cXResolution)
        ret = capL.set(cv2.CAP_PROP_FRAME_HEIGHT,self.cYResolution)
        ret = capR.set(cv2.CAP_PROP_FRAME_WIDTH,self.cXResolution)
        ret = capR.set(cv2.CAP_PROP_FRAME_HEIGHT,self.cYResolution)

        while True:
            retL, imageL = capL.read()
            retR, imageR = capR.read()
            imageSaveL = copy.deepcopy(imageL)
            imageSaveR = copy.deepcopy(imageR)
            grayR = cv2.cvtColor(imageR,cv2.COLOR_BGR2GRAY)
            grayL = cv2.cvtColor(imageL,cv2.COLOR_BGR2GRAY)
            #Positionen der inneren Ecken des Schachbretts bestimmen.
            retR, cornersR = cv2.findChessboardCorners(grayR, (self.nRows,self.nCols),None)
            if retR: #getrenntes if zum beschleunigen
                retL, cornersL = cv2.findChessboardCorners(grayL, (self.nRows,self.nCols),None)
                if retL:
                    #Verbesserung der Genauigkeit der Positionen
                    corners2R = cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),self.criteria)
                    corners2L = cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),self.criteria)
                    #Ecken anzeigen
                    imageR = cv2.drawChessboardCorners(imageR, (self.nRows,self.nCols), corners2R,retR)
                    imageL = cv2.drawChessboardCorners(imageL, (self.nRows,self.nCols), corners2L,retL)
                    cv2.imshow('imageR', imageR)
                    cv2.imshow('imageL', imageL)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    if key == ord(' '):
                        print ("Saving image ", count)
                        cv2.imwrite("%sL%d.jpg"%(fileName, count), imageSaveL)
                        cv2.imwrite("%sR%d.jpg"%(fileName, count), imageSaveR)
                        count += 1
        capL.release()
        capR.release()

    def cameraCalibrationStereo(self):
        #Vorbereitung der benötigten Arrays
        objp = np.zeros((self.nCols*self.nRows,3), np.float32)
        objp[:,:2] = np.mgrid[0:self.nRows,0:self.nCols].T.reshape(-1,2)

        # Arrays zum Speichern der Bildinformationen
        objpoints = [] # 3D Ebene (Realebene)
        imgpointsR = [] # 2D Ebene (Bildebene)
        imgpointsL = [] # 2D Ebene (Bildebene)

        #Import der Bilder
        imagesR = sorted(glob.glob("./imagesDual/R*.jpg"))
        imagesL = sorted(glob.glob("./imagesDual/L*.jpg"))
        print "Anzahl rechts: " + str(len(imagesR))
        print "Anzalh links: " + str(len(imagesL))

        if (len(imagesR) < 10 or len(imagesL) < 10):
            print "Nicht genug Bilder zum Durchführen der Kalibrierung. Mindestens 10 Bilder sind erforderlich"
        else:
            nPatternFound = 0
            print "Genug Bilder gefunden. Drücken sie die Leertaste um das aktuelle Bild für die Kalibrierung zu nutzen; 'q' zum überspringen"

        for i in xrange(len(imagesR)):
            imgR = cv2.imread(imagesR[i])
            imgL = cv2.imread(imagesL[i])
            grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
            grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)

            #Positionen der inneren Ecken des Schachbretts bestimmen.
            retR, cornersR = cv2.findChessboardCorners(grayR, (self.nRows,self.nCols),None)
            retL, cornersL = cv2.findChessboardCorners(grayL, (self.nRows,self.nCols),None)
            if (retR and retL):
                #Verbesserung der Genauigkeit der Positionen
                corners2R = cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),self.criteria)
                corners2L = cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),self.criteria)
                #Ecken anzeigen
                DispImgR = cv2.drawChessboardCorners(imgR, (self.nRows,self.nCols), corners2R,retR)
                DispImgL = cv2.drawChessboardCorners(imgL, (self.nRows,self.nCols), corners2L,retL)
                cv2.imshow('imgR',DispImgR)
                cv2.imshow('imgL',DispImgL)
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    print "Bild übersprungen"
                    continue
                if key == ord(' '):
                    objpoints.append(objp)
                    imgpointsR.append(corners2R)
                    imgpointsL.append(corners2L)
                    print "Bild akzeptiert"
            else:
                print "error"

        cv2.destroyAllWindows()
        
        #Kamerakalibrierung einzeln
        ret, self.cameraMatrixR, self.distCoeffsR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1],None,None)
        ret, self.cameraMatrixL, self.distCoeffsL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1],None,None)

        print "Calibration Matrix: "
        print self.cameraMatrixR
        print self.cameraMatrixL
        print "Disortion: "
        print self.distCoeffsR
        print self.distCoeffsL

        #Speichern der Ergebnisse
        np.savetxt("./cameraMatrixR.txt", self.cameraMatrixR, delimiter=',')
        np.savetxt("./cameraMatrixL.txt", self.cameraMatrixL, delimiter=',')
        np.savetxt("./cameraDistortionR.txt", self.distCoeffsR, delimiter=',')
        np.savetxt("./cameraDistortionL.txt", self.distCoeffsL, delimiter=',')
        
        #Berechnen des Wiederausstoßfehlers
        mean_errorR = 0
        mean_errorL = 0
        for i in xrange(len(objpoints)):
            imgpoints2R, _ = cv2.projectPoints(objpoints[i], rvecsR[i], tvecsR[i], self.cameraMatrixR, self.distCoeffsR)
            imgpoints2L, _ = cv2.projectPoints(objpoints[i], rvecsL[i], tvecsL[i], self.cameraMatrixL, self.distCoeffsL)
            errorR = cv2.norm(imgpointsR[i],imgpoints2R, cv2.NORM_L2)/len(imgpoints2R)
            errorL = cv2.norm(imgpointsL[i],imgpoints2L, cv2.NORM_L2)/len(imgpoints2L)
            mean_errorR += errorR
            mean_errorL += errorL
        print "error right: ", mean_errorR/len(objpoints)
        print "error left: ", mean_errorL/len(objpoints)

        #Berechnung einer neuen, optimalen Kameramatrix auf Basis des Skalierungsparameters
        hR,  wR = (cv2.imread(imagesR[1])).shape[:2]
        hL,  wL = (cv2.imread(imagesL[1])).shape[:2]
        newCameraMatrixR, roiR = cv2.getOptimalNewCameraMatrix(self.cameraMatrixR,self.distCoeffsR,(wR,hR),1,(wR,hR))
        newCameraMatrixL, roiL = cv2.getOptimalNewCameraMatrix(self.cameraMatrixL,self.distCoeffsL,(wL,hL),1,(wL,hL))

        #Stereokalibrierung
        retS, MLS, dLS, MRS, dRS, R, T, E, F= cv2.stereoCalibrate(objpoints,imgpointsL,imgpointsR,self.cameraMatrixL,self.distCoeffsL,self.cameraMatrixR,self.distCoeffsR, grayR.shape[::-1],None,None,None,None, cv2.CALIB_FIX_INTRINSIC, self.criteria_stereo)
        #Berechnung der Rotationsmatrizen um beide Bilder auf die selbe Ebene zu projezieren.
        RL, RR, PL, PR, Q, roiL, roiR= cv2.stereoRectify(MLS, dLS, MRS, dRS,grayR.shape[::-1], R, T, None, None, None, None, None,cv2.CALIB_ZERO_DISPARITY, -1)
        #Berechnung der Stereo Maps für "Pixel zu Pixel" Zuordnung der Ausgangs- und Zielbilder
        self.Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR, grayR.shape[::-1], cv2.CV_16SC2)
        self.Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,grayR.shape[::-1], cv2.CV_16SC2)
        
        '''
        #TODO: Speichern funktionert so, aber import nicht! Die Tuple-Dimensionen sind nicht gleich?
        save = np.asarray(self.Right_Stereo_Map[:][:][0])
        np.savetxt("./Right_Stereo_MapX.txt", save[:,:,0], delimiter=',')
        np.savetxt("./Right_Stereo_MapY.txt", self.Right_Stereo_Map[:][:][1], delimiter=',')
        save = np.asarray(self.Left_Stereo_Map[:][:][0])
        np.savetxt("./Left_Stereo_MapX.txt", save[:,:,0], delimiter=',')
        np.savetxt("./Left_Stereo_MapY.txt", self.Left_Stereo_Map[:][:][1], delimiter=',')
        '''

    def cameraCalibration(self):
        #Kamerakalibrierung anhand gemachter Snapshots des Chessboards. Min. 10 Bilder in ./images benötigt
        #basierend auf der Anleitung der OpenCV-Doku: https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html

        #Vorbereitung der benötigten Arrays
        objp = np.zeros((self.nCols*self.nRows,3), np.float32)
        objp[:,:2] = np.mgrid[0:self.nRows,0:self.nCols].T.reshape(-1,2)

        # Arrays zum Speichern der Bildinformationen
        objpoints = [] # 3D Ebene (Realebene)
        imgpoints = [] # 2D Ebene (Bildebene)
        images = glob.glob("./images/*.jpg")

        print len(images)
        if len(images) < 10:
            print "Nicht genug Bilder zum Durchführen der Kalibrierung. Mindestens 10 Bilder sind erforderlich"
        else:
            nPatternFound = 0
            badImg = images[1]
            print "Genug Bilder gefunden. Drücken sie die Leertaste um das aktuelle Bild für die Kalibrierung zu nutzen; 'q' zum überspringen"

        for i in images:
            img = cv2.imread(i)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            #Positionen der inneren Ecken des Schachbretts bestimmen.
            ret, corners = cv2.findChessboardCorners(gray, (self.nRows,self.nCols),None)
            if ret == True:
                #Verbesserung der Genauigkeit der Positionen
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self.criteria)
                #Ecken anzeigen
                img = cv2.drawChessboardCorners(img, (self.nRows,self.nCols), corners2,ret)
                cv2.imshow('img',img)
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    print "Bild übersprungen"
                    continue
                if key == ord(' '):
                    objpoints.append(objp)
                    imgpoints.append(corners2)
                    print "Bild akzeptiert"

        cv2.destroyAllWindows()
        #Kamerakalibrierung
        ret, self.cameraMatrix, self.distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

        print "Calibration Matrix: "
        print self.cameraMatrix
        print "Disortion: ", self.distCoeffs
        #Speichern der Ergebnisse
        np.savetxt("./cameraMatrix.txt", self.cameraMatrix, delimiter=',')
        np.savetxt("./cameraDistortion.txt", self.distCoeffs, delimiter=',')
        mean_error = 0
        #Berechnen des Wiederausstoßfehlers
        for i in xrange(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], self.cameraMatrix, self.distCoeffs)
            error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        print "reprojection error: ", mean_error/len(objpoints)

    '''
    #Ansatz zum Speichern von Bildern ohne GUI
    def saveSnapshotsStereoAuto(self):        
        folder = "./imagesDual"
        try:
            if not os.path.exists(folder):
                os.makedirs(folder)
                try:
                    os.stat(folder)
                except:
                    os.mkdir(folder)
        except:
            pass

        count = 0
        fileName    = "%s/" %folder
        capL = cv2.VideoCapture(1)
        capR = cv2.VideoCapture(0)
        capL.open(1)
        capR.open(0)
        ret = capL.set(cv2.CAP_PROP_FRAME_WIDTH,self.cXResolution)
        ret = capL.set(cv2.CAP_PROP_FRAME_HEIGHT,self.cYResolution)
        ret = capR.set(cv2.CAP_PROP_FRAME_WIDTH,self.cXResolution)
        ret = capR.set(cv2.CAP_PROP_FRAME_HEIGHT,self.cYResolution)

        while count < 100:
            retL, imageL = capL.read()
            retR, imageR = capR.read()
            grayR = cv2.cvtColor(imageR,cv2.COLOR_BGR2GRAY)
            grayL = cv2.cvtColor(imageL,cv2.COLOR_BGR2GRAY)
            retL, cornersL = cv2.findChessboardCorners(grayL, (7,9),None)
            retR, cornersR = cv2.findChessboardCorners(grayR, (7,9),None)
            print count
            if (retR and retL):
                cv2.imwrite("%sL%d.jpg"%(fileName, count), imageL)
                cv2.imwrite("%sR%d.jpg"%(fileName, count), imageR)
                count += 1
        capL.release()
        capR.release()
        '''