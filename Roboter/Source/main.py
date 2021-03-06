# coding: utf-8
#Thassilo Bücker, Alexander Orzol, Frederick Müller, Moritz Kolb
#Campus Velbert/Heiligenhaus, Hochschule Bochum, 2016/2017
#
#Überarbeitet von Frederick Müller 
#Campus Velbert/Heiligenhaus, Hochschule Bochum, 2018/2019

#Startprogramm fuer Start ueber Konsole bzw. Hauptmenue.

import IKfunction
IKfunction.Init2()
import time
openCVEnabled = True

try:
	from IKcvfunction import IKcvfunction
	IKcvfunction = IKcvfunction()
except:
	print "IKcvfunction konnte nicht importiert werden. (Ist OpenCV installiert?)"
	print "Fortfahren ohne Kamerafunktionen \n"
	openCVEnabled = False
	time.sleep(1)

def ProgrammSelect():

	while(1):
		print "Hauptmenu"
		print "Handbetrieb          = 1"
		print "Automatikbetrieb     = 2"
		print "Fernbedinungsbetrieb = 3"
		print "Drehfunktionstest = 4"
		if openCVEnabled:
			print "Ball spielen = 5"
		print "Programm beenden = 0"

		programmSelect = raw_input("Welches Programm wollen Sie ausfuehren") 
		if (programmSelect == "1"):
			IKfunction.Handbetrieb();
		if (programmSelect == "2"):
			IKfunction.newAutomatikbetrieb();
		if (programmSelect == "3"):
			IKfunction.Fernbedienungsbetrieb();
		if (programmSelect == "4"):
			programmSelect = int(raw_input("Um wie viel Grad wollen Sie drehen? Rechtsrum positiv "))
			IKfunction.turn(programmSelect)
		if (programmSelect == "0"):
			if openCVEnabled:
				IKcvfunction.destroyCV()
			break;
		if openCVEnabled:
			if (programmSelect == "5"):
				programmSelect = raw_input("Welcher Ball wird genutzt (1=red/2=blue/3=yellow)") 
				if programmSelect == "1" : IKcvfunction.ballSpielen("red")
				elif programmSelect == "2" : IKcvfunction.ballSpielen("blue")
				elif programmSelect == "3" : IKcvfunction.ballSpielen("yellow")
ProgrammSelect()