# coding: utf-8
#Thassilo Bücker, Alexander Orzol, Frederick Mueller, Moritz Kolb
#Campus Velbert/Heiligenhaus, Hochschule Bochum, 2016/2017
#
#Überarbeitet von Frederick Mueller 
#Campus Velbert/Heiligenhaus, Hochschule Bochum, 2018/2019

#Das Programm stellt Funktionen zum Tasten zur Verfügung   

cWalkingSpeed = 150
cSyncSpeed = False

##### Imports #########################
from IKfunction import *
from ax12 import Ax12
import time
from fernbed import *

servos = Ax12()
#Init()

def Inittasten():
	#Initialposition zum Tasten / mittleren Beine vorne, vordere Beine oben 
	initalpos = [512,512,658,658,257,257,512,512,658,658,257,257,512,512,658,658,257,257]
	pos1 = [512,512,658,658,257,257,512,512,658,658,257,257,512,512,800,800,257,257]
	pos2 = [512,512,658,658,257,257,512,512,658,658,257,257,350,674,800,800,257,257]
	pos3 = [512,512,658,658,257,257,512,512,658,658,257,257,350,674,658,658,257,257]
	pos4 = [512,512,750,750,257,257,512,512,658,658,257,257,350,674,658,658,257,257]

	start = [412,612,700,700,255,255,512,512,624,624,255,255,350,674,624,624,255,255]
	speed = [300,300,300,300,300,300,300,300,300,300,300,300,300,300,300,300,300,300]
	try:
		servos.moveSpeedSync(pos1,speed)
		time.sleep(1)
		servos.moveSpeedSync(pos2,speed)
		time.sleep(1)
		servos.moveSpeedSync(pos3,speed)
		time.sleep(1)
		servos.moveSpeedSync(pos4,speed)
		time.sleep(1)
	except:
		print "Servo move failed"

def Vorderbeinefuehlen():
	# Mit den Vorderbeinen wird geprüft ob sich ein Hindernis vor ihm befindet	
	start= [512,512,750,750,257,257,512,512,658,658,257,257,350,674,658,658,257,257]
	speed = [300,300,300,300,300,300,300,300,300,300,300,300,300,300,300,300,300,300]
	#start= [412,612,255,767,512,512,575,449,400,624,700,324,350,674,400,624,700,324]
	for i in xrange(0,18):
		try:

			servos.move(i+1,start[i])
		except:
			print "Servos move failed"	
	load = 0
	for i in reversed(xrange(525,767,4)):
		if load > 1000 or load < 60:
			load = -1
			try:
				servos.move(4,i)
				time.sleep(0.25)
				load = servos.readLoad(4)
			except:
				i+=1
			print "|" +str(i) + "|" + str(load) + "\n"
		else:
			print "Gegestand erkannt"
			break
	try:
		servos.move(4,767)
	except:
		print "Servos move failed"
	load = 0	
	for i in xrange(255,490,4):
		if load < 1050:
			load = -1
			try:
				servos.move(3,i)
				time.sleep(0.25)
				load = servos.readLoad(3)
			except:
				i-=1
			print "|" +str(i) + "|" + str(load) + "\n"
		else:
			print "Gegenstand erkannt"
			break
	try:
		servos.move(3,255)
	except:
		print "Servos move failed"

def packen():
	#mit den Vorderbeinen wird versucht ein Gegenstand zu packen
	gegenstand = 0
	Inittasten()
	time.sleep(1)
	start= [512,512,750,750,257,257,512,512,658,658,257,257,350,674,658,658,257,257]
	speed = [300,300,300,300,300,300,300,300,300,300,300,300,300,300,300,300,300,300]
	#start= [412,612,255,767,512,512,575,449,400,624,700,324,350,674,400,624,700,324]
	#start = [511, 509, 255, 767, 763, 259, 509, 514, 361, 666, 764, 259, 512, 510, 359, 659, 763, 258]
	#for i in xrange(0,18):
	try:
		servos.moveSpeedSync(start,speed)
	except:
		print "Servo move failed"
			
	load = 0
	load1 = 0
	for i in xrange(512,300,-5):
		#if load >1000 or load <250 and  load1 < 1250:
		if load >1000 or load <300 and  load1 < 1325:
			load = -1
			load1 = -1
			try:
				servos.move(1,i)
				servos.move(2,512+abs(512-i))
				time.sleep(0.25)
				load  = servos.readLoad(1)
				load1 = servos.readLoad(2)
			except:
				i+=1
			print  "|1|" +str(i) + "|" + str(load) + " |2|" +str(512+abs(512-i)) + "|" + str(load1)
		else:
			print "Gegestand erkannt"
			gegenstand = 1
			break
	try:
		servos.moveSpeedSync([412,612,750,750,257,257,512,512,658,658,257,257,350,674,658,658,257,257], 500)
	except:
		print "Servo move failed"
	return gegenstand

def werfen():
	#Wirft einen gepackten Gegenstand nach vorne
	initalpos = [512,512,658,658,257,257,512,512,658,658,257,257,512,512,658,658,257,257]
	pos1 = [512,512,658,658,257,257,512,512,658,658,257,257,512,512,800,800,257,257]
	pos2 = [512,512,658,658,257,257,512,512,658,658,257,257,350,674,800,800,257,257]
	start= [512,512,750,750,257,257,512,512,658,658,257,257,350,674,658,658,257,257]
	speed = [500,500,500,500,500,500,500,500,500,500,500,500,500,500,500,500,500,500]
	try:
		servos.moveSpeedSync([300,720,750,750,700,700,512,512,658,658,257,257,350,674,658,658,257,257], speed)
		time.sleep(0.1)
		servos.moveSpeedSync([462,562,750,750,700,700,512,512,658,658,257,257,350,674,658,658,257,257], speed)
		time.sleep(1)
		servos.moveSpeedSync([512,512,658,658,257,257,512,512,658,658,257,257,350,674,658,658,257,257], speed)
		time.sleep(1)
		servos.moveSpeedSync(pos2, speed)
		time.sleep(1)
		servos.moveSpeedSync(pos1, speed)
		time.sleep(1)
		servos.moveSpeedSync(initalpos, speed)
		time.sleep(1)
	except:
		print "werfen failed"
#Inittasten()
