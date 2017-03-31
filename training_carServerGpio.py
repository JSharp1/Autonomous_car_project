#NB run sudo pigpiod to start pigpio thread before running this file
import pigpio
import socket
import numpy as np
#import cv2
import shutil
import os
import threading
import time

print 'starting GPIO server'
# start a socket and listen for clients (0.0.0.0 
# on all interfaces)
server_socket = socket.socket()
server_socket.bind(('0.0.0.0', 12346))
server_socket.listen(0)
print "socket is listening for a client"
conn, addr = server_socket.accept()
print "connected"
print 'Got connection from', addr

SERVO_PIN = 25
MOTOR_PIN = 24
#connect to pigpiod daemon
pi = pigpio.pi()
# setup pin as an output
pi.set_mode(SERVO_PIN, pigpio.OUTPUT)
pi.set_mode(MOTOR_PIN, pigpio.OUTPUT)
d1 = 0
d2 = 0
sampleNo = 0
label_array = np.zeros((1, 2))

def saveData():
	global sampleNo
	sampleNo = sampleNo + 1
	# print sampleNo
	filePath_1 = "data/tempImg.jpg"
	filePath_2 = "data/" + "{0:04d}.jpg".format(sampleNo)
	shutil.copyfile(filePath_1,filePath_2)
	global d1,d2,label_array
	label_array = np.vstack((label_array, np.array([d1,d2])))



def main():
	# saveFlag = True
	# t1 = time.time()
	# t2 = .5 + t1
	exitFlag = False

	while not exitFlag:
		try:
			while True:
				# if time.time() < t2:
				# 	saveData()
				# 	t1 = time.time()
				# 	t2 = .5 + t1
				# t = threading.Timer(1, saveData, args=())
				# t.daemon = True
				# t.start()
				print 'save'
				# inc = 0
				# if saveFlag == True:
				# 	saveData()
				saveData()
				time.sleep(.1)
				data = conn.recv(20)
				if data:
					if data == "close":
						print 'client close connection'
						break
					elif data == "save":
						saveFlag = True
					elif data == "stop saving":
						saveFlag == False
					print 'data recieved'
					print data
					global d1,d2
					d1, d2  = data.split(",", 1 )
					data = "s"+d1+","+d2
					print 'servo: ', d1,' ','motor: ', d2

					pi.set_servo_pulsewidth(SERVO_PIN, int(d1))
					pi.set_PWM_dutycycle(MOTOR_PIN,int(d2))
				else:
					print 'client closed connection'
					break
		finally:
			conn.close()
			print 'closing and saving data'
			np.save('data/data.npz', label_array)
			exitFlag = True

if __name__ == '__main__':
	main()
