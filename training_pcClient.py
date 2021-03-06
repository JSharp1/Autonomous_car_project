import socket#tcp/ip coms
import pygame#GUI and user input 
from pygame.locals import*
import time

print 'connecting to GPIO server'
sock_pi = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_pi.connect(('192.168.0.101',44476))
print 'connected to GPIO server'

#gui- start pygame, create a display import font
pygame.init()
screen = pygame.display.set_mode((320, 240))
pygame.display.set_caption("training")
f1 = pygame.font.SysFont("comicsansms", 24)
pygame.key.set_repeat(100, 150)

data = { # initital values
   1 : {'name' : 'steeringAngle', 'state' : 1400, 'min' : 1050, 'max' : 1750},
   2 : {'name' : 'motorSpeed', 'state' : 0, 'min' : 0, 'max' : 255}
   }


def sendData(var):
	print var
	sock_pi.send(var)

def clamp(n, minn, maxn):
	if n < minn:
		return minn
	elif n > maxn:
		return maxn
	else:
		return n

def prepData(x, val):
	data[x]['state'] += val
	data[x]['state'] = clamp(data[x]['state'],data[x]['min'],data[x]['max'])
	return str(data[1]['state'])+","+str(data[2]['state'])

def main():
	quit = False
	motorToggle = False#toggle motor on/off
	try:
		while not quit:
			for event in pygame.event.get():
				if event.type == QUIT:
					sendData("0,0")
					quit = True
					break
				elif event.type == KEYUP:
					pass
				elif event.type == KEYDOWN:	
					if event.key == K_UP:
						sendData(prepData(2,5))
					elif event.key == pygame.K_DOWN:
						sendData(prepData(2,-5))
					elif event.key == pygame.K_LEFT:
						sendData(prepData(1,40))
					elif event.key == pygame.K_RIGHT:
						sendData(prepData(1,-40))
					elif event.key == K_RETURN:
						sendData(prepData(2,255))
					elif event.key == K_BACKSPACE:
						sendData(prepData(2,-255))
					elif event.key == K_ESCAPE:
						quit = True
					elif event.key == pygame.K_s:
						sendData("123,123")#'code' for save flag
					elif event.key == pygame.K_c:
						sendData("321,321")#'code' for not save flag
						break
				time.sleep(.05)
	finally:
		print "closing..."
		sendData("close")
		sock_pi.close()
		pygame.display.quit()
		pygame.quit


if __name__ == '__main__':
	main()
