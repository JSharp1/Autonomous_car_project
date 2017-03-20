import socket#tcp/ip coms
import pygame#GUI and user input 
from pygame.locals import*

print 'connecting to GPIO server'
sock_pi = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_pi.connect(('192.168.0.101',12346))
print 'connected to GPIO server'

#gui- start pygame, create a display import font
pygame.init()
screen = pygame.display.set_mode((320, 240))
pygame.display.set_caption("training")
f1 = pygame.font.SysFont("comicsansms", 24)
pygame.key.set_repeat(50, 100)

data = { # initital values
   1 : {'name' : 'steeringAngle', 'state' : 1400, 'min' : 500, 'max' : 2400},
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
	try:
		while not quit:
			for event in pygame.event.get():
				if event.type == QUIT:
					quit = True
					break
				elif event.type == KEYUP:
					pass
				elif event.type == KEYDOWN:
					if event.key == K_UP:
						sendData(prepData(2,10))
					elif event.key == pygame.K_DOWN:
						sendData(prepData(2,-10))
					elif event.key == pygame.K_LEFT:
						sendData(prepData(1,-50))
					elif event.key == pygame.K_RIGHT:
						sendData(prepData(1,50))
					elif event.key == K_ESCAPE:
						quit = True
						break
	finally:
		print "closing..."
		sendData("close")
		sock_pi.close()
		pygame.display.quit()
		pygame.quit


if __name__ == '__main__':
	main()