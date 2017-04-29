"""
dataset: http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset

file structure
main folder (training set)
-> folder 1 (class 1) 
--> image 1
--> image 2
--> image n
--> csv 1

-> folder 2 (class 2)
--> image 1
--> image 2
--> image n
--> csv 2

format of files
-> 00000_00001
--> 00000_00001.pnn
--> 00000_00002.pnn
--> 00002_0000n.pnn (set 2 of n images in class 1)
--> csv n

#useful links: https://matplotlib.org/examples/pylab_examples/subplots_demo.html
http://matplotlib.org/examples/color/colormaps_reference.html
""" 
import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess(rgbImage):
	"""Convert RGB img to YUV color space 
		uses CLAHE to normalise brightness channel (Y) of YUV image.
		N.B opencv will load as BGR.
	Args: 
		rgbImage: raw rgb image
	Returns:
		yuvImage: normalized yuv image
	"""
	#convert RGB img to YUV color space
	yuvImage = cv2.cvtColor(rgbImage,cv2.COLOR_BGR2YUV)
	cv2.imshow("RGB2YUV",yuvImage)
	#copy the Y channel
	Y = yuvImage[:,:,0].copy()
	cv2.imshow("Y channel before CLAHE",Y)
	#apply CLACHE method to YUV image and return
	clache = cv2.createCLAHE(clipLimit=1.0,tileGridSize=(8,8))
	yuvImage[:,:,0] = clache.apply(Y)
	cv2.imshow("Y channel after CLAHE", yuvImage[:,:,0])
	return Y , yuvImage


def main():
	#variables to increment through data
	folderPath = "GTSRB/Final_Training/Images/"
	numberOfClasses = 42#43 classes
	classCounter = "{0:05d}/".format(4)#class number var
	imageCounter_1 = "{0:05d}".format(0)#set of image in class var
	imageCounter_2 = "{0:05d}".format(1)#number of image in set var
	# file path to image and show img
	PATH = folderPath + classCounter + imageCounter_1 + "_" + imageCounter_2 + ".ppm"
	print PATH
	src = cv2.imread(PATH)
	cv2.imshow("rgb", src)
	# process img and show
	Y,yuvImage = preprocess(src)
	cv2.imshow("yuv CLAHE",cv2.cvtColor(yuvImage,cv2.COLOR_YUV2BGR))

	plt.figure(1)
	#src
	plt.subplot(2,2,1).set_title("SRC")
	plt.subplot(2,2,1).axes.get_xaxis().set_visible(False), plt.subplot(2,2,1).axes.get_yaxis().set_visible(False)
	plt.subplot(2,2,1).imshow(cv2.cvtColor(src,cv2.COLOR_BGR2RGB))
	#y channel before CLAHE
	plt.subplot(2,2,2).set_title("Y channel before CLAHE")
	plt.subplot(2,2,2).axes.get_xaxis().set_visible(False), plt.subplot(2,2,2).axes.get_yaxis().set_visible(False)
	plt.set_cmap('gray')
	plt.subplot(2,2,2).imshow(Y)
	#y channel after CLAHE
	plt.subplot(2,2,3).set_title("SRC CLAHE")
	plt.subplot(2,2,3).axes.get_xaxis().set_visible(False), plt.subplot(2,2,3).axes.get_yaxis().set_visible(False)
	plt.subplot(2,2,3).imshow(cv2.cvtColor(yuvImage,cv2.COLOR_YUV2RGB))
	#src CLAHE
	plt.subplot(2,2,4).set_title("Y channel after CLAHE")
	plt.subplot(2,2,4).axes.get_xaxis().set_visible(False), plt.subplot(2,2,4).axes.get_yaxis().set_visible(False)
	plt.set_cmap('gray')
	plt.subplot(2,2,4).imshow(yuvImage[:,:,0])

	plt.show()
	cv2.waitKey(5000); 
	cv2.destroyAllWindows()
	plt.close('all')

	print "end"

if __name__ == '__main__':
	main()