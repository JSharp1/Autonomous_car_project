# https://deparkes.co.uk/2014/11/23/how-to-load-data-into-python/
# https://docs.scipy.org/doc/numpy/user/basics.io.genfromtxt.html
# http://stackoverflow.com/questions/10819330/numpy-genfromtxt-column-names
# using data set http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset
# https://www.ibm.com/developerworks/community/blogs/jfp/entry/Elementary_Matrix_Operations_In_Python?lang=en
"""
load data into a dictionary with 4 key words: features, labels, sizes, coords. each containing
features 4D array: number, raw pixel data(height,width,channels) the image is converted to size (32,32)
labels 2D array: class, filename1, filename2
sizes 2D array: original image width and height
coords 4D array: containing orignal coords of object in image (x1,y1,x2,y2)

# The program enters each folder (class) and extracts images and data from the csv file to dictionary
# images are preprocessed with CLAHE and resized to 32,32
# classCounter = 43
#
# create a dictonary with keywords: features, labels, sizes, coords
# for each class (folder)
# 	read csv length (find how many to itterate over files)
# 	for each image/csv val in class store data into dictionary
# 		convert image to 32,32 and use CLAHE
#		store data into dict
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
Flag = True
#variables to increment through data
numberOfClasses = 42#43 classes/folders to loop through
names = ["Filename", "Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId"]	
features = np.zeros((1,3072),dtype=np.int8)#32x32)
labels = np.zeros((1, 3),dtype=np.int8)
sizes = np.zeros((1,2),dtype=np.int8)
coords = np.zeros((1,4),dtype=np.int8)
#			data_dict = {"Filename":data['Filename'][j], "ClassId":data['ClassId'][j]}

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
	#copy the Y channel
	Y = yuvImage[:,:,0].copy()
	#apply CLACHE method to YUV image and return
	clache = cv2.createCLAHE(clipLimit=1.0,tileGridSize=(8,8))
	yuvImage[:,:,0] = clache.apply(Y)
	return yuvImage

def main():
	folderPath = "GTSRB/Final_Training/Images/"
	if Flag == True:
		for i in range(1):#loop through each folder
			classCounter = "{0:05d}".format(i)#class increment var
			csvCounter = "GT-" + classCounter#csv file name
			csvPATH = str(folderPath + classCounter + "/" + csvCounter + ".csv")
			print "class", i
			print "***** saving training data from " + csvPATH
			data = np.genfromtxt(csvPATH, dtype=None, delimiter=';', skip_header=0, names=True)
			print "converting ", len(data), " samples"
			for j in range(len(data)):#loop over each row of data #len(data)
				#j = 28
				#save labels
				s = data['Filename'][j]
				#split file name into set and idx
				s1, s2  = s.split("_")
				s3, s4 = s2.split(".")
				x = data['ClassId'][j]
				global labels
				labels = np.vstack((labels, np.array([int(x),int(s1),int(s3)])))
				# print s,x,s1,s3,labels[j+1]
				#save img sizes
				h = data['Height'][j]
				w = data['Width'][j]
				global sizes
				sizes = np.vstack((sizes, np.array([h,w])))
				# print h,w,sizes[j+1]

				#save coords
				x1,x2,y1,y2 = data['RoiX1'][j], data['RoiX2'][j], data['RoiY1'][j], data['RoiY2'][j]
				global coords
				coords = np.vstack((coords, np.array([x1,x2,y1,y2])))
				# print j,x1,coords[j+1][0]

				#save images
				imageCounter_1 = "{0:05d}".format(int(s1))#the set of image in class
				imageCounter_2 = "{0:05d}".format(int(s3))#the number of image in set
				imagePATH = str(folderPath + classCounter + "/" + imageCounter_1 + "_"+ imageCounter_2 + ".ppm")
				# print "loading img from " + imagePATH
				src = cv2.imread(imagePATH)
				yuvImage = preprocess(src)

				#resize the image
				ratio = 32.0 / src.shape[1]
				dim = (32, int(src.shape[0] * ratio))
				resized = cv2.resize(yuvImage, dim, interpolation=cv2.INTER_NEAREST)
				resized1 = cv2.resize(resized, (32,32))
				img_array = resized1.reshape(1, 3072).astype(np.uint8)
				global features
				features = np.vstack((features, img_array))
				# img_mat = np.reshape(img_array,(32,32,3)).astype(np.uint8)
				# cv2.waitKey(0) 
				# cv2.destroyAllWindows()
				# print j

	np.savez('trainingData_dict.npz', features=features, labels=labels, sizes= sizes, coords = coords)
	elif Flag == False:
		print Flag
		folderPath = "GTSRB/Final_Test/Images"
		csvPATH = str(folderPath + "/GT-final_test.csv")
		print "***** saving test data from " + csvPATH
		data = np.genfromtxt(csvPATH, dtype=None, delimiter=';', skip_header=0, names=True)
		print "converting ", len(data), " samples"
		for j in range(len(data)):#loop over each row of data #len(data)

			#save labels: class,index
			s = data['Filename'][j]#string
			#split file name into idx
			s1, s2 = s.split(".")
			x = data['ClassId'][j]
			global labels
			labels = np.vstack((labels, np.array([int(x),int(s1),0])))
			# print s,x,s1,s3,labels[j+1]

			#save img sizes
			h = data['Height'][j]
			w = data['Width'][j]
			global sizes
			sizes = np.vstack((sizes, np.array([h,w])))
			# print h,w,sizes[j+1]

			#save coords
			x1,x2,y1,y2 = data['RoiX1'][j], data['RoiX2'][j], data['RoiY1'][j], data['RoiY2'][j]
			global coords
			coords = np.vstack((coords, np.array([x1,x2,y1,y2])))
			# print j,x1,coords[j+1][0]

			#save images
			imageCounter_1 = "{0:05d}".format(int(s1))#the set of image in class
			imagePATH = str(folderPath + "/" + imageCounter_1 + ".ppm")
			print "loading img from " + imagePATH + " converting ", len(data), " samples"
			src = cv2.imread(imagePATH)
			yuvImage = preprocess(src)

			#resize the image
			ratio = 32.0 / src.shape[1]
			dim = (32, int(src.shape[0] * ratio))
			resized = cv2.resize(yuvImage, dim, interpolation=cv2.INTER_NEAREST)
			resized1 = cv2.resize(resized, (32,32))
			img_array = resized1.reshape(1, 3072).astype(np.uint8)
			global features
			features = np.vstack((features, img_array))
			img_mat = np.reshape(img_array,(32,32,3)).astype(np.uint8)
			#cv2.imshow("rgb", src)
			# cv2.imshow("img_mat",cv2.cvtColor(img_mat,cv2.COLOR_YUV2BGR))
			# cv2.imshow("rgb resized", cv2.cvtColor(resized,cv2.COLOR_YUV2BGR))
			# print yuvImage.shape
			# print resized.shape
			# cv2.waitKey(0) 
			# cv2.imshow("yuv CLAHE",cv2.cvtColor(yuvImage,cv2.COLOR_YUV2BGR))
			# cv2.waitKey(0)     
			# cv2.destroyAllWindows()
			# print j

			#get a random sample and display
			# plt.figure()
			# #src
			# plt.subplot(1,2,1).set_title("SRC")
			# plt.subplot(1,2,1).axes.get_xaxis().set_visible(False), plt.subplot(1,2,1).axes.get_yaxis().set_visible(False)
			# plt.subplot(1,2,1).imshow(cv2.cvtColor(src,cv2.COLOR_BGR2RGB))
			# #y channel after CLAHE
			# plt.subplot(1,2,2).set_title("SRC CLAHE")
			# plt.subplot(1,2,2).axes.get_xaxis().set_visible(False), plt.subplot(1,2,2).axes.get_yaxis().set_visible(False)
			# plt.subplot(1,2,2).imshow(cv2.cvtColor(yuvImage,cv2.COLOR_YUV2RGB))
			# plt.ion()
			# # plt.pause(.001)
			# plt.close(j)
		np.savez('testData_dict.npz', features=features, labels=labels, sizes= sizes, coords = coords)

		# print npzfile["features"][3].shape
		# temp = np.array(npzfile["features"][3])
		# print temp
		# img_mat = np.reshape(npzfile["features"][3],(32,32,3)).astype(np.uint8)
		# cv2.imshow("img_mat",cv2.cvtColor(img_mat,cv2.COLOR_YUV2BGR))
		# cv2.waitKey(0) 
		# cv2.destroyAllWindows()

		# print npzfile.files
		# print npzfile["labels"][2][0],npzfile["labels"][2][1],npzfile["labels"][2][2]
		# print npzfile["labels"][2]
		# print labels.size
		# print labels[1].dtype
		# print data['Filename'][0]
		# print sizes[1][0].dtype
		# print data['Width'][0].dtype
		# print labels.shape
		# print data['Filename'].dtype
		# print data_dict['Filename'][210]
		# print data[1].dtype

	
	print '-' * 10
	print "data saved"
	npzfile = np.load('data_dict.npz', 'r')
	print "loading data"
	print "keys: " + str(npzfile.files)
	print "length of data " + str(len(npzfile["labels"])-1)#minus one as first row are zeros
	print '-' * 10


if __name__ == '__main__':
	main()
