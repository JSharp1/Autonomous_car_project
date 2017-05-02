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


"""
load data into a dictionary with 4 key words: features, labels, sizes, coords. containing
features 4D array: number, raw pixel data(height,width,channels) the image is converted to size (32,32)
labels 2D array: class, filename
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

"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

Flag = False #if true store training data else test data

#variables to increment through data
numberOfClasses = 43# classes/folders to loop through
names = ["Filename", "Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId"]	

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


def jitterImage(I):
    """ perturb the image.
    rotating it by between [-15,15],
    Randomly jitter image by scaling it by between [0.85,1.15], 
    change position by [-4,4]
    Args:
        I: input image
    Returns:
        jitteredI: jittered image
    """
    cols = I.shape[1]
    rows = I.shape[0]
    # rotation
    dst = I.copy()
    theta = random.uniform(-15,15)
    R = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
    dst = cv2.warpAffine(I,R,(cols,rows),dst,borderMode=cv2.BORDER_TRANSPARENT,flags=cv2.INTER_CUBIC)
    # scale
    fx = random.uniform(0.85,1.15)
    fy = random.uniform(0.85,1.15)
    dst = cv2.resize(dst,(0,0),dst,fx,fy)
    # transformation
    xdisplacement = random.uniform(-4, 4)
    ydisplacement = random.uniform(-4, 4)
    M = np.float32([[1,0,xdisplacement],[0,1,ydisplacement]])
    dst = cv2.warpAffine(dst,R,(cols,rows),dst,borderMode=cv2.BORDER_TRANSPARENT)
    return dst


def total_sample_count(numOfClasses):
	total_size_data = 0
	folderPath = "GTSRB/Final_Training/Images/"
	for i in range(numOfClasses):#loop through each folder
		classCounter = "{0:05d}".format(i)#class increment var
		csvCounter = "GT-" + classCounter#csv file name
		csvPATH = str(folderPath + classCounter + "/" + csvCounter + ".csv")
		data = np.genfromtxt(csvPATH, dtype=None, delimiter=';', skip_header=0, names=True)
		size_data = len(data)
		total_size_data += size_data
	return total_size_data


def main():
	sample_count_sum = total_sample_count(numberOfClasses)
	sample_count_running_total = 0

	print Flag
	if Flag == True:
		features = np.array(np.zeros((sample_count_sum,32,32,3),dtype=np.int8));# print "features", features.shape 
		labels = np.array(np.zeros((sample_count_sum,1, 3),dtype=np.int8));# print "labels", labels.shape 
		sizes = np.array(np.zeros((sample_count_sum,1,2),dtype=np.int8));# print "sizes", sizes.shape 
		coords = np.array(np.zeros((sample_count_sum,1,4),dtype=np.int8));# print "coords", coords.shape
		folderPath = "GTSRB/Final_Training/Images/"

		for i in range(numberOfClasses):#loop through each folder
			classCounter = "{0:05d}".format(i)#class/folder increment var
			csvCounter = "GT-" + classCounter#csv file located inside each folder
			csvPATH = str(folderPath + classCounter + "/" + csvCounter + ".csv")
			data = np.genfromtxt(csvPATH, dtype=None, delimiter=';', skip_header=0, names=True)
			#initialise all arrays to store data

			for j in range(len(data)):#loop over each row of data #len(data)
				#loop over rows in class and get the image file name for processing
				s = data['Filename'][j]
				# print "total stored data ", j+sample_count_running_total+1
				# print "total img count", sample_count_sum
				# print "***** saving training data from " + csvPATH + " converting ", j, "of", len(data), " samples"

				#split file name into set and idx
				s1, s2  = s.split("_")
				s3, s4 = s2.split(".")
				x = data['ClassId'][j]
				global labels
				labels[j+sample_count_running_total] = [int(x),int(s1),int(s3)]

				# print s,x,s1,s3,labels[j+1]
				#save img sizes
				h = data['Height'][j]
				w = data['Width'][j]
				global sizes
				sizes[j+sample_count_running_total] = [h,w]
				# print h,w,sizes[j+1]

				#save coords
				x1,x2,y1,y2 = data['RoiX1'][j], data['RoiX2'][j], data['RoiY1'][j], data['RoiY2'][j]
				global coords
				coords[j+sample_count_running_total] = [x1,x2,y1,y2]
				# print j,x1,coords[j+1][0]

				#save images
				imageCounter_1 = "{0:05d}".format(int(s1))#the set of image in class
				imageCounter_2 = "{0:05d}".format(int(s3))#the number of image in set
				imagePATH = str(folderPath + classCounter + "/" + imageCounter_1 + "_"+ imageCounter_2 + ".ppm")
				# print "saving training data from " + imagePATH + " converting ", j, "of", len(data), " samples"

				# print "loading img from " + imagePATH
				src = cv2.imread(imagePATH)
				#resize the image
				ratio = 32.0 / src.shape[1]
				dim = (32, int(src.shape[0] * ratio))
				resized = cv2.resize(src, dim, interpolation=cv2.INTER_NEAREST)
				resized1 = cv2.resize(resized, (32,32)).astype(np.uint8); 

				yuvImage = preprocess(resized1)
				jitImg = jitterImage(yuvImage)

				# img_array = resized1.reshape(1, 3072).astype(np.uint8)
				global features
				# features[j] = cv2.cvtColor(resized1 , cv2.COLOR_YUV2BGR)
				features[j+sample_count_running_total] = jitImg

				# print j,src.shape,yuvImage.shape, jitImg.shape, resized1.shape, features.shape

				# features = np.vstack((features,cv2.cvtColor(resized1 , cv2.COLOR_YUV2RGB)), axis=0)
				# img_mat = np.reshape(img_array,(32,32,3)).astype(np.uint8)

				# print j

				# cv2.imshow("rgb", src)
				# cv2.imshow("img_mat",cv2.cvtColor(img_mat,cv2.COLOR_YUV2BGR))
				# cv2.imshow("rgb resized", cv2.cvtColor(resized,cv2.COLOR_YUV2BGR))

				# cv2.imshow("src", cv2.cvtColor(src  ,cv2.COLOR_YUV2BGR))
				# cv2.imshow("src resized", cv2.cvtColor(resized1  ,cv2.COLOR_YUV2BGR))
				# cv2.imshow("clashe", cv2.cvtColor(yuvImage  ,cv2.COLOR_YUV2BGR))
				# cv2.imshow("jit", cv2.cvtColor(jitImg  ,cv2.COLOR_YUV2BGR))

				# print yuvImage.shape
				# print resized.shape
				# cv2.waitKey(0) 
				# cv2.imshow("yuv CLAHE",cv2.cvtColor(yuvImage,cv2.COLOR_YUV2BGR))
				# cv2.waitKey(0)     
				# cv2.destroyAllWindows()

				# Image[0:32,0:32,:] = cv2.cvtColor(X_train[cidx[0],:,:,:],cv2.COLOR_YUV2RGB)
    # 			Image[34:66,0:32,:] = cv2.cvtColor(X_train[cidx[1],:,:,:],cv2.COLOR_YUV2RGB)
    # 			Image[0:32,34:66,:] = cv2.cvtColor(X_train[cidx[2],:,:,:],cv2.COLOR_YUV2RGB)
    # 			Image[34:66,34:66,:] = cv2.cvtColor(X_train[cidx[3],:,:,:],cv2.COLOR_YUV2RGB)

				# get a random sample and display
				# plt.figure()
				# #src
				# plt.subplot(2,2,1).set_title("SRC")
				# plt.subplot(2,2,1).axes.get_xaxis().set_visible(False), plt.subplot(2,2,1).axes.get_yaxis().set_visible(False)
				# plt.subplot(2,2,1).imshow(cv2.cvtColor(src,cv2.COLOR_BGR2RGB))
				# #y channel after CLAHE
				# plt.subplot(2,2,2).set_title("SRC resized")
				# plt.subplot(2,2,2).axes.get_xaxis().set_visible(False), plt.subplot(2,2,2).axes.get_yaxis().set_visible(False)
				# plt.subplot(2,2,2).imshow(cv2.cvtColor(resized1,cv2.COLOR_BGR2RGB))
				# #y channel after CLAHE
				# plt.subplot(2,2,3).set_title("SRC CLAHE")
				# plt.subplot(2,2,3).axes.get_xaxis().set_visible(False), plt.subplot(2,2,3).axes.get_yaxis().set_visible(False)
				# plt.subplot(2,2,3).imshow(cv2.cvtColor(yuvImage,cv2.COLOR_YUV2RGB))
				# #y channel after CLAHE
				# plt.subplot(2,2,4).set_title("SRC JITTER")
				# plt.subplot(2,2,4).axes.get_xaxis().set_visible(False), plt.subplot(2,2,4).axes.get_yaxis().set_visible(False)
				# plt.subplot(2,2,4).imshow(cv2.cvtColor(jitImg,cv2.COLOR_YUV2RGB))
				# # plt.ion()
				# plt.pause(.8)
				# cv2.waitKey(0)     
				# cv2.destroyAllWindows()
				# plt.tight_layout()
				# plt.show()
				# plt.close()
				# print j

				print "-"*10
				print "class ", i
				print "saving training data from " + imagePATH + " converting ", j+1, "of", len(data), " samples"
				print str(j+sample_count_running_total+1) + " of " + str(sample_count_sum) + " total"

			sample_count_running_total += len(data)


		# print i
		npFileName = 'trainingData_dict.npz'

	elif Flag == False:
		folderPath = "GTSRB/Final_Test/Images"
		csvPATH = str(folderPath + "/GT-final_test.csv")
		data = np.genfromtxt(csvPATH, dtype=None, delimiter=';', skip_header=0, names=True)

		features = np.array(np.zeros((len(data),32,32,3),dtype=np.int8));# print "features", features.shape 
		labels = np.array(np.zeros((len(data),1, 3),dtype=np.int16));# print "labels", labels.shape 
		sizes = np.array(np.zeros((len(data),1,2),dtype=np.int8));# print "sizes", sizes.shape 
		coords = np.array(np.zeros((len(data),1,4),dtype=np.int8));# print "coords", coords.shape

		for j in range(len(data)):#loop over each row of data #len(data)
			print "-"*10
			#save labels: class,index
			s = data['Filename'][j]#get the string
			#split file name into idx,file extension : an example file name is 00005.ppm
			s1, s2 = s.split(".")
			x = data['ClassId'][j]
			global labels
			labels[j] = ([int(x),int(0),s1])#add a zero to keep the format / dimensions the same
			# print s,x,s1,labels[j]

			#save img sizes
			h = data['Height'][j]
			w = data['Width'][j]
			global sizes
			sizes[j] = [h,w]
			# print h,w,sizes[j+1]

			#save coords
			x1,x2,y1,y2 = data['RoiX1'][j], data['RoiX2'][j], data['RoiY1'][j], data['RoiY2'][j]
			global coords
			coords[j] = [x1,x2,y1,y2]
			# print j,x1,coords[j+1][0]

			#save images
			imageCounter_1 = "{0:05d}".format(int(s1))#the set of image in class
			imagePATH = str(folderPath + "/" + imageCounter_1 + ".ppm")
			src = cv2.imread(imagePATH)
			yuvImage = preprocess(src)
			jitImg = jitterImage(yuvImage)
			#resize the image
			ratio = 32.0 / src.shape[1]
			dim = (32, int(src.shape[0] * ratio))
			resized = cv2.resize(jitImg, dim, interpolation=cv2.INTER_NEAREST)
			resized1 = cv2.resize(jitImg, (32,32)).astype(np.uint8); 
			global features
			features[j] = resized1
			# img_mat = np.reshape(img_array,(32,32,3)).astype(np.uint8)
			print "saving test data from " + imagePATH + " converting ", j+1, "of", len(data), " samples"

		npFileName = 'testData_dict.npz'


	print '-' * 10
	# cut the first row of zeros out
	features = features[:,:,:]; labels = labels[:,:]; sizes = sizes[:,:]; coords = coords[:,:]
	# save data
	np.savez( npFileName, features = features, labels = labels, sizes = sizes, coords = coords)
	print "data saved"

	print '-' * 10
	# check sizes
	print "check... loading data:", npFileName
	npzfile = np.load( npFileName, 'r')
	print "features.shape", features.shape, "labels.shape", labels.shape, "sizes.shape", sizes.shape, "coords.shape", coords.shape

	print "keys: " + str(npzfile.files)
	print "number of samples " + str(len(npzfile["labels"]))

	print "loading labels sample:", npzfile["labels"][0:1,:,:]#second sample, all vector, last position
	# print "features", npzfile["features"][1:2,:,:]#second sample, all vector, last position
	sample_img = np.array(npzfile["features"][1,:,:])
	img_mat = np.reshape(sample_img,(32,32,3)).astype(np.uint8)
	print "sample mat shape, datatype",img_mat.shape, img_mat.dtype
	cv2.imshow("sample feature", cv2.cvtColor(img_mat,cv2.COLOR_YUV2BGR))

	# cv2.waitKey(5000) 
	# cv2.destroyAllWindows()
	# classTemp = npzfile["labels"]; 
	# print classTemp.shape, classTemp.dtype
	# print classTemp[:5,0:3,1:3]

if __name__ == '__main__':
	main()
