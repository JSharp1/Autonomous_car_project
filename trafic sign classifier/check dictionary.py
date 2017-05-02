import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

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


def get_image(rand,npzfile, Flag):
	if Flag == True:
		folderPath = "GTSRB/Final_Training/Images/"
		#reconstruct the filename to show org img
		label = npzfile["labels"][rand,:,:]
		classCounter = "{0:05d}".format(int(label[:,0]))#class/folder increment var
		imageCounter_1 = "{0:05d}".format(int(label[:,1]))#the set of image in class
		imageCounter_2 = "{0:05d}".format(int(label[:,2]))#the number of image in set
		imagePATH = str(folderPath + classCounter + "/" + imageCounter_1 + "_"+ imageCounter_2 + ".ppm")
		print "Stored filename", imageCounter_1 + "_"+ imageCounter_2 + ".ppm"

	elif Flag == False:
		folderPath = "GTSRB/Final_Test/Images"
		#reconstruct the filename to show org img
		label = npzfile["labels"][rand,:,:]
		# classCounter = "{0:05d}".format(int(label[:,0]))#class/folder increment var
		imageCounter_2 = "{0:05d}".format(int(label[:,2]))#the set of image in class
		imagePATH = str(folderPath + "/" + imageCounter_2 + ".ppm")
		print "Stored filename",imageCounter_2 + ".ppm"


	print "label",label
	print "loading original image from " + imagePATH 
	src = cv2.imread(imagePATH)
	# load stored image
	sample_img = np.array(npzfile["features"][rand,:,:])
	img_mat = np.reshape(sample_img,(32,32,3)).astype(np.uint8)
	print "loading stored img shape, datatype, file name", img_mat.shape, img_mat.dtype
	return src,img_mat


Flag = False #if true check training data else test data_dict
total_samples = 0

def main():
	print Flag
	print "loading data"

	if Flag == True:
		
		npzfile = np.load('trainingData_dict.npz', 'r')
		print "loading training data"
		total_samples = len(npzfile["labels"])
		print "number of samples", total_samples
		print "keys: " + str(npzfile.files)
		print "features.shape", npzfile["features"].shape, "labels.shape", npzfile["labels"].shape, "sizes.shape", npzfile["sizes"].shape, "coords.shape", npzfile["coords"].shape
		for i in range(1):
			print '-' * 10
			#get a rand sample
			print "rand sample "
			src1,mat1 = get_image(np.random.randint(0,total_samples), npzfile, Flag)
			src2,mat2 = get_image(np.random.randint(0,total_samples), npzfile, Flag)


			# cv2.imshow("sample feature", cv2.cvtColor(img_mat,cv2.COLOR_YUV2BGR))
			# cv2.waitKey(0) 
			# cv2.destroyAllWindows()

			# get a random sample and display
			plt.figure()
			#src
			plt.subplot(2,2,1).set_title("SRC")
			plt.subplot(2,2,1).axes.get_xaxis().set_visible(False), plt.subplot(2,2,1).axes.get_yaxis().set_visible(False)
			plt.subplot(2,2,1).imshow(cv2.cvtColor(src1,cv2.COLOR_BGR2RGB))

			plt.subplot(2,2,2).set_title("SRC stored img")
			plt.subplot(2,2,2).axes.get_xaxis().set_visible(False), plt.subplot(2,2,2).axes.get_yaxis().set_visible(False)
			plt.subplot(2,2,2).imshow(cv2.cvtColor(mat1,cv2.COLOR_YUV2RGB))

			plt.subplot(2,2,3).set_title("SRC")
			plt.subplot(2,2,3).axes.get_xaxis().set_visible(False), plt.subplot(2,2,3).axes.get_yaxis().set_visible(False)
			plt.subplot(2,2,3).imshow(cv2.cvtColor(src2,cv2.COLOR_BGR2RGB))

			plt.subplot(2,2,4).set_title("SRC stored img")
			plt.subplot(2,2,4).axes.get_xaxis().set_visible(False), plt.subplot(2,2,4).axes.get_yaxis().set_visible(False)
			plt.subplot(2,2,4).imshow(cv2.cvtColor(mat2,cv2.COLOR_YUV2RGB))

			# plt.ion()
			plt.pause(.8)
			cv2.waitKey(0)     
			cv2.destroyAllWindows()
			plt.tight_layout()
			plt.show()
			plt.close()

	elif Flag == False:

		npzfile = np.load('testData_dict.npz', 'r')
		print "loading test data"
		total_samples = len(npzfile["labels"])
		print "number of samples", total_samples
		print "keys: " + str(npzfile.files)
		print "features.shape", npzfile["features"].shape, "labels.shape", npzfile["labels"].shape, "sizes.shape", npzfile["sizes"].shape, "coords.shape", npzfile["coords"].shape



		for i in range(1):
			print '-' * 10
			#get a rand sample
			print "rand sample "
			src1,mat1 = get_image(np.random.randint(0,total_samples), npzfile, Flag)
			print '-' * 10
		 	src2,mat2 = get_image(np.random.randint(0,total_samples), npzfile, Flag)

			# get a random sample and display
			plt.figure()
			#src
			plt.subplot(2,2,1).set_title("SRC")
			plt.subplot(2,2,1).axes.get_xaxis().set_visible(False), plt.subplot(2,2,1).axes.get_yaxis().set_visible(False)
			plt.subplot(2,2,1).imshow(cv2.cvtColor(src1,cv2.COLOR_BGR2RGB))

			plt.subplot(2,2,2).set_title("SRC stored img")
			plt.subplot(2,2,2).axes.get_xaxis().set_visible(False), plt.subplot(2,2,2).axes.get_yaxis().set_visible(False)
			plt.subplot(2,2,2).imshow(cv2.cvtColor(mat1,cv2.COLOR_YUV2RGB))

			plt.subplot(2,2,3).set_title("SRC")
			plt.subplot(2,2,3).axes.get_xaxis().set_visible(False), plt.subplot(2,2,3).axes.get_yaxis().set_visible(False)
			plt.subplot(2,2,3).imshow(cv2.cvtColor(src2,cv2.COLOR_BGR2RGB))

			plt.subplot(2,2,4).set_title("SRC stored img")
			plt.subplot(2,2,4).axes.get_xaxis().set_visible(False), plt.subplot(2,2,4).axes.get_yaxis().set_visible(False)
			plt.subplot(2,2,4).imshow(cv2.cvtColor(mat2,cv2.COLOR_YUV2RGB))

			# plt.ion()
			plt.pause(.8)
			cv2.waitKey(0)     
			cv2.destroyAllWindows()
			plt.tight_layout()
			plt.show()
			plt.close()



if __name__ == '__main__':
	main()
