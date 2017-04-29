import numpy as np
import cv2
import matplotlib.pyplot as plt

Flag = False #if true check training data else test data_dict

def main():
	print "loading data"
	print Flag

	if Flag == True:
		folderPath = "GTSRB/Final_Training/Images/"
		npzfile = np.load('trainingData_dict.npz', 'r')
		print "keys: " + str(npzfile.files)
		print "length of data " + str(len(npzfile["labels"])-1)#minus one as first row are zeros

		for i in range(10):
			print '-' * 10
			#get a rand sample
			rand = np.random.randint(0,len(npzfile["labels"])-1)
			print "rand no: ",rand

			#show the image
			img_array = npzfile["features"][rand]
			img_mat = np.reshape(img_array,(32,32,3)).astype(np.uint8)
			# cv2.imshow("rgb", src)
			# cv2.imshow("yuv CLAHE",cv2.cvtColor(yuvImage,cv2.COLOR_YUV2BGR))
			# cv2.imshow("img_mat",cv2.cvtColor(img_mat,cv2.COLOR_YUV2BGR))
			# cv2.imshow("rgb resized", cv2.cvtColor(resized,cv2.COLOR_YUV2BGR))
			# print yuvImage.shape
			# print resized.shape

			#classId = npzfile["labels"][rand][0]
			classId = "{0:05d}".format(int(npzfile["labels"][rand][0]))#class increment var
			imageId_1 = "{0:05d}".format(int(npzfile["labels"][rand][1]))#the set of image in class
			imageId_2 = "{0:05d}".format(int(npzfile["labels"][rand][2]))#the number of image in set
			imagePATH = str(folderPath + classId + "/" + imageId_1 + "_"+ imageId_2 + ".ppm")

			print npzfile["labels"][rand]
			print imagePATH
			
			src = cv2.imread(imagePATH)
			# cv2.imshow("src", src)

			plt.figure()
			#src
			plt.subplot(1,2,1).set_title("SRC")
			plt.subplot(1,2,1).axes.get_xaxis().set_visible(False), plt.subplot(1,2,1).axes.get_yaxis().set_visible(False)
			plt.subplot(1,2,1).imshow(cv2.cvtColor(src,cv2.COLOR_BGR2RGB))
			#y channel after CLAHE
			plt.subplot(1,2,2).set_title("SRC CLAHE")
			plt.subplot(1,2,2).axes.get_xaxis().set_visible(False), plt.subplot(1,2,2).axes.get_yaxis().set_visible(False)
			plt.subplot(1,2,2).imshow(cv2.cvtColor(img_mat,cv2.COLOR_YUV2RGB))
			plt.ion()
			plt.pause(.6)
			#cv2.waitKey(0) 
			cv2.destroyAllWindows()
			plt.close()

	elif Flag == False:
		folderPath = "GTSRB/Final_Test/Images"
		npzfile = np.load('testData_dict.npz', 'r')
		print "keys: " + str(npzfile.files)
		print "length of data " + str(len(npzfile["labels"])-1)#minus one as first row are zeros

		for i in range(10):
			print '-' * 10
			#get a rand sample
			rand = np.random.randint(0,len(npzfile["labels"])-1)
			print "rand no: ",rand

			#show the image
			img_array = npzfile["features"][rand]
			img_mat = np.reshape(img_array,(32,32,3)).astype(np.uint8)
			# cv2.imshow("rgb", src)
			# cv2.imshow("yuv CLAHE",cv2.cvtColor(yuvImage,cv2.COLOR_YUV2BGR))
			# cv2.imshow("img_mat",cv2.cvtColor(img_mat,cv2.COLOR_YUV2BGR))
			# cv2.imshow("rgb resized", cv2.cvtColor(resized,cv2.COLOR_YUV2BGR))
			# print yuvImage.shape
			# print resized.shape

			#classId = npzfile["labels"][rand][0]
			classId = "{0:05d}".format(int(npzfile["labels"][rand][0]))#class increment var
			imageId_1 = "{0:05d}".format(int(npzfile["labels"][rand][1]))#the set of image in class
			imageId_2 = "{0:05d}".format(int(npzfile["labels"][rand][2]))#the number of image in set
			imagePATH = str(folderPath + "/" + imageId_1 + ".ppm" )

			print npzfile["labels"][rand]
			print imagePATH

			
			src = cv2.imread(imagePATH)
			# cv2.imshow("src", src)

			plt.figure()
			#src
			plt.subplot(1,2,1).set_title("SRC")
			plt.subplot(1,2,1).axes.get_xaxis().set_visible(False), plt.subplot(1,2,1).axes.get_yaxis().set_visible(False)
			plt.subplot(1,2,1).imshow(cv2.cvtColor(src,cv2.COLOR_BGR2RGB))
			#y channel after CLAHE
			plt.subplot(1,2,2).set_title("SRC CLAHE")
			plt.subplot(1,2,2).axes.get_xaxis().set_visible(False), plt.subplot(1,2,2).axes.get_yaxis().set_visible(False)
			plt.subplot(1,2,2).imshow(cv2.cvtColor(img_mat,cv2.COLOR_YUV2RGB))
			plt.ion()
			plt.pause(.6)
			#cv2.waitKey(0) 
			cv2.destroyAllWindows()
			plt.close()


if __name__ == '__main__':
	main()
