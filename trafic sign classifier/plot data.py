import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
	#load data
	trainingData = np.load('trainingData_dict.npz', 'r')
	testData = np.load('testData_dict.npz', 'r')

	#rand int
	rand = np.random.randint(0,len(trainingData["labels"])-1)
	print "-"*10
	print "rand no: ",rand
	

	#print keys
	print "-"*10
	print "trainingData keys: " + str(trainingData.files)
	print "testData keys: " + str(testData.files)

	#print no of samples
	print "-"*10
	noTrainingSamples = len(trainingData['labels'])
	print "no of training samples " , noTrainingSamples
	noTestSamples = len(testData['labels'])
	print "no of test samples " , noTestSamples

	#print shape of image
	img_array = trainingData["features"][rand]
	img_mat = np.reshape(img_array,(32,32,3)).astype(np.uint8)
	print "-"*10
	print "features shape",trainingData["features"].shape
	print "shape of image " , img_mat.shape

	#trainingData label shape
	print "-"*10
	trainingLabels = np.array(trainingData['labels'])
	print "trainingData label shape", trainingLabels.shape
	print "training label sample", trainingData['labels'][rand][0]
	testLabels = np.array(testData['labels'])
	print "testLabels label shape", testLabels.shape

	#show sample
	print "-"*10
	print "trainingData label sample "
	for i in range(6): print "no: ", i, trainingLabels[i]

	#show unique classes
	print "-"*10
	testData = np.array(testData['labels'])
	uniqueTrain = len(np.unique(trainingLabels[:,:,0]))
	uniqueTest = len(np.unique(testData[:,:,0]))
	print "unique classes in trainingData", uniqueTrain
	print "unique classes in testData", uniqueTest

	# hist of training data
	print "-"*10
	classes = 43
	trainingBins = np.zeros(classes)# create 1D vector
	testBins = np.zeros(classes+1)
	
	#loop through training data
	for i in range(noTrainingSamples):
		# print i, trainingLabels[i,0,0];
		trainingBins[int(trainingLabels[i,0,0])] += 1
	print trainingBins

	for i in range(noTestSamples): 
		# print i, testLabels[i,0];
		testBins[int(testLabels[i,0,0])] += 1
	print testBins

	#plot training data
	from matplotlib.pyplot import *	
	plt.subplot(1,2,1)
	title('training samples per class')
	plt.bar(range(classes), trainingBins, width=1/1.5, color='blue')
	plt.subplot(1,2,2)
	title('test samples per class')
	plt.bar(range(classes+1), testBins, width=1/1.5, color='blue')
	plt.show()

if __name__ == '__main__':
	main()
