# https://deparkes.co.uk/2014/11/23/how-to-load-data-into-python/
# https://docs.scipy.org/doc/numpy/user/basics.io.genfromtxt.html
# http://stackoverflow.com/questions/10819330/numpy-genfromtxt-column-names
# using data set http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset

import numpy as np

def main():
	#variables to increment through data
	numberOfClasses = 42#43 classes
	numberOfFeatures = 0#init as zero for now
	classIdx, featIdx = 0,0
	# dynamic file path
	folderPath = "GTSRB/Final_Training/Images/"
	classCounter = "{0:05d}".format(0)#class increment var
	csvCounter = "GT-" + classCounter#csv file name
	imageCounter_1 = "{0:05d}".format(0)#the set of image in class
	imageCounter_2 = "{0:05d}".format(0)#the number of image in set
	imagePATH = str(folderPath + classCounter + "/" + imageCounter_1 + "_"+ imageCounter_2 + ".ppm")
	csvPATH = str(folderPath + classCounter + "/" + csvCounter + ".csv")
	print "***** loading img from " + imagePATH
	print "***** loading data from " + csvPATH

	names = ["Filename", "Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId"]
	data = np.genfromtxt(csvPATH, dtype=None, delimiter=';', skip_header=0, names=True)
	print data[0]
	print data[0].dtype
	print data[0].dtype.names
	print data['Filename'][0]

if __name__ == '__main__':
	main()