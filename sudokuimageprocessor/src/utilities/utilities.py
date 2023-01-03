#!/usr/bin/env python3
import numpy as np
import cv2 as cv
from os.path import realpath, dirname


thisDirectory = realpath(dirname(__file__))

config = {
	'model_data': thisDirectory+'/../../'+'resources/modeldata/knn_model_data_compressed.npz',
	'training_assets': thisDirectory+'/../../'+'resources/sudokutrainingdigits/assets/',
	'test_image': thisDirectory+'/../../'+'resources/testimages/5.tiff',
	'sample': 'testdata'
}

def saveModel(name,train,train_labels):
	np.savez_compressed(name,train=train, train_labels=train_labels)

def resizeImage(img,dim):
	img = cv.resize(img, (dim), interpolation = cv.INTER_AREA)
	return img
 
def reshapeData(img,dim):
	img = np.reshape(img,(dim[0],dim[1]*dim[1])).astype(np.float32)
	return img

def filterImages(image,filenames):
	filtered = []
	filteredFileNames = []
	index = 0
	for img in image:		 
		number_of_white_pix = np.sum(img == 255) 
		number_of_black_pix = np.sum(img == 0)
		if(number_of_black_pix != 0 and number_of_white_pix != 0):
			percent = number_of_white_pix/number_of_black_pix
			# try under 0.05
			if (percent > 0.05):
				filtered.append(img)
				filteredFileNames.append(filenames[index])
		index+=1
	return filtered,filteredFileNames

def knnTraining(images,labels,save=False,filename=config['model_data']):
	"""Returns a KNN object"""    
	images = np.array(images)
	labels = np.array(labels)
	# The images are 28*28 so reshape into rows 784 long (28*28)
	images = images[:,:].reshape((-1,28*28)).astype(np.float32)
	# Initiate kNN, train it on the training data
	knn = cv.ml.KNearest_create()
	knn.train(images, cv.ml.ROW_SAMPLE, labels)
	# Save trained data
	if save == True:
		saveModel(filename,images,labels)
	return knn

def knnfindNearest(knn,data,k):
	"""returns ret,result,neighbours,dist"""
	data = np.array(data)
	return knn.findNearest(data,k)