import cv2 as cv
from os import listdir
from os.path import isfile, join, realpath, dirname
import sys
import numpy as np
thisDirectory = realpath(dirname(__file__))
sys.path.append(thisDirectory+'/../utilities')
from utilities import knnTraining

config = {
	'model_data': thisDirectory+'/../../'+'resources/modeldata/knn_model_data_compressed.npz',
	'training_assets': thisDirectory+'/../../'+'resources/sudokutrainingdigits/assets/',
	'test_image': thisDirectory+'/../../'+'resources/testimages/5.tiff',
	'sample': 'testdata'
}

def saveModel(name,train,train_labels):
	np.savez_compressed(name,train=train, train_labels=train_labels)

def getImagesAndLabels(path):
# Given a path (assuming the last directories are named by number) return an array of images and the associated labels
# E.g path is archive/assets and last directory is 1 or 2 or 3, i.e. archive/assets/1
	labels=[]
	allFiles=[]
	filesInDir=[]
	for dir in range(1,10):
		filesInDir = [path+str(dir)+'/'+f for f in listdir(path+str(dir)) if isfile(join(path+str(dir), f))]
		allFiles += filesInDir
		labels += [dir]*len(filesInDir)
	images = [cv.imread(file) for file in allFiles]
	images = [cv.cvtColor(img,cv.COLOR_BGR2GRAY) for img in images]
	return (images,labels)

def reshapeData(img,dim):
	img = img[:,:].reshape(-1,dim[0]*dim[1]).astype(np.float32)
	return img
 
def resizeImage(img,dim):
	img = cv.resize(img, (dim), interpolation = cv.INTER_AREA)
	return img

#def knnTraining(images,labels,save,filename=config['model_data']):
#	"""Returns a KNN object"""    
#	images = np.array(images)
#	labels = np.array(labels)
#	# The images are 28*28 so reshape into rows 784 long (28*28)
#	images = reshapeData(images,(28,28))
#	# Initiate kNN, train it on the training data
#	knn = cv.ml.KNearest_create()
#	knn.train(images, cv.ml.ROW_SAMPLE, labels)
#	# Save trained data
#	if save == True:
#		saveModel(filename,images,labels)
#	return knn

def getTestData(path):
	"""returns an np array image"""
	img = cv.imread(path)
	img = cv.cvtColor(img,cv.COLOR_BGR2GRAY) 
	img=np.array(img)
	img = resizeImage(img,(28,28))
	img = reshapeData(img,(28,28))
	return img

def knnfindNearest(knn,data,k):
	"""returns ret,result,neighbours,dist"""
	data = np.array(data)
	return knn.findNearest(data,k)

def main():
	images,labels  = getImagesAndLabels(config['training_assets'])

	knn = knnTraining(images,labels,True)

	img = getTestData(config['test_image'])

	ret,result,neighbours,dist = knnfindNearest(knn,img,5)

	print(int(result[0][0]))

if __name__ == "__main__":
	main()
