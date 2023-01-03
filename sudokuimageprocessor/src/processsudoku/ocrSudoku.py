#!/usr/bin/env python3
import numpy as np
from ..utilities.utilities import filterImages, resizeImage, reshapeData, knnTraining, knnfindNearest
import os
import cv2 as cv

thisDirectory = os.path.realpath(os.path.dirname(__file__))

config = {
	'model_data': thisDirectory+'/../../resources/modeldata/knn_model_data_compressed.npz',
	'savedata': thisDirectory+ '/../../resources/testdata'
}

def tileImage(warped,savedata=False):
	width = int(warped.shape[1] /9)
	height = int(warped.shape[0] / 9)

	shrinkRate = 0.8
	widthShrunk = int(width*shrinkRate)
	heightShrunk = int(height*shrinkRate)

	halfWidthDiffShrunk = int((width-widthShrunk) / 2)
	halfHeightDiffShrunk = int((height-heightShrunk) / 2)
 
	tiledImages = []
	for column in range(9):
		for row in range(9):		
			tiledImages.append(warped[ height*column+halfHeightDiffShrunk:height*(column+1)-halfHeightDiffShrunk,width*row+halfWidthDiffShrunk:width*(row+1)-halfWidthDiffShrunk])
			if savedata == True:
				cv.imwrite(f'{config["savedata"]}/{column}-{row}.tiff',warped[ height*column+halfHeightDiffShrunk:height*(column+1)-halfHeightDiffShrunk,width*row+halfWidthDiffShrunk:width*(row+1)-halfWidthDiffShrunk])
	return tiledImages
	

def filterOutBlankTiles(tiledImages):
	fileNames=[f"{i}-{j}" for i in range(9) for j in range(9)]
	filtered,filteredFileNames = filterImages(tiledImages,fileNames)
	imgArrayFiltered = np.array(filtered)
	return (imgArrayFiltered,filteredFileNames)
 
def reshapeImage(imgArrayFiltered):
	resizedImageArray = [resizeImage(img,(28,28)) for img in imgArrayFiltered ]
	reshapedImages = reshapeData(resizedImageArray,(-1,28,28))
	return reshapedImages
 
def knnOCR(reshapedImages):
	with np.load(config['model_data']) as data:
		train = data['train']
		train_labels = data['train_labels']
	knn = knnTraining(train,train_labels)
	ret,result,neighbours,dist = knnfindNearest(knn,reshapedImages,5)
	return result

def printResults(result,filteredFileNames,sudokoGrid=True):
	res=[[0]*9 for i in range(9)]

	index =0    
	for fn in filteredFileNames:
		res[int(fn[0])][int(fn[2])] = int(result[index][0])
		index+=1

	for i in range(0,9):
		for j in range(0,9):
			print(res[i][j], end=" ")
		if sudokoGrid == True:
			print(" ")