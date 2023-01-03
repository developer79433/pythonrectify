#!/usr/bin/env python3
import sys
from ....pythonrectify import processCamera, processImage
from .cleanSudokuImage import cleanFinalImage2, cleanFinalImage
from .ocrSudoku import tileImage, filterOutBlankTiles, reshapeImage, knnOCR, printResults
import cv2 as cv

def main():
	if len(sys.argv) > 1:
			filename = sys.argv[1]
			img = cv.imread(filename)
			warped = processImage(img)
			warped = cleanFinalImage2(warped)
			tiledImages = tileImage(warped,savedata=False)
			imgArrayFiltered,filteredFileNames = filterOutBlankTiles(tiledImages)
			reshapedImages = reshapeImage(imgArrayFiltered)
			result = knnOCR(reshapedImages)
			printResults(result,filteredFileNames,True)
	else:
		processCamera()