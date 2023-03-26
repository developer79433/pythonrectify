#!/usr/bin/env python3
import cv2 as cv
import numpy as np
import os

def cleanFinalImage(warped):
	warped = cv.medianBlur(warped,5)
	warped = cv.GaussianBlur(warped,(5,5),0)
	warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
	warped = cv.adaptiveThreshold(warped,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,31,2)
	return warped

def cleanFinalImage2(warped):
	warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
	warped = cv.GaussianBlur(warped,(5,5),0)
	kernel = np.ones((3, 3), np.uint8)
	(thresh, warped) = cv.threshold(warped, 127, 255, cv.THRESH_BINARY_INV)
	#warped = cv.dilate(warped, kernel, iterations=1)
	return warped

