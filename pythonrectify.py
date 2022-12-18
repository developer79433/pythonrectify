#!/usr/bin/env python3

import math
import numpy as np
import cv2 as cv


EDGE_DETECTION_SIZE=500
CONTOUR_COUNT=5


def showImage(title, img):
	if True:
		cv.imshow(title, img)
		cv.waitKey(0)
		cv.destroyAllWindows()

def min_element_idx(arr):
	return min(range(len(arr)), key=arr.__getitem__)

def max_element_idx(arr):
	return max(range(len(arr)), key=arr.__getitem__)

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	# NOTE: Therefore the "top-left point" is neither necessarily the top-most point,
	# nor necessarily the left-most point.
	sums = [ pts[0][0] + pts[0][1], pts[1][0] + pts[1][1], pts[2][0] + pts[2][1], pts[3][0] + pts[3][1] ]
	# dump_array("sum:", sums)
	ordered = [None] * 4
	ordered[0] = pts[min_element_idx(sums)]
	ordered[2] = pts[max_element_idx(sums)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	differences = [ pts[0][1] - pts[0][0], pts[1][1] - pts[1][0], pts[2][1] - pts[2][0], pts[3][1] - pts[3][0] ]
	# dump_array("differences:", differences)

	ordered[1] = pts[min_element_idx(differences)]
	ordered[3] = pts[max_element_idx(differences)]
	# dump_point_array("order_points ordered", ordered)
	return ordered

def pythagorean_distance(p1, p2):
	return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def four_point_transform(image, pts):
	# dump_point_array("four_point_transform", pts);
	# obtain a consistent order of the points and unpack them individually
	rect = np.float32(order_points(pts))
	# dump_point_array("four_point_transform rect", rect);
	tl = rect[0]
	tr = rect[1]
	br = rect[2]
	bl = rect[3]

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	maxWidth = int(max(pythagorean_distance(br, bl), pythagorean_distance(tr, tl)))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	maxHeight = int(max(pythagorean_distance(tr, br), pythagorean_distance(tl, bl)))

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.float32([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]])
	# dump_array("dst", dst);

	# compute the perspective transform matrix and then apply it
	transformMatrix = cv.getPerspectiveTransform(rect, dst)
	# print("Matrix: %d" % transformMatrix)
	# display_image("Perspective input", image)
	warped = cv.warpPerspective(image, transformMatrix, (maxWidth, maxHeight))
	# showImage("Perspective output", warped)
	return warped

def processImage(image):
	showImage("Original", image)
	if image.shape[1] > EDGE_DETECTION_SIZE or image.shape[0] > EDGE_DETECTION_SIZE:
		# The image is bigger than we need in order to detect edges.
		# Isotropically scale it down to speed up edge detection.
		ratio = max(image.shape[1], image.shape[0]) / EDGE_DETECTION_SIZE
		smaller = cv.resize(image, (int(image.shape[1] // ratio), int(image.shape[0] // ratio)))
	else:
		ratio = 1
		smaller = image
	# showImage("Work", smaller)
	# Convert inplace to greyscale for contour detection
	smaller = cv.cvtColor(smaller, cv.COLOR_BGR2GRAY)
	# Remove high frequency noise
	smaller = cv.GaussianBlur(smaller, (3, 3), 0)
	# Find edges using Canny
	edges = cv.Canny(smaller, 75, 200)
	# showImage("Edges", edges)
	# Find contours in edge image
	contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	# Find the largest roughly quadrilateral contour
	quadrilateral = None
	largestArea = 0
	idx = 0
	quadrilateral_index = 0
	for contour in contours:
		contour_len = cv.arcLength(contour, True)
		simplified = cv.approxPolyDP(contour, 0.02 * contour_len, True)
		if len(simplified) == 4:
			area = cv.contourArea(contour, False)
			if area > largestArea:
				largestArea = area
				quadrilateral = simplified
				quadrilateral_index = idx
		idx += 1
	# print("border_idx: %d" % quadrilateral_index)
	cv.drawContours(smaller, contours, quadrilateral_index, (255, 0, 0), cv.FILLED)
	# showImage("Contours", smaller)
	# Warp image
	if quadrilateral is not None:
		# dump_point_vector("Quadrilateral before scaling", quadrilateral);
		# Rescale the quadrilateral's extreme points, which were calculated from a scaled-down temporary image,
		# back up to the full-size input image.
		quadrilateral = quadrilateral * ratio
		# dump_point_vector("Quadrilateral after scaling", quadrilateral);
		quadrilateral = [ point[0] for point in quadrilateral ]
		warped = four_point_transform(image, quadrilateral)
		showImage("Rectified", warped)

if __name__ == "__main__":
	img = cv.imread('input.jpg')
	processImage(img)
