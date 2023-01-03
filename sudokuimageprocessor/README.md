# sudokuimageprocessor
This project takes a photgraph as input (a file, e.g .jpg, .tiff, .png etc) of a sudoku puzzle and outputs the numbers interpretted through opencv image cleaning and opencv ocr using k-nearest-neighbour machine learning algorithm.

# Preliminary steps

## 1 

unzip sudokuimageprocessor/resources/sudokutrainingdigits/archive.zip into the folder it is in.

## 2

Run the following to produce a zipped file of the model...
sudokuimageprocessor/src/k_nearest_neighbor_training/trainKNN.py

# Usage

Run pythonrectify as a module passing in the image file to process.

e.g python -m pythonrectify pythonrectify/sudokuimageprocessor/resources/testimages/sudoku2.jpg


