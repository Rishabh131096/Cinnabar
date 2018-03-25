# Cinnabar


Allocates 2D array from photo of Sudoku grid.


OCR is a standard K-Nearest Neighbour Algorithm. Adaptive Thresholding is used to make the program robust to illumination changes and get a binary image. Then, the four corner points of the largest contour are taken. This is the Sudoku Grid in the newspaper/magazine.


The image is then converted to a 600 by 600 image to make it robust to perspective transform. Straight Lines are detected using Canny edge Detector and Hough Line Detector. A mask is created and is used to remove the straight grid lines to get a grid of numbers.


The contours are detected in the image(the numbers) and are passed to the OCR. We then receive a 2D grid which we sort according to x coordinates and y cordinates to get the final array.
