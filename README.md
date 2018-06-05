# c-numberDetector
Detects numbers by using KNN classification

For the training program we converted a colored image to grayscale and then a Gaussian Blur to blur
the image to reduce noise. A green rectangle will be drawn around the region of interests that the
program finds. Then when the program is executed, the training image is displayed for the user and then
they can assign each region of interest to an ASCII character for the entire image. Once you are done, a
classifications and images XML file are produced that we use for the testing program as the training set.

Once the testing program is run, the webcam will go live and you can point it in any direction you please
and it will have contours outline any points of interest. It will then do it’s best job with the images
and classifications files that we have provided to match up the points of interest to the number it resembles
the most. This program uses a KNN approach to match data to the training set. KNN or, k-nearest neighbors
algorithm, can be used for classification or regression. Usually a weight is assigned to the neighbors of a
point, so the nearest neighbors contribute more to the average than farther ones. Neighbors are taken from
a set for which the class is known.

