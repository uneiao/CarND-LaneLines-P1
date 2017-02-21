#**Finding Lane Lines on the Road** 

##Writeup Report

###This is a writeup report for CarND project 1.

---

**NOTES**

* The `LaneFinding` class implements the main pipeline.
* I implemented average/filter/extrapolate methods inside the pipeline class instead of the `draw_lines` function.
* The codes also could be found in the proj1.py file.

[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

###1. About implementation of the pipeline.

My pipeline consisted of several steps including:
First, I converted the images to grayscale;
Then I used Gaussian filter to blur the grayscale image;
Third, applied Canny Edge Transform to find the potential edges of lane lines;
And then restricted the image into a smaller ROI with a triangular mask;
Fifth, used Hough Transform to help group points along the edges and form output lines;
At Last, I used my binning and averaging method to get two solid lines over the Hough lines;

In order to draw a single line on the left and right lanes after I get Hough lines output,
I divide the set of lines into two bins by checking if slope of a line is positive or negative, putting
lines with positive slope into positive bin and others to negative bin.
Then for each line of the two bins, calculated the euclidean distance from a vanishing point to the line.
I considered lines with large distance as noises and filtered out them, because obviously both lanes in
the videos are all heading to the middle of a frame.
After filtering noises, I averaged previous output lines with their length as theirs weights to get the final output.
I assumed that longer a line was, more convincing for it to cover the lane line.

[//]: # (If you'd like to include images to show how the pipeline works, here is how to include an image: )

[//]: # (![alt text][image1])

###2. Shortcomings of current pipeline

Some shortcomings could be:

* Hough Transform is hard to tune and not robust enough to get desired edge lines,
especially driving along some 'dirty' dashed lane lines.
* The vanishing point filter cannot work when a car is crossing a lane, lane lines are not heading towards
the camera's vanishing point.
* A straight line equation does not suit curvy lane lines.
* Simple grayscale over the whole image will lose some gradient information as shown in challenge.mp4.
* Sometimes width of a lane line varies, so does its shape. I didn't check the width in the example videos and
used a constant line thickness. But lane line width matters, also different shape of lane line has different
meaning.

###3. Suggest possible improvements to your pipeline

Some possible improvement would be:

* Try RANSAC or some other methods to fit or extract lines over the edges.
* Maybe use PCA to analyse the distribution of edge points, to help find out a lane line's orientation.
* Use polynomial equation with higher degree or Bezier splines to represent a lane line.
* Use some local histogram equalization method to keep contrast values at any part of a image better.
* Find some proper patch features, apply machine learning methods in order to recognize different lane lines.
* Use some tracking methods to restrict ROI and smooth the results.
