# -*- coding: utf-8 -*-

import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    # return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[0, 0, 255], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img, lines

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)




# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.
class LaneFinding:

    def __init__(self, test_image_dir):
        self.input_dir = test_image_dir
        self.output_dir = test_image_dir

    def load_image(self, image_name):
        image = cv2.imread("%s/%s" % (self.input_dir, image_name))
        return image

    def save_image(self, image_name, image):
        return cv2.imwrite("%s/%s" % (self.output_dir, image_name), image)

    def slope_binning(self, lines):
        """
        Binning lines by their slopes, into two bins,
        one for positive, another for negative
        """
        positve_idx = 0
        negative_idx = 1

        binnings = [[], []]
        slopes = [[], []]
        binnings[positve_idx] = []
        binnings[negative_idx] = []

        for line in lines:
            dx, dy = line[0][0:2] - line[0][2:4]
            slope = dy / dx
            if slope >= 0:
                binnings[positve_idx].append(line)
                slopes[positve_idx].append(slope)
            else:
                binnings[negative_idx].append(line)
                slopes[negative_idx].append(slope)

        return binnings, slopes

    def scan_binning(self, binning, slopes, delta_slope_err=0.07):
        filtered_binning = []
        filtered_slopes = []
        median_slope = np.median(slopes)
        for i, slope in enumerate(slopes):
            if abs(slope - median_slope) <= delta_slope_err:
                filtered_binning.append(binning[i])
                filtered_slopes.append(slope)

        #TODO: empty filtered_binning
        _max_vec = np.max(filtered_binning, axis=0)
        max_vpos = np.max([_max_vec[0][1], _max_vec[0][3]])
        _min_vec = np.min(filtered_binning, axis=0)
        min_vpos = np.min([_min_vec[0][1], _min_vec[0][3]])

        step = 5
        vpos = min_vpos
        scanned_middle_points = []
        scanned_lane_width = []
        while vpos < max_vpos:
            holder = []
            for line in filtered_binning:
                if (vpos - line[0][1]) * (vpos - line[0][3]) < 0:
                    holder.append([self.get_horizontal_pos(vpos, line), vpos, ])
                if len(holder) >= 2:
                    break
            if len(holder) >= 2:
                width = abs(holder[0][0] - holder[1][0])
                if width > 2:
                    scanned_middle_points.append(np.average(holder, axis=0))
                    scanned_lane_width.append(width)
            vpos += step

        return scanned_middle_points, scanned_lane_width

    def get_horizontal_pos(self, vpos, line):
        k = (line[0][1] - vpos) / (vpos - line[0][3])
        return (line[0][0] + k * line[0][2]) / (1 + k)

    def fit_line(self, points, widths, thickiria=0.6, vmax=539, vmin=330):
        thickness = widths[-1] * thickiria
        _line = np.array([np.concatenate([points[0], points[-1]])])
        #vmin = min(vmin, points[0][1])
        #vmax = max(vmax, points[1][1])
        hmax = self.get_horizontal_pos(vmax, _line)
        hmin = self.get_horizontal_pos(vmin, _line)
        expanded_line = np.array([list(map(int, [hmax, vmax, hmin, vmin]))])
        return expanded_line, 13#thickness

    def pipeline(self, img):
        copy = np.copy(img) * 0

        gray_image = grayscale(img)

        kernel_size = 5
        blurred_image = gaussian_blur(gray_image, kernel_size)

        low_threshold = 100
        high_threshold = 200
        edges = cv2.Canny(blurred_image, low_threshold, high_threshold)

        vertices = np.array([[(0, 540), (960, 540), (480, 330)]], np.int32)
        masked_image = region_of_interest(edges, vertices)

        rho = 1
        theta = np.pi / 135
        threshold = 1
        min_line_len = 10
        max_line_gap = 4
        hough_lines_image, lines = hough_lines(
            masked_image, rho, theta, threshold, min_line_len, max_line_gap)

        hough_image = weighted_img(hough_lines_image, img)
        return hough_image
        #cv2.imshow("hough", blended_image)
        #cv2.waitKey(-1)

        binnings, slopes = self.slope_binning(lines)

        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for b, s in zip(binnings, slopes):
            middle_points, lane_widths = self.scan_binning(
                b, s, delta_slope_err=0.1)
            if len(lane_widths) <= 1:
                continue
            line, thickness = self.fit_line(middle_points, lane_widths)
            draw_lines(line_img, [line, ], thickness=int(thickness))

        blended_image = weighted_img(line_img, img)
        #cv2.imshow("mine", blended_image)
        #cv2.waitKey(-1)
        return hough_image


    def run_image_dir(self):
        for image_name in os.listdir(self.input_dir):
            if image_name.startswith("output_"):
                continue
            #if image_name != "solidYellowCurve2.jpg":
            #    continue

            image = self.load_image(image_name)

            output_image = self.pipeline(image)

            self.save_image("output_%s" % image_name, output_image)


    def run_image(self, image):
        output_image = self.pipeline(image)
        return output_image


def test():
    lf = LaneFinding("test_images")
    lf.run_image_dir()

test()
