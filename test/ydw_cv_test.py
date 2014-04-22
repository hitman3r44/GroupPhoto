import unittest
import sys
sys.path.append('../src')
from cv2 import *
from ydw_cv import *
from os import *


class YdwCvTest(unittest.TestCase):

    def setUp(self):
        self.image = image_to_gray(convert_to_pfm(imread('../res/screenshot.png')))

    def test_gradient(self):
        namedWindow('testx')
        namedWindow('testy')
        dx, dy = gradient(self.image)
        imshow('testx', dx)
        imshow('testy', dy)
        waitKey()

    def test_harris_corner(self):
        kernel = 2
        harris = harris_corner(self.image, ksize=kernel, kappa=0.04)
        suppress = non_max_suppression(harris, ksize=kernel)
        # is_corner = harris * suppress > 0.0001
        is_corner = harris > 0.0001
        imshow('prim', self.image)
        imshow('image without non_max_suppression', np.array(is_corner, np.float) + self.image)
        imshow('image with non_max_suppression', np.array(is_corner * suppress) + self.image)
        waitKey()

    def test_max_filter(self):
        image = np.array([[1, 2, 3], [2, 3, 1], [5, 2, 4]])
        print max_filter(image, ksize=1)

    def tearDown(self):
        pass
