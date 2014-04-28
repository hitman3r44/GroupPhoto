import unittest
import sys
from cv2 import *

sys.path.append('../src')

from ydw_cv import *
from math import cos, sin, pi


class YdwCvTest(unittest.TestCase):

    def setUp(self):
        self.image = image_to_gray(convert_to_pfm(imread('../res/input2_compressed.jpg')))

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
        print slow_max_filter(image, ksize=1)

    def test_gaussian(self):
        print gaussian(5, 3)

    def test_sift_descriptor(self):
        # self.image = np.zeros((100, 100), np.float)
        # for i in range(25, 76):
        #     for j in range(25, 76):
        #         self.image[i][j] = 1.0

        ksize = 1
        cornerness = harris_corner(self.image, ksize)
        is_corner = cornerness > 0.000001
        suppress = non_max_suppression(cornerness, ksize=ksize)
        # imshow('image with non_max_suppression', np.array(is_corner * suppress) + self.image)
        # waitKey()
        result = np.array(is_corner * suppress)
        indices = np.transpose(np.where(result))

        # '''
        # new_image = self.image.copy()
        for ind in indices:
            descriptor = sift_descriptor(self.image, ind, ksize * 5)
            # for direct in direction:
                # draw_direction(new_image, ind, direct)
        # imwrite('output2.png', new_image * 256)
        # '''
        # direction = sift_descriptor(self.image, indices[0], 3)

    def test_texture(self):
        new_image = np.zeros((800, 800), np.float)
        h, w = self.image.shape
        for i in range(800):
            for j in range(800):
                new_image[i][j] = texture(self.image, (i + 8000) / 16000.0 * h, (j + 4000) / 8000.0 * w)

        imshow('scaled', new_image)
        waitKey()

    def test_gen_block(self):
        mat = np.zeros((100, 100))
        for i in range(100):
            for j in range(100):
                mat[i][j] = i + j
        print gen_block(mat, (50, 50), -pi / 4.0)

    def tearDown(self):
        pass


def draw_direction(mat, pos, direct):
    h, w = mat.shape
    r, c = pos
    radius = 16
    if radius > r:
        radius = r
    if radius > c:
        radius = c
    if radius > abs(h - r):
        radius = abs(h - r)
    if radius > abs(w - c):
        radius = abs(w - c)

    dx, dy = cos(direct), sin(direct)
    mat[r][c] = 0.5
    if r > 1 and c > 1:
        mat[r - 1][c - 1] = 0.5
    if r < h - 1 and c < w - 1:
        mat[r + 1][c + 1] = 0.5
    if r > 1 and c < w - 1:
        mat[r - 1][c + 1] = 0.5
    if r < h - 1 and c > 1:
        mat[r + 1][c - 1] = 0.5
    for t in range(radius):
        mat[int(r + dy * t)][int(c + dx * t)] = 1
