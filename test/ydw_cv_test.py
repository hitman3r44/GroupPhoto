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
        # image = image_to_gray(convert_to_pfm(imread('../res/group_rotate28.jpg')))
        image = image_to_gray(convert_to_pfm(imread('../res/group1.png')))
        image2 = image_to_gray(convert_to_pfm(imread('../res/group1_r.png')))

        ksize = 2
        cornerness = harris_corner(image, ksize)
        is_corner = cornerness > 0.0001
        suppress = non_max_suppression(cornerness, ksize=ksize)
        # imshow('image with non_max_suppression', np.array(is_corner * suppress) + image)
        # waitKey()
        result = np.array(is_corner * suppress)
        indices = np.transpose(np.where(result))
        print 'indices[0]: {}'.format(indices[0])
        sift_descriptor(image, indices[0], ksize * 5)
        sift_descriptor(image2, (indices[0][1], indices[0][0]), ksize * 5)

        '''
        new_image = image.copy()
        for ind in indices:
            directions, lengths = sift_descriptor(image, ind, ksize * 5)
            for i, direct in enumerate(directions):
                draw_direction(new_image, ind, direct, lengths[i])
        imshow('result', new_image)
        imwrite('result2.jpg', new_image * 256)
        waitKey()
        '''

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

    def test_stitch(self):
        # image_color1 = convert_to_pfm(imread('../res/input1_compressed.jpg'))
        # image_color2 = convert_to_pfm(imread('../res/input2_compressed.jpg'))
        image_color1 = convert_to_pfm(imread('../res/group1.png'))
        image_color2 = convert_to_pfm(imread('../res/group1_r.png'))
        image1 = image_to_gray(image_color1)
        image2 = image_to_gray(image_color2)

        mat_m, vec_t = stitch(2, [image1, image2])

        image2_, shift = affine(image2, mat_m, vec_t)
        imshow('image2_trans', image2_)
        waitKey()

    def test_affine(self):
        image_color1 = convert_to_pfm(imread('../res/group1.jpg'))
        image1 = image_to_gray(image_color1)
        mat_m = np.ndarray((2, 2), np.float)
        mat_m[0][0], mat_m[1][0], mat_m[0][1], mat_m[1][1] = 0.2, 1.4, -0.3, 0.7
        vec_t = np.zeros((2), np.float)
        aff, pt = affine(image1, mat_m, vec_t)

        # print pt
        # print 'aff.shape: ', aff.shape
        imshow('aff', aff)
        waitKey()

    def tearDown(self):
        pass


def draw_direction(mat, pos, direct, length):
    h, w = mat.shape
    r, c = pos
    radius = int(4 * length)
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
