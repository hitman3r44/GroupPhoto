import cv2
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from math import ceil, exp, pi, cos, sin
from scipy.ndimage import convolve1d, gaussian_filter
from sklearn.neighbors import NearestNeighbors
from helper import draw_cross

LINEAR = 'ydwCv_linear'
NEAREST = 'ydwCv_nearst'
KEEP_RATIO = 'ydwCv_keep_ratio'
CORRELATION = 'ydwCv_correlation'
CONVOLUTION = 'ydwCv_convoluton'

GRAD_KERN = np.array([1, 0, -1], dtype=np.float)
LAPLACIAN_KERN = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float)


# def resize(mat, width=0, height=0, flag=KEEP_RATIO, method=NEAREST):
#     assert(width > 0 and height > 0)
#     w, h, d = mat.shape
#     print mat.shape
#     if flag == KEEP_RATIO:
#         factor = min(width / float(w), height / float(h))

#     ret = np.ndarray(mat.shape)

#     indices = range(0, mat.size-1)
#     id_x = map(lambda ind: ind / (h * d), indices)
# id_y = map(lambda ind: (ind / d) % w, indices)
# id_z = map(lambda ind: ind % d, indices)
#     print id_x
# print id_y
# print id_z

#     return ret


def convert_to_pfm(mat):
    return mat / 256.0


def draw_lines_on_mat(vertices, mat, polygon=False, lineType=cv2.CV_AA):
    ret = np.copy(mat)
    for i in range(len(vertices) - 1):
        cv2.line(ret, vertices[i], vertices[i + 1], (1, 1, 1), lineType=lineType)

    if polygon:
        cv2.line(ret, vertices[-1], vertices[0], (1, 1, 1), lineType=lineType)

    return ret


def bounding_rect(points):
    (minx, miny) = reduce(lambda (x1, y1), (x2, y2): (min(x1, x2), min(y1, y2)), points)
    (maxx, maxy) = reduce(lambda (x1, y1), (x2, y2): (max(x1, x2), max(y1, y2)), points)
    return ((minx, miny), (maxx + 1, maxy + 1))


def range_of_interest(mat, rect):
    return mat[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]


def in_range(pt, rect):
    return pt[0] >= rect[0][0] and pt[0] < rect[1][0] and pt[1] >= rect[0][1] and pt[1] < rect[1][1]


def flood_fill(mat, point, val, threshold=1, seeds=None):
    [h, w] = mat.shape[0:2]
    if seeds is None:
        seeds = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    rect = ((0, 0), (h, w))
    queue = list()
    queue += seeds
    while len(queue) > 0:
        pt = queue.pop()
        if mat[pt].all() >= threshold:
            continue
        mat[pt] = val
        tmp = neighbor(pt)
        tmp = filter(lambda x: in_range(x, rect), tmp)
        queue += tmp


def gen_mask_from_polygon(src, track):
    [h, w, d] = src.shape
    mask = np.zeros((h, w, d), np.float)
    mask = draw_lines_on_mat(track, mask, polygon=True, lineType=8)
    flood_fill(mask, track, (1.0, 1.0, 1.0), threshold=1)
    return mask


def paste_mat(image, paste, mask, pos):
    [h, w, d] = paste.shape
    x = pos[0] - w / 2
    y = pos[1] - h / 2

    ret = np.copy(image)
    ret[y:y + h, x:x + w] = ret[y:y + h, x:x + w] * mask + paste * (1 - mask)
    return ret


def resize(image, ratio, method=LINEAR):
    (h, w, d) = image.shape
    h = int(ceil(h * ratio))
    w = int(ceil(w * ratio))
    if method == LINEAR:
        ret = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    elif method == NEAREST:
        ret = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        assert False, 'Unrecognized method {}'.format(method)

    return ret


def neighbor(pt):
    (i, j) = pt
    ret = []
    ret.append((i + 1, j))
    ret.append((i - 1, j))
    ret.append((i, j - 1))
    ret.append((i, j + 1))
    return ret


def laplacian(mat):
    [h, w, d] = mat.shape
    ret = mat.copy()
    for i in range(0, h):
        for j in range(0, w):
            tmp = filter(lambda pt: in_range(pt, ((0, 0), (h, w))), neighbor((i, j)))
            minus = sum(mat[p] for p in tmp)
            ret[i, j] = mat[i, j] * len(tmp) - minus

    return ret


def poisson_blending(image, paste, mask, pos, alpha=1.0):
    [h, w, d] = paste.shape
    ret = image.copy()
    x = pos[0] - w / 2
    y = pos[1] - h / 2

    new_mask = 1.0 - mask[:, :, 0]
    make_border(new_mask)
    point_to_ind, ind_to_point = compact_mask(new_mask)
    nPts = len(ind_to_point)

    lap_of_paste = laplacian(paste)
    lap_of_image = laplacian(image[y:y + h, x:x + w])
    grad_mixed = alpha * lap_of_paste + (1 - alpha) * lap_of_image
    # construct equation
    for depth in range(0, d):
        vec_b = np.ndarray([nPts])
        mat_a = scipy.sparse.dok_matrix((nPts, nPts))
        for i in range(0, nPts):
            pt = ind_to_point[i]
            mat_a[i, i] = 4
            vec_b[i] = grad_mixed[pt][depth]
            tmp = neighbor(pt)
            for px in tmp:
                if px in point_to_ind.keys():  # inferior
                    mat_a[i, point_to_ind[px]] = -1
                else:                   # is inferior
                    vec_b[i] = vec_b[i] + image[px[0] + y, px[1] + x][depth]

        # solve poisson equation
        vecX = scipy.sparse.linalg.spsolve(mat_a.tocsc(), vec_b)
        for i, pt in enumerate(ind_to_point):
            ret[y + pt[0], x + pt[1], depth] = vecX[i]

    return ret

    # return paste_mat(image, paste, mask, pos)


def compact_mask(mask):
    point_to_ind = dict()
    ind_to_point = list()
    [h, w] = mask.shape

    for i in range(0, h):
        for j in range(0, w):
            if mask[i, j] == 1:
                point_to_ind[(i, j)] = len(point_to_ind)
                ind_to_point.append((i, j))

    return point_to_ind, ind_to_point


def make_border(mask):
    [h, w] = mask.shape
    for i in range(0, h):
        for j in range(0, w):
            if mask[i, j] > 0:
                continue
            tmp = neighbor((i, j))
            tmp = filter(lambda x: in_range(x, ((0, 0), (h, w))), tmp)
            tmp = filter(lambda x: mask[x] == 1, tmp)
            if len(tmp) > 0:
                mask[i, j] = 0.5


# Efficient Non-maximum Suppression
def eff_non_max_suppression(mat, ksize):
    assert len(mat.shape) == 2, 'Matrix must be 2d array.'
    ret = np.zeros(mat.shape, dtype=np.bool)
    [h, w] = mat.shape
    n = ksize

    for i in range(n, h - n, n + 1):
        for j in range(n, w - n, n + 1):
            mi, mj = i, j
            for i2 in range(i, i + n + 1):
                for j2 in range(j, j + n + 1):
                    if mat[i2][j2] > mat[mi][mj]:
                        mi, mj = i2, j2

            failed = False
            for i2 in range(mi - n, min(mi + n + 1, h)):
                for j2 in range(mj - n, min(mj + n + 1, w)):
                    if i2 >= i and i2 <= i + n and j2 >= j and j2 <= j + n:
                        continue
                    if mat[i2][j2] > mat[mi][mj]:
                        # print (i2, j2), '>', (mi, mj)
                        failed = True
                        break
                if failed:
                    break
            if not failed:
                ret[mi][mj] = True

    return ret


def slow_max_filter(mat, ksize):
    assert len(mat.shape) == 2, 'Matrix must be 2d array'
    ret = np.ndarray(mat.shape, dtype=np.float)
    [h, w] = mat.shape
    for i in range(h):
        for j in range(w):
            max_val = mat[i, j]
            for ii in range(max(0, i - ksize), min(h, i + ksize + 1)):
                for jj in range(max(0, j - ksize), min(w, j + ksize + 1)):
                    max_val = max(max_val, mat[ii, jj])
            ret[i][j] = max_val
    return ret


def slow_non_max_suppression(mat, ksize=3):
    return mat == slow_max_filter(mat, ksize)


def non_max_suppression(mat, ksize=3):
    # return slow_max_filter(mat, ksize=ksize) == mat
    return eff_non_max_suppression(mat, ksize=ksize)


def harris_corner(image, ksize=3, kappa=0.04):
    gray = image_to_gray(image)
    i_x, i_y = gradient(gray)
    mat_a = i_x * i_x
    mat_b = i_x * i_y
    mat_c = i_y * i_y
    mat_a = gaussian_filter(mat_a, ksize)
    mat_b = gaussian_filter(mat_b, ksize)
    mat_c = gaussian_filter(mat_c, ksize)
    a_add_c = mat_a + mat_c
    corn_factor = mat_a * mat_c - mat_b * mat_b - kappa * a_add_c * a_add_c
    return corn_factor


def image_to_gray(image):
    dim = len(image.shape)
    if dim <= 2:
        return image
    tt = 1
    for i in range(2, dim):
        tt *= image.shape[i]

    return np.sum(image, axis=tuple(range(2, dim))) / tt


def gradient(mat):
    return (convolve1d(mat, GRAD_KERN, axis=1, mode="nearest"), convolve1d(mat, GRAD_KERN, axis=0, mode="nearest"))


def gaussian(height, width, center, sigma):
    mat = np.ndarray((height, width), np.float)
    for i in range(height):
        for j in range(width):
            x = i - center[0]
            y = j - center[1]
            mat[i][j] = exp((-x * x - y * y) / (2.0 * sigma * sigma))
    return mat


def sift_descriptor(img, pt, ksize):
    roi = img[pt[0] - ksize:pt[0] + ksize + 1, pt[1] - ksize:pt[1] + ksize + 1]
    h, w = roi.shape
    center = (min(ksize, pt[0]), min(ksize, pt[1]))

    gx, gy = gradient(roi)
    mag = (np.sqrt(gx * gx + gy * gy) * gaussian(h, w, center, 1.5 * ksize)).flatten()

    ang = (np.arctan2(gy, gx) / pi * 180).flatten()
    indices = np.searchsorted(np.arange(-175, 185, 10), ang)

    hist = np.zeros((36))
    for i in range(indices.shape[0]):
        hist[indices[i] % 36] += mag[i]

    # print indices
    # print roi
    # print gx
    # print gy
    # print (np.sqrt(gx * gx + gy * gy) * gaussian(h, w, center, 1.5 * ksize))
    # print (np.arctan2(gy, gx) / pi * 180)
    # print hist

    max_ind = 0
    max_ind2 = 1
    max_val = hist[0]
    max_val2 = hist[1]
    if max_val < max_val2:
        max_val, max_val2 = max_val2, max_val
        max_ind, max_ind2 = max_ind2, max_ind
    for i in range(2, hist.shape[0]):
        if max_val < hist[i]:
            max_val, max_val2 = hist[i], max_val
            max_ind, max_ind2 = i, max_ind
        elif max_val2 < hist[i]:
            max_val2 = hist[i]
            max_ind2 = i

    convert = lambda x: (x - 18) * pi / 18
    directions = tuple()

    if max_val * 0.8 < max_val2:
        directions = (convert(max_ind), convert(max_ind2))
        lengths = (hist[max_ind], hist[max_ind2])
    else:
        directions = (convert(max_ind),)
        lengths = (hist[max_ind],)

    # return directions, lengths
    # print directions, lengths

    descriptors = np.ndarray((len(directions), 128), np.float)
    for seq, theta in enumerate(directions):
        block = gen_block(img, pt, theta)
        descriptor = np.zeros((128), np.float)
        idx = 0
        for i in range(0, 16, 4):
            for j in range(0, 16, 4):
                roi = block[i:i + 4, j:j + 4]
                gx, gy = gradient(roi)
                mag = (np.sqrt(gx * gx + gy * gy)).flatten()
                ang = (np.arctan2(gy, gx) / pi * 180).flatten()
                indices = np.searchsorted(np.arange(-180 + 22.5, 180 + 22.5, 45), ang)

                hist = np.zeros(8)
                for k in range(indices.shape[0]):
                    hist[indices[k] % 8] += mag[k]

                descriptor[idx:idx + 8] = hist[:]
                idx += 8
        assert idx == 128, 'Error!'
        descriptor = descriptor / np.linalg.norm(descriptor)
        descriptors[seq, :] = descriptor[:]

    return descriptors


def gen_block(img, pt, theta):
    ret = np.ndarray((16,) * 2, np.float)
    for i in range(16):
        for j in range(16):
            y = i - 7.5
            x = j - 7.5
            ct = cos(theta)
            st = sin(theta)
            xr = x * ct - y * st
            yr = x * st + y * ct
            ret[i][j] = texture(img, yr + pt[0], xr + pt[1])
    return ret


def texture(mat, row, col):
    '''
    Bilinear interpolation of grayscale 2d image.
    '''
    assert len(mat.shape) == 2, 'texture only support 2d-matrix!'
    [h, w] = mat.shape
    if row < 0 or col < 0 or row > h - 1 or col > w - 1:
        return 0.0
    else:
        r = int(row)
        c = int(col)
        rb = row - r
        cb = col - c
        ra = 1 - rb
        ca = 1 - cb
        return mat[r][c] * ra * ca + mat[r][c + 1] * ra * cb + mat[r + 1][c] * rb * ca + mat[r + 1][c + 1] * rb * cb


def stitch(ksize, images):
    assert len(images) == 2, 'stitch() only supporet two images!'

    desc_of_imgs = []
    indc_of_imgs = []
    for img in images:
        # Detect harris corner
        cornerness = harris_corner(img, ksize)
        is_corner = cornerness > 0.001
        suppress = non_max_suppression(cornerness, ksize=ksize)
        result = np.array(is_corner * suppress)
        indices = np.transpose(np.where(result))

        desc = np.ndarray((0, 128), np.float)
        indc = np.ndarray((0, 2), np.integer)
        # Extract the feature
        for ind in indices:
            descriptor = sift_descriptor(img, ind, 5 * ksize)
            desc = np.vstack((desc, descriptor))
            indc = np.vstack((indc, np.tile(ind, (2, 1))))

        desc_of_imgs.append(desc)
        indc_of_imgs.append(indc)
        print desc.shape
        print indices.shape

    desc1, desc2 = desc_of_imgs[0], desc_of_imgs[1]
    indc1, indc2 = indc_of_imgs[0], indc_of_imgs[1]
    nbrs = NearestNeighbors(n_neighbors=2).fit(desc2)

    dists, inds = nbrs.kneighbors(desc1)
    result1 = np.zeros_like(images[0])
    result2 = np.zeros_like(images[1])
    for i in range(desc1.shape[0]):
        if dists[i][0] / dists[i][1] < 0.6:
            result1[indc1[i][0], indc1[i][1]] = 1
            result2[indc2[inds[i][0]][0], indc2[inds[i][0]][1]] = 1
            # print dists[i], indc1[i], indc2[inds[i][0]]

    ptnum = 5

    dist_ind = np.argsort(dists[:, 0])
    ptx = [indc1[dist_ind[i]] for i in range(ptnum)]
    pty = [indc2[inds[dist_ind[i]][0]] for i in range(ptnum)]

    print ptx
    print pty
    print dists[dist_ind[0:3]]

    mat_a = np.zeros((2 * ptnum, 6), np.float)
    vec_b = np.zeros((2 * ptnum), np.float)
    for i in range(ptnum):
        mat_a[2 * i][0] = mat_a[2 * i + 1][2] = pty[i][0]
        mat_a[2 * i][1] = mat_a[2 * i + 1][3] = pty[i][1]
        mat_a[2 * i][4] = mat_a[2 * i + 1][5] = 1
        vec_b[2 * i] = ptx[i][0]
        vec_b[2 * i + 1] = ptx[i][1]
    vec_x = np.linalg.lstsq(mat_a, vec_b)[0]

    mat_m = np.zeros((2, 2), np.float)
    vec_t = np.zeros((2), np.float)
    mat_m[0][0] = vec_x[0]
    mat_m[0][1] = vec_x[1]
    mat_m[1][0] = vec_x[2]
    mat_m[1][1] = vec_x[3]
    vec_t[0] = vec_x[4]
    vec_t[1] = vec_x[5]

    # Debug

    # zero1, zero2 = np.zeros_like(result1), np.zeros_like(result2)
    result1 = np.zeros_like(images[0])
    result2 = np.zeros_like(images[1])

    for i in range(ptnum):
        result1[ptx[i][0], ptx[i][1]] = 1
        result2[pty[i][0], pty[i][1]] = 1

    gray_img1 = np.dstack((images[0], images[0], draw_cross(result1) + images[0]))
    gray_img2 = np.dstack((images[1], images[1], draw_cross(result2) + images[1]))
    cv2.imshow('result1', gray_img1)
    cv2.imshow('result2', gray_img2)
    cv2.waitKey()

    return mat_m, vec_t


def affine(img, mat_m, vec_t):
    h, w = img.shape
    pt0 = np.array([0, 0], dtype=np.float)
    pt1 = np.array([h - 1, 0], dtype=np.float)
    pt2 = np.array([0, w - 1], dtype=np.float)
    pt3 = np.array([h - 1, w - 1], dtype=np.float)
    pt0_t, pt1_t, pt2_t, pt3_t = mat_m.dot(pt0) + vec_t, mat_m.dot(
        pt1) + vec_t, mat_m.dot(pt2) + vec_t, mat_m.dot(pt3) + vec_t

    # print mat_m
    # print vec_t

    # print '======'
    # print pt0, pt1, pt2, pt3
    # print '======'
    # print pt0_t, pt1_t, pt2_t, pt3_t
    # print '======'

    minx, miny = min((pt0_t[0], pt1_t[0], pt2_t[0], pt3_t[0])), min(pt0_t[1], pt1_t[1], pt2_t[1], pt3_t[1])
    maxx, maxy = max((pt0_t[0], pt1_t[0], pt2_t[0], pt3_t[0])), max(pt0_t[1], pt1_t[1], pt2_t[1], pt3_t[1])

    # print minx, miny, maxx, maxy
    hh, ww = int(ceil(maxx - minx)), int(ceil(maxy - miny))

    # print hh, ww

    ret = np.ndarray((hh, ww), np.float)

    mat_m_r = np.linalg.inv(mat_m)
    vec_t_r = mat_m_r.dot(-vec_t)

    for i in range(hh):
        for j in range(ww):
            rij = mat_m_r.dot(np.asarray([i + minx, j + miny])) + vec_t_r
            ret[i][j] = texture(img, rij[0], rij[1])

    return ret, [minx, miny]
