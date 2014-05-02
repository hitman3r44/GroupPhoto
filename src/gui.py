import os
from cv2 import namedWindow, moveWindow, setMouseCallback, waitKey, EVENT_LBUTTONUP, EVENT_RBUTTONUP, imread
from ydw_cv import *
from PyQt4.QtGui import QMainWindow, QFileDialog, QStandardItem, QStandardItemModel, QPixmap, QImage
from PyQt4 import QtCore
from PyQt4.QtCore import QObject
from main_win import Ui_main_win
from helper import *

PLACE_HOLDER = 'res/placeholder.png'


class GroupPhotoWindow(QMainWindow, Ui_main_win):

    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)

        self.image_model = QStandardItemModel()
        self.image_list.setModel(self.image_model)

        QObject.connect(self.add_btn, QtCore.SIGNAL('clicked()'), self.on_add_image)
        QObject.connect(self.delete_btn, QtCore.SIGNAL('clicked()'), self.on_remove_image)
        QObject.connect(self.image_list.selectionModel(), QtCore.SIGNAL(
            'currentChanged (const QModelIndex &, const QModelIndex &)'), self.on_row_changed)
        QObject.connect(self.harris_btn, QtCore.SIGNAL('clicked()'), self.on_harris_detect)
        QObject.connect(self.stitch_btn, QtCore.SIGNAL('clicked()'), self.on_stitch)

    def on_add_image(self):
        image_path = QFileDialog.getOpenFileName(self, 'Select image', filter='Images (*.png *.jpg *.bmp)')
        ignored, name = os.path.split(unicode(image_path))
        self.image_model.appendRow([QStandardItem(name), QStandardItem(image_path)])

    def on_remove_image(self):
        row = self.image_list.selectionModel().currentIndex().row()
        if row == -1:
            return
        self.image_model.removeRows(row, 1)

    def on_row_changed(self, current, previous):
        row = current.row()
        if row == -1:
            file_path = PLACE_HOLDER
        else:
            file_path = self.image_model.item(row, 1).text()
        self.set_pixmap(QPixmap(file_path))

    def on_harris_detect(self):
        row = self.image_list.selectionModel().currentIndex().row()
        if row == -1:
            return
        file_path = self.image_model.item(row, 1).text()

        image_pfm = convert_to_pfm(imread(unicode(file_path)))
        image = image_to_gray(image_pfm)
        kernel = self.ksize_slider.value()
        cornerness = harris_corner(image, ksize=kernel)
        is_corner = cornerness > 0.000001
        suppress = non_max_suppression(cornerness, ksize=kernel)
        result = np.array(is_corner * suppress)
        zero = np.zeros_like(result)
        gray_img = np.dstack((zero, zero, draw_cross(result)))

        self.set_pixmap(QPixmap.fromImage(image_from_ndarray(np.array(gray_img + image_pfm))))

    def on_stitch(self):
        i = 0
        images = []
        while self.image_model.item(i):
            file_path = self.image_model.item(i, 1).text()
            image = image_to_gray(convert_to_pfm(imread(unicode(file_path))))
            images.append(image)
            i += 1
        stitch(self.ksize_slider.value(), images)

    def set_pixmap(self, pixmap):
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio))


def image_from_ndarray(mat):
    scaled = mat * 256
    [h, w, d] = mat.shape
    clipped = np.clip(scaled, 0, 255)
    a = np.array(clipped, np.uint8)
    bgra = np.empty((h, w, 4), np.uint8)
    bgra[..., 0] = a[..., 0]
    bgra[..., 1] = a[..., 1]
    bgra[..., 2] = a[..., 2]
    bgra[..., 3].fill(255)
    return QImage(bgra.data, w, h, QImage.Format_RGB32)
