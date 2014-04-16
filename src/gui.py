import os
from cv2 import namedWindow, imshow, moveWindow, setMouseCallback, waitKey, EVENT_LBUTTONUP, EVENT_RBUTTONUP, imread
from ydw_cv import *
from PyQt4.QtGui import QMainWindow, QFileDialog, QStandardItem, QStandardItemModel, QPixmap, QImage
from PyQt4 import QtCore
from PyQt4.QtCore import QObject
from main_win import Ui_main_win

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
        self.image_label.setPixmap(QPixmap(file_path))

    def on_harris_detect(self):
        row = self.image_list.selectionModel().currentIndex().row()
        if row == -1:
            return
        file_path = self.image_model.item(row, 1).text()

        image_pfm = convert_to_pfm(imread(unicode(file_path)))
        image = image_to_gray(image_pfm)
        kernel = self.ksize_slider.value()
        print('ksize: %d' % kernel)
        cornerness = harris_corner(image, ksize=kernel)
        is_corner = cornerness > 0.0001
        suppress = non_max_suppression(cornerness, ksize=kernel)
        result = np.array(is_corner * suppress)
        zero = np.zeros_like(result)
        gray_img = np.dstack((zero, zero, result))

        self.image_label.setPixmap(QPixmap.fromImage(imageFromNdArray(np.array(gray_img + image_pfm))))


def imageFromNdArray(mat):
    scaled = mat * 256
    [h, w, d] = mat.shape
    clipped = np.clip(scaled, 0, 255)
    a = np.array(clipped, np.uint8)
    imshow('test', a)
    print a[:, :, 0]
    print a[:, :, 0] << 16
    b = (255 << 24 | a[:, :, 0] << 16 | a[:, :, 1] << 8 | a[:, :, 2]).flatten()
    return QImage(b, w, h, QImage.Format_RGB32)
