#!/usr/bin/env python
from PyQt5.QtCore import QDir, Qt
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QLabel,
        QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy)
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter

import tensorflow as tf
from keras import backend as K
from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.engine import Layer
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate, Activation, Dense, Flatten
from keras.models import Model
from keras.layers.core import RepeatVector, Permute
from keras.layers.pooling import MaxPooling2D
from keras.models import model_from_json

from PIL import Image
from keras import layers
from keras.applications import DenseNet121
from keras.utils.data_utils import get_file
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam

import numpy as np
from skimage.transform import resize
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave, imread



class ImageViewer(QMainWindow):
    def __init__(self):
        super(ImageViewer, self).__init__()

        self.printer = QPrinter()
        self.scaleFactor = 0.0

        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)

        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.setCentralWidget(self.scrollArea)

        self.createActions()
        self.createMenus()

        self.setWindowTitle("Retinopathy")
        self.resize(500, 400)

    def open(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                QDir.currentPath())
        if fileName:
            image = QImage(fileName)
            if image.isNull():
                QMessageBox.information(self, "Retinopathy",
                        "Cannot load %s." % fileName)
                return

            self.imageLabel.setPixmap(QPixmap.fromImage(image))
            self.scaleFactor = 1.0

            self.printAct.setEnabled(True)
            self.fitToWindowAct.setEnabled(True)
            self.updateActions()

            if not self.fitToWindowAct.isChecked():
                self.imageLabel.adjustSize()
                
            DENSENET_121_WEIGHTS_PATH_NO_TOP = (r'https://github.com/titu1994/DenseNet/releases/'
                                                r'download/v3.0/DenseNet-BC-121-32-no-top.h5')
            densenet = DenseNet121(
                                    weights=get_file('DenseNet-BC-121-32-no-top.h5',
                                    DENSENET_121_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='55e62a6358af8a0af0eedf399b5aea99'),
                                    include_top=False,
                                    input_shape=(224,224,3)
                                    )
            def build_model():
                model = Sequential()
                model.add(densenet)
                model.add(layers.GlobalAveragePooling2D())
                model.add(layers.Dropout(0.5))
                model.add(layers.Dense(5, activation='sigmoid'))
                return model
            
            model=build_model()

            model.compile(
                loss='binary_crossentropy',
                optimizer=Adam(lr=0.00005),
                metrics=['accuracy']
            )

            model.load_weights('model.h5')
            load_model('model.h5')
            predict_me = []
            predict_me.append(img_to_array(load_img(fileName, target_size=(224,224))))
            predict_me = np.array(predict_me, dtype=float)
            result = model.predict(predict_me) > 0.5
            result = result.astype(int).sum(axis=1) - 1
            #i = image.load_img(fileName, target_size=(224,224))
            #i = np.expand_dims(i, axis = 0)
            #result = model.predict(i)
            if result == 4:
                self.setWindowTitle("Proliferative DR")
            elif result == 3:
                self.setWindowTitle("Severe")
            elif result == 2:
                self.setWindowTitle("Moderate")
            elif result == 1:
                self.setWindowTitle("Mild")
            else:
                self.setWindowTitle("No DR")

    def print_(self):
        dialog = QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QPainter(self.printer)
            rect = painter.viewport()
            size = self.imageLabel.pixmap().size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.imageLabel.pixmap().rect())
            painter.drawPixmap(0, 0, self.imageLabel.pixmap())

    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.imageLabel.adjustSize()
        self.scaleFactor = 1.0

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()

        self.updateActions()

    def about(self):
        QMessageBox.about(self, "About Retinopathy",
                "<p> <b>paypal.me/retinopathy</b> </p>"
                "<p>This program evaluated on the quadratic weighted kappa "
                "with an average over 80. "
                "Images have five possible ratings, 0,1,2,3,4. "
                "0 - No DR, 1 - Mild, 2 - Moderate, 3 - Severe, "
                "4 - Proliferative DR</p>"
                "<p>Diabetic retinopathy is the leading cause of  "
                "blindness among working aged adults. "
                "Hopefully this program will improve "
                "the ability to identify potential patients. "
                "<p>This piece was made by <b>Vadim Makarenkov</b></p>")

    def createActions(self):
        self.openAct = QAction("&Open...", self, shortcut="Ctrl+O",
                triggered=self.open)

        self.printAct = QAction("&Print...", self, shortcut="Ctrl+P",
                enabled=False, triggered=self.print_)

        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q",
                triggered=self.close)

        self.zoomInAct = QAction("Zoom &In (25%)", self, shortcut="Ctrl++",
                enabled=False, triggered=self.zoomIn)

        self.zoomOutAct = QAction("Zoom &Out (25%)", self, shortcut="Ctrl+-",
                enabled=False, triggered=self.zoomOut)

        self.normalSizeAct = QAction("&Normal Size", self, shortcut="Ctrl+S",
                enabled=False, triggered=self.normalSize)

        self.fitToWindowAct = QAction("&Fit to Window", self, enabled=False,
                checkable=True, shortcut="Ctrl+F", triggered=self.fitToWindow)

        self.aboutAct = QAction("&About", self, triggered=self.about)

        self.aboutQtAct = QAction("About &Qt", self,
                triggered=QApplication.instance().aboutQt)

    def createMenus(self):
        self.fileMenu = QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.printAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.viewMenu = QMenu("&View", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)

        self.helpMenu = QMenu("&Help", self)
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.helpMenu)

    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 3.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.333)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                                + ((factor - 1) * scrollBar.pageStep()/2)))


if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    imageViewer = ImageViewer()
    imageViewer.show()
    sys.exit(app.exec_())