# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 00:25:40 2023

@author: Stan
"""

import sys  # sys нужен для передачи argv в QApplication
import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import pathlib
from pathlib import Path
import ast
from glob import glob
from PyQt5 import QtWidgets
from PyQt5.QtCore import QRect, QEvent, QUrl, QPoint, Qt
from PyQt5.QtGui import QIcon, QPixmap, QPicture, QColor, QImage
from PyQt5.QtWidgets import (QMainWindow, QTextEdit, QAction, QFileDialog, QApplication, QMessageBox, QPushButton, QLabel, QFrame, QMenu, QStatusBar, QGridLayout)
import EndoDesign  # Это наш конвертированный файл дизайна

class ExampleApp(QtWidgets.QMainWindow, EndoDesign.Ui_MainWindow):
    def __init__(self):
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. в файле EndoDesign.py
        super().__init__()
        self.setupUi(self)  # Это нужно для инициализации нашего дизайна
        self.openAction.triggered.connect(self.openFile)
        #self.saveAction.triggered.connect("")
        self.exitAction.triggered.connect(self.onExit)
        self.loadBttn.clicked.connect(self.loadImage)
        
        #Название
        title = 'Endo Ultra Digital Classifier v23'
        self.setWindowTitle(title)
        
        #Заставка
        pixmap = QPixmap('C:/Users/User/Desktop/Endo/logo/EUDC.png')
        self.imgLbl.setPixmap(pixmap)
        
        self.work_dir = os.getcwd() # Получение рабочей директории через метод getcwd
        self.WidgetMes = QLabel("") # Создание виджета-сообщения для дальнейшего использования в статус-баре
        self.nf = np.float32
        self.objList = []
        

    def openFile(self):
        fname = QFileDialog.getOpenFileName(self, "Открыть файл", self.work_dir,
                                            "Графика (*.jpg *.jpeg *.bmp *.png *.tiff)\n Все файлы (*.*)")[0]
        if (fname == ""):
            return
        self.file_name = fname
        self.openImageFile()
        
        # Статусбар с именем файла
        url = QUrl.fromLocalFile(fname)
        self.WidgetMes.setText("Файл: {}".format(url.fileName()))
        self.statusbar.addPermanentWidget(self.WidgetMes)
        
    def openImageFile(self):
        if (self.file_name == ""):
            return
        self.imgLbl.setPixmap(QPixmap(self.file_name))
        
        #self.perValue.setText("000%")
        self.objName.setText("")
        
        
    def loadImage(self):
        path = self.file_name
        files = sorted(glob(str(path)))
        for file_name in files:
            img_load = cv.imread(file_name)
            height = 86
            width = 124
            dim = (width, height)
            img_load = cv.resize(img_load, dim, interpolation = cv.INTER_AREA)
            img_load = ((img_load - img_load.min())/(img_load.max() - img_load.min()))
            img_load.astype('float32')
            self.objList.append(img_load)
            self.proModel(self.objList)
        
    def proModel(self, objList):
        dir_path = Path(pathlib.Path.cwd(), 'Endo.h5')
        #print(dir_path)
        model = tf.keras.models.load_model(dir_path)
        sample = tf.expand_dims(objList[-1], 0)
        sample = tf.convert_to_tensor(sample)
        pred = model.predict(sample)
        
        labels = ['Норма',                      # 0
                  'Гиперпластический полип',    # 1
                  'Аденоматозный полип',        # 2
                  'Аденокарцинома (рак)']       # 3
        class_names = labels
        
        score = tf.nn.softmax(pred[0])
        #self.percent = str(int(round(100*np.max(score), 0))) + "%"
        #self.perValue.setText(self.percent)
        self.class_names = str(class_names[np.argmax(score)])
        self.objName.setText(self.class_names)

    
    def onExit(self):
        self.close()
        
def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = ExampleApp()  # Создаём объект класса ExampleApp
    window.show()  # Показываем окно
    app.exec_()  # и запускаем приложение
    
if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()