# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\Users\bo.chen18\pyproject\.vscode\main.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(341, 331)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(253, 180, 75, 24))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(172, 180, 75, 24))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(91, 180, 75, 24))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(161, 280, 75, 24))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(80, 280, 75, 24))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(10, 180, 75, 24))
        self.pushButton.setObjectName("pushButton")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(20, 70, 320, 26))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.PV = QtWidgets.QPushButton(self.widget)
        self.PV.setObjectName("PV")
        self.horizontalLayout.addWidget(self.PV)
        self.Load = QtWidgets.QPushButton(self.widget)
        self.Load.setObjectName("Load")
        self.horizontalLayout.addWidget(self.Load)
        self.Price = QtWidgets.QPushButton(self.widget)
        self.Price.setObjectName("Price")
        self.horizontalLayout.addWidget(self.Price)
        self.Wind = QtWidgets.QPushButton(self.widget)
        self.Wind.setObjectName("Wind")
        self.horizontalLayout.addWidget(self.Wind)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 341, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_2.setText(_translate("MainWindow", "Model1"))
        self.pushButton_3.setText(_translate("MainWindow", "Model2"))
        self.pushButton_4.setText(_translate("MainWindow", "Model3"))
        self.pushButton_5.setText(_translate("MainWindow", "Open result"))
        self.pushButton_6.setText(_translate("MainWindow", "View"))
        self.pushButton.setText(_translate("MainWindow", "Model4"))
        self.PV.setText(_translate("MainWindow", "PV"))
        self.Load.setText(_translate("MainWindow", "Load"))
        self.Price.setText(_translate("MainWindow", "Price"))
        self.Wind.setText(_translate("MainWindow", "Wind"))
