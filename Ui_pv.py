# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\Users\bo.chen18\pyproject\.vscode\pv.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_PVWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(604, 483)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.plotWidget = QtWidgets.QWidget(self.centralwidget)
        self.plotWidget.setGeometry(QtCore.QRect(40, 110, 501, 281))
        self.plotWidget.setObjectName("plotWidget")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(40, 410, 246, 26))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.miAnalysisButton = QtWidgets.QPushButton(self.layoutWidget)
        self.miAnalysisButton.setObjectName("miAnalysisButton")
        self.horizontalLayout_2.addWidget(self.miAnalysisButton)
        self.featureEngineeringButton = QtWidgets.QPushButton(self.layoutWidget)
        self.featureEngineeringButton.setObjectName("featureEngineeringButton")
        self.horizontalLayout_2.addWidget(self.featureEngineeringButton)
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(40, 60, 501, 31))
        self.textEdit.setObjectName("textEdit")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(40, 30, 246, 26))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.importDataButton = QtWidgets.QPushButton(self.widget)
        self.importDataButton.setObjectName("importDataButton")
        self.horizontalLayout.addWidget(self.importDataButton)
        self.showHeadButton = QtWidgets.QPushButton(self.widget)
        self.showHeadButton.setObjectName("showHeadButton")
        self.horizontalLayout.addWidget(self.showHeadButton)
        self.cleanDataButton = QtWidgets.QPushButton(self.widget)
        self.cleanDataButton.setObjectName("cleanDataButton")
        self.horizontalLayout.addWidget(self.cleanDataButton)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 604, 22))
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
        self.miAnalysisButton.setText(_translate("MainWindow", "MIA"))
        self.featureEngineeringButton.setText(_translate("MainWindow", "FeatureEngneer"))
        self.importDataButton.setText(_translate("MainWindow", "ImportData"))
        self.showHeadButton.setText(_translate("MainWindow", "ShowHead"))
        self.cleanDataButton.setText(_translate("MainWindow", "CleanData"))
