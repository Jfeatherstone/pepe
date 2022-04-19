"""
Wizard to help user select parameters for various operations. WIP.
"""
import numpy as np
import sys

from PyQt5 import QtWidgets, QtCore, Qt, QtGui

class HelpWindow(QtWidgets.QMainWindow):

    def __init__(self, app):
        super(HelpWindow, self).__init__()
        
        #######################
        #    Basic setup
        #######################
        screen = app.primaryScreen()
        defaultWindowSize = screen.size() * .5
        self.resize(defaultWindowSize)
        #self.setWindowTitle('pepe Parameter Help')
        self.setWindowTitle('float')

        self.cWidget = QtWidgets.QWidget()
        self.vLayout = QtWidgets.QVBoxLayout()
        self.cWidget.setLayout(self.vLayout)

        self.titleLbl = QtWidgets.QLabel()
        self.titleLbl.setText('Howdy!\nWelcome to the pepe parameter selection wizard!')
        self.titleLbl.setFont(QtGui.QFont('Sans', 20))

        self.titleLbl.setAlignment(Qt.Qt.AlignCenter)
        self.vLayout.addWidget(self.titleLbl)

        self.setCentralWidget(self.cWidget)

        self.nextBtn = QtWidgets.QPushButton('Next')
        self.nextBtn.setMaximumWidth(75)
        self.cancelBtn = QtWidgets.QPushButton('Cancel')
        self.cancelBtn.setMaximumWidth(75)
        self.restartBtn = QtWidgets.QPushButton('Restart')
        self.restartBtn.setMaximumWidth(75)

        self.bottomHLayout = QtWidgets.QHBoxLayout()

        self.bottomHLayout.addWidget(self.cancelBtn)
        self.bottomHLayout.addWidget(self.restartBtn)
        self.bottomHLayout.addWidget(self.nextBtn)

        self.vLayout.addLayout(self.bottomHLayout)


        #######################
        #   Parameter Selection
        #######################
        # Here we decide which features we want to select parameters
        # for
        bundledParameters = {'Circle tracking': 'ctrack',
                             'Rotation tracking': 'rtrack',
                             'Masking': 'mask',
                             'Optimization': 'opt'}

        availableParameters = {}


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    window = HelpWindow(app)
    window.show()

    sys.exit(app.exec())
