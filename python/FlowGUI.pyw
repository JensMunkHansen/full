import os, sys, multiprocessing

from PyQt4 import QtGui, QtCore
from PyQt4.uic import loadUiType
from PyQt4.QtCore import (QString, QRect,QObject,QRunnable, QThreadPool, Qt)
from PyQt4.QtCore import (pyqtSignal)
from PyQt4.Qt import (SIGNAL, SLOT)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)

import matplotlib.pyplot as plt

ui, QMainWindow = loadUiType('FlowGUI.ui')

class Main(QMainWindow, ui):
  def __init__(self, ):
    super(Main, self).__init__()
    self.setupUi(self)

if __name__ == '__main__':

  fig1 = Figure()
  #ax1 = fig1.add_subplot(111)

  app = QtGui.QApplication(sys.argv)
  app.setStyle(QtGui.QStyleFactory.create('Fusion'))
  main = Main()
  #main.addmpl(fig)
  #main.addfield(fig1)
  main.show()
  sys.exit(app.exec_())


# Local variables: #
# tab-width: 2 #
# python-indent: 2 #
# indent-tabs-mode: nil #
# End: #
