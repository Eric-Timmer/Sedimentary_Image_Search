import sys
import os
from PyQt5 import QtWidgets, QtCore, QtGui
import glob
import corePhotoSearch
import pandas as pd
import numpy as np

class ShowImages(QtCore.QAbstractListModel):
    def __init__(self, f_list, parent):
        QtCore.QAbstractListModel.__init__(self, parent)
        self.f_list = f_list
        QtCore.QModelIndex()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.f_list)

    def data(self, index, role):
            if index.isValid() and role == QtCore.Qt.DecorationRole:
                im = QtGui.QIcon(QtGui.QPixmap(self.f_list[index.row()]))
                return im


class CorePhotoSearch(QtWidgets.QListView):
    def __init__(self):
        super().__init__()
        self.user_interface()



        # add data
        # self.similarity_matrix = pd.read_csv("similarity_matrix.csv", index_col=0)
        self.f_list = glob.glob("out/*/*.jpg")[0:50]

        image_model = ShowImages(self.f_list, self)
        self.setModel(image_model)
        self.show()


    def user_interface(self):
        # window parameters
        self.showFullScreen()
        self.setWindowTitle("CorePhotoSearch")

        # set QListView to IconMode
        self.setViewMode(self.IconMode)
        # rescale images dynamically
        self.setResizeMode(1)

        #load images in batch
        self.LayoutMode(1)
        self.setBatchSize(50)

        # wrap images onto screen
        self.setWrapping(True)

        # set image and grid size ( TODO allow the user to change this size dynamically)
        self.setIconSize(QtCore.QSize(300, 300))
        self.setGridSize(QtCore.QSize(305, 305))
        self.setMovement(self.Static)

    def mouseDoubleClickEvent(self, e: QtGui.QMouseEvent):

        query = self.f_list[self.currentIndex().row()]

        self.f_list = np.argsort(self.similarity_matrix.loc[query].values)
        image_model = ShowImages(self.f_list, self)
        self.setModel(image_model)
        self.show()

        image_model = ShowImages(self.f_list, self)
        self.setModel(image_model)
        self.show()

        print()


if __name__ =="__main__":


    app = QtWidgets.QApplication(sys.argv)
    window = CorePhotoSearch()
    window.show()
    window.raise_()
    sys.exit(app.exec_())