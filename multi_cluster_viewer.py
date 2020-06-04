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
    def __init__(self, f_list):
        super().__init__()
        self.user_interface()
        self.f_list = f_list
        image_model = ShowImages(self.f_list, self)
        self.setModel(image_model)
        # self.show()


    def user_interface(self):
        # window parameters
        self.setMinimumHeight(300+10)
        self.setFlow(self.flow())

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


class MultiClusterViewer(QtWidgets.QMainWindow):
    def __init__(self, labels, f_list):
        super().__init__()
        self.cluster_dict = self.separate_f_list_into_clusters(f_list, labels)

        scroll_area = QtWidgets.QScrollArea()
        central_widget = QtWidgets.QWidget(self)
        self.layout = QtWidgets.QVBoxLayout()
        self.initiate_cluster_separators()
        central_widget.setLayout(self.layout)
        scroll_area.setWidget(central_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setVisible(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)

        self.setCentralWidget(scroll_area)

        self.setWindowTitle("CorePhotoSearch")



    def initiate_cluster_separators(self):

        for k, f in enumerate(range(0, len(self.cluster_dict.keys()))):
            f_list = self.cluster_dict[f]
            lab = QtWidgets.QLabel("Cluster %i" % f)
            self.layout.addWidget(lab)
            self.layout.addWidget(CorePhotoSearch(f_list))

            self.layout.addSpacing(50)



    @staticmethod
    def separate_f_list_into_clusters(f_list, labels):
        cluster_dict = dict()
        for i, f in enumerate(f_list):

            try:
                cluster_dict[labels[i]].append(f)
            except KeyError:
                cluster_dict[labels[i]] = [f]
        return cluster_dict



if __name__ == "__main__":
    f_list = glob.glob("out/*/*.jpg")[0:50]

    labels = list(np.random.randint(0, 5, 50))
    app = QtWidgets.QApplication(sys.argv)
    window = MultiClusterViewer(labels, f_list)
    window.show()
    window.raise_()
    sys.exit(app.exec_())