from PyQt5 import QtWidgets, QtCore, QtGui
import sys
import glob
import numpy as np


class NewProjectDialog(QtWidgets.QWizard):

    def __init__(self):
        super().__init__()

    def setup_window(self):

        return


class App(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.orignial_img_size = (300, 300)
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()

        self.setWindowTitle("CorePhotoSearch")
        self.init_menu()

        self.init_viewer()

        self.init_toolbar()


        # setup the buttons toolbar
        self.show()

    def init_menu(self):
        # Setup the menu bar
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('File')

        new_project_action = QtWidgets.QAction('New Project...', self)
        open_existing_project_action = QtWidgets.QAction('Open...', self)
        file_menu.addAction(new_project_action)
        file_menu.addAction(open_existing_project_action)

    def init_viewer(self):
        self.viewer = CorePhotoSearch()
        self.setCentralWidget(self.viewer)
        # self.setCentralWidget(self.show_images)

    def init_toolbar(self):
        toolbar = self.addToolBar("Tool Bar")

        test = QtWidgets.QAction("new", self)
        toolbar.addAction(test)

        self.zoom_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.zoom_slider.valueChanged.connect(self.zoom_images)
        self.zoom_slider.setRange(20, 200)
        self.zoom_slider.setMaximumSize(QtCore.QSize(200, 20))
        self.zoom_slider.setTickInterval(50)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.zoom_slider.setPageStep(50)
        toolbar.addWidget(QtWidgets.QLabel("Zoom"))

        toolbar.addWidget(self.zoom_slider)

    def zoom_images(self):

        size = self.zoom_slider.value()

        new_x = int(size/100. * self.orignial_img_size[0])
        new_y = int(size/100. * self.orignial_img_size[1])

        CorePhotoSearch.resize_icons(self.viewer, new_x, new_y)




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
        # # rescale images dynamically
        self.setResizeMode(1)

        #load images in batch
        self.LayoutMode(1)
        self.setBatchSize(50)

        # wrap images onto screen
        self.setWrapping(True)

        # set image and grid size
        self.setIconSize(QtCore.QSize(300, 300))
        self.setMovement(self.Static)
        self.ResizeMode(self.Fixed)

    def mouseDoubleClickEvent(self, e: QtGui.QMouseEvent):

        query = self.f_list[self.currentIndex().row()]

        self.f_list = np.argsort(self.similarity_matrix.loc[query].values)
        image_model = ShowImages(self.f_list, self)
        self.setModel(image_model)
        self.show()


    def resize_icons(self, new_x, new_y):
        self.setIconSize(QtCore.QSize(new_x, new_y))
        self.repaint()
        print(new_x, new_y)



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())