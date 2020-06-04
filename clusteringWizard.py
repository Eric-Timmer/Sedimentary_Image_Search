from PyQt5 import QtCore, QtWidgets
import corePhotoSearch
import sys
import time


class ClusteringWizard(QtWidgets.QWizard):
    page1 = 1
    page2 = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.params = list()
        self.recalc_sim = False

        self.setPage(self.page1, Page1(self))
        self.setPage(self.page2, Page2(self))
        self.setStartId(1)

        # self.setGeometry((500, 700))
        # self.showFullScreen()
        self.resize(700, 600)
        self.setOption(self.DisabledBackButtonOnLastPage)


    def set_params(self, params):
        self.params = params

    @property
    def get_params(self):
        return self.params


class Page1(QtWidgets.QWizardPage):
    def __init__(self, parent=None, ):
        super().__init__(parent)
        self.parent = parent
        self.similarity_calculated = True

        self.layout = QtWidgets.QVBoxLayout()
        self.widget_list = list()
        self.matrix_already_calculated = self.check_if_matrix_calculated()
        self.specify_layout()
        self.setLayout(self.layout)


    def spacing(self):
        # add space
        spacer = QtWidgets.QWidget()
        spacer.setMinimumHeight(10)
        # self.widget_list.append(self.spacing())
        return spacer


    def specify_layout(self):
        if self.matrix_already_calculated:
            self.calculate_similarity_check = QtWidgets.QCheckBox("Re-calculate Similarity Scores")
            self.calculate_similarity_check.stateChanged.connect(self.change_similarity_calculated_state)
            self.calculate_similarity_check.setChecked(True)
            self.layout.addWidget(self.calculate_similarity_check)
            self.layout.addSpacing(10)

        self.n_clusters = QtWidgets.QLabel("Specify the number of clusters to split images into: ")
        self.n_clusters_input = QtWidgets.QLineEdit()
        self.layout.addWidget(self.n_clusters)
        self.layout.addWidget(self.n_clusters_input)
        # self.layout.addWidget(self.spacing())


        self.tile_size_label = QtWidgets.QLabel("Number of tiles per image\n(larger number = slower)")
        self.widget_list.append(self.tile_size_label)

        self.tile_size_input = QtWidgets.QLineEdit()
        self.widget_list.append(self.tile_size_input)

        self.layout.addWidget(self.tile_size_label)
        self.layout.addWidget(self.tile_size_input)
        # self.layout.addWidget(self.spacing())

        self.overlap_label = QtWidgets.QLabel("Proportion of overlap between tiles:")
        self.widget_list.append(self.overlap_label)

        self.overlap_input = QtWidgets.QLineEdit()
        self.widget_list.append(self.overlap_input)

        self.layout.addWidget(self.overlap_label)
        self.layout.addWidget(self.overlap_input)
        # self.layout.addWidget(self.spacing())

        self.similarity_label = QtWidgets.QLabel("Select which parameters to use for image similarity scoring:")
        self.widget_list.append(self.similarity_label)
        self.layout.addWidget(self.similarity_label)

        self.mean_check = QtWidgets.QCheckBox("Mean")
        self.var_check = QtWidgets.QCheckBox("Variance")
        self.otsu_check = QtWidgets.QCheckBox("Otsu heterogeneity")
        self.psnr_check = QtWidgets.QCheckBox("Tiled PSNR")
        self.rmse_check = QtWidgets.QCheckBox("Tiled RMSE")
        self.hist_check = QtWidgets.QCheckBox("Tiled histogram")
        self.widget_list.append(self.mean_check)
        self.widget_list.append(self.var_check)
        self.widget_list.append(self.otsu_check)
        self.widget_list.append(self.psnr_check)
        self.widget_list.append(self.rmse_check)
        self.widget_list.append(self.hist_check)


        self.layout.addWidget(self.mean_check)
        self.layout.addWidget(self.var_check)
        self.layout.addWidget(self.otsu_check)
        self.layout.addWidget(self.psnr_check)
        self.layout.addWidget(self.rmse_check)
        self.layout.addWidget(self.hist_check)
        # self.layout.addWidget(self.spacing())

        self.multi_label = QtWidgets.QLabel("Specify number of processors to use:\n(specify -1 for all processors)")
        self.widget_list.append(self.multi_label)
        self.multi_input = QtWidgets.QLineEdit()
        self.widget_list.append(self.multi_input)

        self.layout.addWidget(self.multi_label)
        self.layout.addWidget(self.multi_input)


    def initializePage(self):
        self.tile_size_input.setMaximumSize(QtCore.QSize(50, 20))
        self.overlap_input.setMaximumSize(QtCore.QSize(50, 20))
        self.multi_input.setMaximumSize(QtCore.QSize(50, 20))
        self.n_clusters_input.setMaximumSize(QtCore.QSize(50, 20))
        self.tile_size_input.setMinimumSize(QtCore.QSize(50, 20))
        self.overlap_input.setMinimumSize(QtCore.QSize(50, 20))
        self.multi_input.setMinimumSize(QtCore.QSize(50, 20))
        self.n_clusters_input.setMinimumSize(QtCore.QSize(50, 20))

        self.tile_size_input.setText("10")
        self.overlap_input.setText("0.33")
        self.multi_input.setText("-1")
        self.n_clusters_input.setText("8")

        self.mean_check.setChecked(True)
        self.var_check.setChecked(True)
        self.otsu_check.setChecked(True)
        self.psnr_check.setChecked(True)
        self.rmse_check.setChecked(True)
        self.hist_check.setChecked(True)



    def return_parameters(self):
        if self.similarity_calculated is False:
            params = [float(self.tile_size_input.text()),
                      float(self.overlap_input.text()),
                      self.mean_check.isChecked(),
                      self.var_check.isChecked(),
                      self.otsu_check.isChecked(),
                      self.psnr_check.isChecked(),
                      self.rmse_check.isChecked(),
                      self.hist_check.isChecked(),
                      int(self.multi_input.text()),
                      int(self.n_clusters_input.text())]
        else:
            params = [int(self.n_clusters_input.text())]
        return params

    def nextId(self):
        if self.similarity_calculated is False:
            try:
                float(self.tile_size_input.text())
            except ValueError:
                error_dialog = QtWidgets.QErrorMessage()
                error_dialog.showMessage("The value you entered in the tile size input field is not valid. Please fix.")
                error_dialog.exec_()
                return ClusteringWizard.page1
            try:
                float(self.overlap_input.text())
            except ValueError:
                error_dialog = QtWidgets.QErrorMessage()
                error_dialog.showMessage("The value you entered in the overlap input field is not valid. Please fix. \n"
                                         "The value must be a decimal value < 1")
                error_dialog.exec_()
                return ClusteringWizard.page1

            try:
                int(self.multi_input.text())
            except ValueError:
                error_dialog = QtWidgets.QErrorMessage()
                error_dialog.showMessage("The value you entered in the multi processing input is not valid. Please fix.")
                error_dialog.exec_()
                return ClusteringWizard.page1

        try:
            int(self.n_clusters_input.text())
        except ValueError:
            error_dialog = QtWidgets.QErrorMessage()
            error_dialog.showMessage("The value you entered in the number of clusters box is not valid. Please fix.")
            error_dialog.exec_()
            return ClusteringWizard.page1


        params = self.return_parameters()
        self.parent.set_params(params)
        return ClusteringWizard.page2

    def check_if_matrix_calculated(self):

        # TODO check if matrix is already calculated
        return True

    def change_similarity_calculated_state(self):
        if self.calculate_similarity_check.isChecked():
            self.similarity_calculated = False
            for i in self.widget_list:
                self.layout.addWidget(i)
        else:
            self.similarity_calculated = True
            for i in reversed(self.widget_list):
                self.layout.removeWidget(i)

class Page2(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.layout = QtWidgets.QVBoxLayout()

        self.title_label = QtWidgets.QLabel()
        self.layout.addWidget(self.title_label)
        self.layout.addSpacing(10)

        self.start_button = QtWidgets.QPushButton("Start Processing (press to start)")
        self.start_button.clicked.connect(self.start_processing)

        self.layout.addWidget(self.start_button)
        self.layout.addSpacing(10)

        self.progress = QtWidgets.QLabel()
        self.layout.addWidget(self.progress)
        self.setLayout(self.layout)

        self.done_processing = False
        self.start = 0

    def initializePage(self):
        self.title_label.setText("This step can take a few minutes (sometimes longer), please be patient. Do not close wizard.")

    def nextID(self):
        return - 1

    def start_processing(self):

        params = self.parent.get_params
        # self.time_elapsed()
        self.threadpool = QtCore.QThreadPool()

        worker = Worker(None, params, corePhotoSearch.SimilarityGenerator)
        self.threadpool.start(worker)
        self.start = time.time()
        worker2 = WorkerTime(self.time_elapsed)
        self.threadpool.start(worker2)

    def isComplete(self):
        if self.done_processing is False:
            return False
        else:
            return True

    def time_elapsed(self):

        while self.threadpool.activeThreadCount() == 2:
            current = time.time() - self.start
            self.progress.setText(time.strftime("Time Elapsed: %H:%M:%S", time.gmtime(current)))
        if self.threadpool.activeThreadCount() == 1:
            self.progress.setText("Processing Complete!")
            self.done_processing = True
            self.completeChanged.emit()




class Worker(QtCore.QRunnable):
    def __init__(self, f_list, params, func):
        super(Worker, self).__init__()
        self.f_list = f_list
        self.params = params
        self.func = func

    @QtCore.pyqtSlot()
    def run(self):

        self.func(self.f_list, self.params)


class WorkerTime(QtCore.QRunnable):
    def __init__(self, func):
        super(WorkerTime, self).__init__()
        self.func = func

    @QtCore.pyqtSlot()
    def run(self):
        self.func()




if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    clustering_wizard = ClusteringWizard()
    clustering_wizard.show()
    sys.exit(app.exec_())