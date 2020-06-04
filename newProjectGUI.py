from PyQt5 import QtCore, QtWidgets, QtGui
import os
import corePhotoSearch
import sys
import logging
import time


class NewProject(QtWidgets.QWizard):
    page1 = 1
    page2 = 2
    page3 = 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self.file_list = list()
        self.params = list()

        self.setPage(self.page1, Page1(self))
        self.setPage(self.page2, Page2(self))
        self.setPage(self.page3, Page3(self))

        self.setStartId(1)

        self.setWindowTitle("Initiate New Project")
        self.resize(700, 500)
        self.setOption(self.DisabledBackButtonOnLastPage)

    def set_file_list(self, file_list):
        self.file_list = file_list

    @property
    def get_file_list(self):
        return self.file_list

    def set_params(self, params):
        self.params = params

    @property
    def get_params(self):
        return self.params


class Page1(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

        self.layout = QtWidgets.QVBoxLayout()
        self.file_finder_button = QtWidgets.QPushButton("Select Core Photo Folder", self)
        self.file_finder_button.setMaximumSize(200,50)
        self.file_finder_button.clicked.connect(self.open_file)
        self.layout.addWidget(self.file_finder_button)

        self.n_img_label = QtWidgets.QLabel()
        self.layout.addWidget(self.n_img_label)

        self.setLayout(self.layout)

        self.f_list = list()

    def open_file(self):
        self.f_list = list()
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder Containing Images (.jpg format)")

        for root, dirs, files in os.walk(directory):
            for f in files:
                if f.lower().endswith(".jpg") or f.lower().endswith(".jpeg"):
                    self.f_list.append(os.path.join(root, f))
        if len(self.f_list) > 0:
            self.parent.set_file_list(self.f_list)
        self.show_file_statistics()
        self.completeChanged.emit()

    def show_file_statistics(self):
        self.n_img_label.setText("%i .jpg files were found in directory" % len(self.f_list))

        return

    def isComplete(self):
        if len(self.f_list) == 0:
            return False
        else:
            return True

    def nextId(self):

        if len(self.f_list) > 0:
            return NewProject.page2
        else:
            return NewProject.page1


class Page2(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        layout = QtWidgets.QVBoxLayout()

        self.title_label = QtWidgets.QLabel()
        layout.addWidget(self.title_label)
        layout.addSpacing(10)

        self.tile_size_label = QtWidgets.QLabel()
        self.tile_size_input = QtWidgets.QLineEdit()
        layout.addWidget(self.tile_size_label)
        layout.addWidget(self.tile_size_input)
        layout.addSpacing(10)

        self.overlap_label = QtWidgets.QLabel()
        self.overlap_input = QtWidgets.QLineEdit()
        layout.addWidget(self.overlap_label)
        layout.addWidget(self.overlap_input)
        layout.addSpacing(10)

        self.similarity_label = QtWidgets.QLabel()
        self.mean_check = QtWidgets.QCheckBox("Mean")
        self.var_check = QtWidgets.QCheckBox("Variance")
        self.otsu_check = QtWidgets.QCheckBox("Otsu heterogeneity")
        self.psnr_check = QtWidgets.QCheckBox("Tiled PSNR")
        self.rmse_check = QtWidgets.QCheckBox("Tiled RMSE")
        self.hist_check = QtWidgets.QCheckBox("Tiled histogram")
        layout.addWidget(self.similarity_label)
        layout.addWidget(self.mean_check)
        layout.addWidget(self.var_check)
        layout.addWidget(self.otsu_check)
        layout.addWidget(self.psnr_check)
        layout.addWidget(self.rmse_check)
        layout.addWidget(self.hist_check)
        layout.addSpacing(10)

        self.multi_label = QtWidgets.QLabel()
        self.multi_input = QtWidgets.QLineEdit()
        layout.addWidget(self.multi_label)
        layout.addWidget(self.multi_input)

        self.setLayout(layout)



    def initializePage(self):
        self.title_label.setText("Similarity measure parameters: ")
        self.tile_size_label.setText("Number of tiles per image\n(larger number = slower)")
        self.overlap_label.setText("Proportion of overlap between tiles:")
        self.similarity_label.setText("Select which parameters to use for image similarity scoring:")
        self.multi_label.setText("Specify number of processors to use:\n(specify -1 for all processors)")

        self.tile_size_input.setText("10")
        self.overlap_input.setText("0.33")
        self.multi_input.setText("-1")

        self.tile_size_input.setMaximumSize(QtCore.QSize(50, 20))
        self.overlap_input.setMaximumSize(QtCore.QSize(50, 20))
        self.multi_input.setMaximumSize(QtCore.QSize(50, 20))

        self.mean_check.setChecked(True)
        self.var_check.setChecked(True)
        self.otsu_check.setChecked(True)
        self.psnr_check.setChecked(True)
        self.rmse_check.setChecked(True)
        self.hist_check.setChecked(True)

    def return_parameters(self):

        params = [float(self.tile_size_input.text()),
                  float(self.overlap_input.text()),
                  self.mean_check.isChecked(),
                  self.var_check.isChecked(),
                  self.otsu_check.isChecked(),
                  self.psnr_check.isChecked(),
                  self.rmse_check.isChecked(),
                  self.hist_check.isChecked(),
                  int(self.multi_input.text())]
        return params

    def nextId(self):
        try:
            float(self.tile_size_input.text())
        except ValueError:
            error_dialog = QtWidgets.QErrorMessage()
            error_dialog.showMessage("The value you entered in the tile size input field is not valid. Please fix.")
            error_dialog.exec_()
            return NewProject.page2
        try:
            float(self.overlap_input.text())
        except ValueError:
            error_dialog = QtWidgets.QErrorMessage()
            error_dialog.showMessage("The value you entered in the overlap input field is not valid. Please fix. \n"
                                     "The value must be a decimal value < 1")
            error_dialog.exec_()
            return NewProject.page2

        try:
            int(self.multi_input.text())
        except ValueError:
            error_dialog = QtWidgets.QErrorMessage()
            error_dialog.showMessage("The value you entered in the multi processing input is not valid. Please fix.")
            error_dialog.exec_()
            return NewProject.page2


        params = self.return_parameters()
        self.parent.set_params(params)
        return NewProject.page3


class Page3(QtWidgets.QWizardPage):
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
        f_list = self.parent.get_file_list
        # self.time_elapsed()
        self.threadpool = QtCore.QThreadPool()

        worker = Worker(f_list, params, corePhotoSearch.SimilarityGenerator)
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
    new_project = NewProject()
    new_project.show()
    sys.exit(app.exec_())