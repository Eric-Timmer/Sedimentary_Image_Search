from PyQt5 import QtCore, QtGui, QtWidgets
import sys

"""
TODO have to add:
1. User controls for label name, line_width, text_size, box_colour.
2. Save dictionary containing bounding box information
3. Re-plot bounding boxes (and labels) from database
4. Link with the main application window
5. Allow User to delete images

6. Crop images? and Rotate Images? Those are the last few things to add.
7


My vision for this product:

1. Bounding Box (for semantic segmentation) feature labeling

2. Extraction of core sleeves from photos automatically
3. Allow user to delete images, rotate images, crop images.
4. Allow user to configure learning method (deep learning is probably best...allow user to input their own net).
5. Allow user to configure sampling interval, etc.
6. Generate logs (geological logs) with data tha can be output into Petrel, etc.

MAke it clear that this is the only core loggin platform that uses deep learning to log core.

"""


class ImageViewer(QtWidgets.QGraphicsView):

    def __init__(self, scene_dict):
        super().__init__()
        self.scene = QtWidgets.QGraphicsScene()
        self.pixmap = QtWidgets.QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap)

        self.setScene(self.scene)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)

        # specify current_image
        self.current_image = 'out/0/l_0_294.jpg.jpg'

        # Once the image viewer has been initialized, load an image.
        self.loadImage()

        # parameters for drawing labeled rectangles
        self.current_rect = None
        self.scene_dict = scene_dict

        self.current_label = "temp"

    def setPhoto(self, pixmap=None):
        # self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.pixmap.setPixmap(pixmap)

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.scale(1.1, 1.1)
        elif event.angleDelta().y() < 0:
            self.scale(0.9, 0.9)

    def loadImage(self):
        self.setPhoto(QtGui.QPixmap(self.current_image))

    def mousePressEvent(self, event):
        if self.validate_rect_position(event) is False:
            return

        if event.button() == QtCore.Qt.LeftButton:
            # left click to draw rectangle
            self.current_rect = QtWidgets.QGraphicsRectItem()
            # self.current_rect.setBrush(QtCore.Qt.black)
            self.current_rect.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
            self.scene.addItem(self.current_rect)
            self._start = self.mapToScene(event.pos())
            rect = QtCore.QRectF(self._start, self._start)
            self.current_rect.setRect(rect)
        else:
            # Right click to delete
            self.delete_rect(self.mapToScene(event.pos()).x(), self.mapToScene(event.pos()).y())

    def mouseMoveEvent(self, event):
        if self.validate_rect_position(event) is False:
            return
        if self.current_rect is None:
            return
        else:
            r = QtCore.QRectF(self._start, self.mapToScene(event.pos())).normalized()
            self.current_rect.setRect(r)

    def mouseReleaseEvent(self, event):

        if self.current_rect is None:
            return

        self.add_label()
        self.current_rect = None
        self.scene_dict[self.current_image] = self.scene.items()

    def validate_rect_position(self, event):
        # do not allow rectangle to be drawn ouside of image boundaries
        if self.pixmap.contains(self.mapToScene(event.pos())):
            return True
        else:
            return False

    def add_label(self, ):
        x0, y0, x1, y1 = self.get_coords(self.current_rect)
        lab = QtWidgets.QGraphicsTextItem(self.current_label)
        lab.setPos(x0, y0-20)
        self.scene.addItem(lab)

    def get_coords(self, obj):
        x0 = obj.boundingRect().topLeft().x()
        y0 = obj.boundingRect().topLeft().y()
        x1 = obj.boundingRect().bottomRight().x()
        y1 = obj.boundingRect().bottomRight().y()
        return x0, y0, x1, y1

    def delete_rect(self, x, y):
        for it, i in enumerate(self.scene.items()):
            if i.type() != QtWidgets.QGraphicsRectItem().type():
                continue
            if i.contains(QtCore.QPoint(x, y)):

                self.scene.removeItem(i)
                if self.scene.items()[it-1].type() == QtWidgets.QGraphicsTextItem().type():
                    self.scene.removeItem(self.scene.items()[it-1])
                self.scene_dict[self.current_image] = self.scene.items()
                return

    def set_label_name(self, input_name):
        self.current_label = input_name

    def clear_scene(self):
        for i in self.scene.items():
            if i.type() != QtWidgets.QGraphicsPixmapItem().type():
                self.scene.removeItem(i)

    def redraw_scene(self):
        try:
            out = self.scene_dict[self.current_image]
        except KeyError:
            return
        for i in out:
            if i not in self.scene.items():
                self.scene.addItem(i)

    def set_image(self, img_path):

        self.clear_scene()
        self.current_image = img_path
        self.loadImage()
        self.redraw_scene()


class WindowInterface(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.label_input = None
        self.setup_toolbar()
        self.setWindowTitle("Core Image Labeler and Training Data Generator CILTDG")
        scene_dict = dict()
        self.image_viewer = ImageViewer(scene_dict)
        self.setCentralWidget(self.image_viewer)

        # TODO CONNECT TO THE Main APPLICATION THAT CONTAINS THE file list, etc.

    def setup_toolbar(self):
        toolbar = self.addToolBar("Tool Bar")

        # Navigation buttons
        prev_button = QtWidgets.QPushButton("Previous Image")
        next_button = QtWidgets.QPushButton("Next Image")
        prev_button.clicked.connect(self.previous_image)
        next_button.clicked.connect(self.next_image)
        toolbar.addWidget(prev_button)
        toolbar.addWidget(next_button)

        # add space
        spacer1 = QtWidgets.QWidget()
        spacer1.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        toolbar.addWidget(spacer1)

        # label input
        label_title = QtWidgets.QLabel()
        label_title.setText("Feature Label: ")
        self.label_input = QtWidgets.QLineEdit()
        self.label_input.setMaximumWidth(200)
        self.label_input.textChanged.connect(self.label_changed)
        toolbar.addWidget(label_title)
        toolbar.addWidget(self.label_input)

        # add space
        spacer2 = QtWidgets.QWidget()
        spacer2.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        toolbar.addWidget(spacer2)

        # options and help
        options_button = QtWidgets.QPushButton("Options")
        options_button.clicked.connect(self.show_options)
        toolbar.addWidget(options_button)
        help_button = QtWidgets.QPushButton("Help")
        help_button.clicked.connect(self.show_help)
        toolbar.addWidget(help_button)

    def previous_image(self):
        self.image_viewer.set_image("out/0/l_0_1035.jpg.jpg")
        return

    def next_image(self):
        self.image_viewer.set_image("out/0/l_0_1012.jpg.jpg")
        return

    def show_options(self):
        dialog = QtWidgets.QDialog()

        layout = QtWidgets.QVBoxLayout()

        """
        Text Size
        Text Colour
        Text Transparency
        Box Line Width
        Box Line Colour
        Box Line Transparency        
        
        """
        # TODO implement options...
        self.text_size = 12
        self.text_transparency = 0
        self.box_line_width = 1

        text_size_label = QtWidgets.QLabel("Select\nText Size")
        text_size_input = QtWidgets.QLineEdit()
        text_size_input.setText(str(self.text_size))
        text_size_input.setMaximumWidth(50)

        text_colour_button = QtWidgets.QPushButton("Select Text Colour")

        text_transparency_label = QtWidgets.QLabel("Text Transparency")
        text_transparency_input = QtWidgets.QLineEdit()
        text_transparency_input.setText(str(self.text_transparency))
        text_transparency_input.setMaximumWidth(50)

        box_line_width_label = QtWidgets.QLabel("Box Line Width")
        box_line_width_input = QtWidgets.QLineEdit()
        box_line_width_input.setText(str(self.box_line_width))
        box_line_width_input.setMaximumWidth(50)

        box_colour_button = QtWidgets.QPushButton("Select Box Colour")

        box_transparency_label = QtWidgets.QLabel("Box Transparency")
        box_transparency_input = QtWidgets.QLineEdit()
        box_transparency_input.setText(str(self.text_transparency))
        box_transparency_input.setMaximumWidth(50)

        layout.addWidget(text_size_label)
        layout.addWidget(text_size_input)
        layout.addWidget(text_colour_button)
        layout.addWidget(text_transparency_label)
        layout.addWidget(text_transparency_input)
        layout.addSpacing(50)
        layout.addWidget(box_line_width_label)
        layout.addWidget(box_line_width_input)
        layout.addWidget(box_colour_button)
        layout.addWidget(box_transparency_label)
        layout.addWidget(box_transparency_input)

        dialog.setLayout(layout)

        dialog.setWindowTitle("Options")
        dialog.exec_()

        return

    def show_help(self):

        return

    def label_changed(self):
        self.image_viewer.set_label_name(self.label_input.text())


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = WindowInterface()
    main.show()
    sys.exit(app.exec_())