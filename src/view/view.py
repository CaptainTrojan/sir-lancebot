from dataclasses import dataclass
import sys
from time import time
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout,
                            QToolBar, QMenuBar, QWidget, QPushButton, QFrame, QSizePolicy,
                            QScrollArea, QDialog, QVBoxLayout, QPushButton, QMenu)
from PyQt6.QtGui import QImage, QPixmap, QAction, QIcon
from PyQt6.QtCore import QTimer, Qt

# Global variables
streams = {}
last_update_time = time()

# External function mock (replace with actual implementation)
def get_streams():
    global streams, last_update_time
    if time() - last_update_time > 1:
        streams.clear()
        last_update_time = time()
        # Generate random images with varying dimensions
        for i in range(np.random.randint(1, 6)):
            w = np.random.randint(100, 301)
            h = np.random.randint(100, 301)
            img = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.line(img, (0, 0), (w, h), (255, 255, 255), 2)
            cv2.line(img, (0, h), (w, 0), (255, 255, 255), 2)
            cv2.putText(img, f"Stream {i+1}", (10, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            streams[f"cam_{i}"] = img
    return streams

@dataclass
class StreamView:
    name: str
    img: np.ndarray
    widget: QLabel
    
    @property
    def width(self):
        return self.img.shape[1]

    @property
    def height(self):
        return self.img.shape[0]

class MainView(QMainWindow):
    def __init__(self, view_model):
        super().__init__()
        self.view_model = view_model
        
        self.setWindowTitle("Main Application")
        self.setGeometry(100, 100, 1200, 800)
        
        # Menu bar
        menu_bar = QMenuBar(self)
        settings_menu = menu_bar.addMenu("Settings")
        
        self.setMenuBar(menu_bar)
        
        # Select Webcam menu
        self.select_webcam_menu = QMenu("Select Webcam", self)
        settings_menu.addMenu(self.select_webcam_menu)
        
        # Main widget setup
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Left panel (fixed width)
        left_panel = QWidget()
        left_panel.setFixedWidth(70)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # Align to top
        for i in range(1, 4):
            button = QPushButton()
            button.setFixedSize(50, 50)
            button.setIcon(QIcon("icon.png"))
            left_layout.addWidget(button)
        
        # Central view layout
        central_layout = QVBoxLayout()
        
        # Central view
        self.main_display = QLabel()
        self.main_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_display.setSizePolicy(QSizePolicy.Policy.Expanding, 
                                      QSizePolicy.Policy.Expanding)
        
        # Stream name label
        self.stream_name_label = QLabel("<No stream selected>")
        self.stream_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Add central view and stream name label to central layout
        central_layout.addWidget(self.main_display)
        central_layout.addWidget(self.stream_name_label)
        
        # Right sidebar with scroll
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setFixedWidth(240)
        sidebar_scroll.setWidgetResizable(True)
        self.sidebar_content = QWidget()
        self.sidebar_layout = QVBoxLayout(self.sidebar_content)
        self.sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.sidebar_layout.setSpacing(15)
        self.sidebar_layout.setContentsMargins(5, 5, 15, 5)
        sidebar_scroll.setWidget(self.sidebar_content)
        
        # Assemble main layout
        layout.addWidget(left_panel)
        layout.addLayout(central_layout)
        layout.addWidget(sidebar_scroll)
        
        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_displays)
        self.timer.start(30)
        
        # Webcam refresh timer
        self.webcam_timer = QTimer()
        self.webcam_timer.timeout.connect(self.refresh_webcam_list)
        self.webcam_timer.start(5000)
        
        # Selected stream
        self.selected_stream = None
        self.opened_streams = {}
        self.webcam_list = []

        # Add webcam listener
        self.view_model.webcam_list_changed.connect(self.update_webcam_list)
        self.refresh_webcam_list()

    def refresh_webcam_list(self):
        self.view_model.update_webcam_list()

    def update_webcam_list(self, webcams):
        self.webcam_list = webcams
        self.select_webcam_menu.clear()
        if not self.webcam_list:
            action = QAction("<No webcams found>", self)
            action.setEnabled(False)
            self.select_webcam_menu.addAction(action)
        else:
            for index, webcam in enumerate(self.webcam_list):
                action = QAction(f"Webcam {webcam}", self)
                action.triggered.connect(lambda _, index=index: self.select_webcam(index))
                self.select_webcam_menu.addAction(action)

    def select_webcam(self, camera_index):
        self.view_model.select_webcam(camera_index)

    def update_displays(self):
        # Update main view
        streams = get_streams()
        if self.selected_stream:
            if self.selected_stream in streams:
                self.update_main_view(streams[self.selected_stream])
            else:
                self.selected_stream = None
                self.stream_name_label.setText("<No stream selected>")
                self.main_display.clear()
        
        # Update sidebar
        self.update_sidebar(streams)

    def update_main_view(self, img):
        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, img.strides[0], QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.main_display.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.main_display.setPixmap(pixmap)
        
    def create_stream_widget(self, key, img, w, h):
        # Create image label
        label = QLabel()
        label.setFixedWidth(w)
        label.setFixedHeight(h)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Scale image appropriately
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation
        )
        label.setPixmap(pixmap)
        
        # Add click event
        label.mousePressEvent = lambda event, key=key: self.select_stream(key)
        
        # Add to sidebar layout
        self.sidebar_layout.addWidget(label)
        
        # Add to opened streams
        self.opened_streams[key] = StreamView(key, img, label)
        
    def destroy_stream_widget(self, key):
        self.opened_streams[key].widget.deleteLater()
        del self.opened_streams[key]
        
    def update_sidebar(self, streams):
        # Find each widget that needs to be removed (stream no longer exists)
        to_delete = []
        for stream_name in self.opened_streams:
            if stream_name not in streams:
                to_delete.append(stream_name)

        # Remove widgets
        for stream_name in to_delete:
            self.destroy_stream_widget(stream_name)
        
        # Handle streams
        sidebar_width = 200
        for key, img in streams.items():
            h, w = img.shape[:2]
            aspect_ratio = h / w
            
            # If we are already showing this stream
            if key in self.opened_streams:  
                # check if the image size has changed
                old_h, old_w = self.opened_streams[key].height, self.opened_streams[key].width
                if old_h != h or old_w != w:
                    # Remove the old widget
                    self.destroy_stream_widget(key)

                    # Create a new widget
                    self.create_stream_widget(key, img, sidebar_width, int(sidebar_width * aspect_ratio))
            
            # If we are not showing this stream yet
            else:
                # Create a new widget
                self.create_stream_widget(key, img, sidebar_width, int(sidebar_width * aspect_ratio))
        
        # Add spacer
        self.sidebar_layout.addStretch()

    def select_stream(self, key):
        self.selected_stream = key
        self.stream_name_label.setText(f"Selected Stream: {key}")

if __name__ == "__main__":
    from src.viewmodel.arena_viewmodel import ArenaViewModel
    from src.model.camera import WebCameraModel
    from src.model.arena import ArenaModel

    app = QApplication(sys.argv)
    camera_model = WebCameraModel()
    arena_model = ArenaModel()
    view_model = ArenaViewModel(arena_model, camera_model)
    window = MainView(view_model)
    window.show()
    sys.exit(app.exec())