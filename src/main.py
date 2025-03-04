import json
import sys
from model.arena import ArenaModel
from model.camera import WebCameraModel
from view.view import MainView
from viewmodel.arena_viewmodel import ArenaViewModel
from PyQt6.QtWidgets import QApplication

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

if __name__ == '__main__':
    # Load configuration from config.json
    config = load_config('config.json')

    app = QApplication(sys.argv)
    camera_model = WebCameraModel()
    arena_model = ArenaModel(**config['arena'])
    view_model = ArenaViewModel(arena_model, camera_model)
    window = MainView(view_model)
    window.show()
    sys.exit(app.exec())