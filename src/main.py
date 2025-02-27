import argparse
import sys
from src.model.arena import ArenaModel
from src.model.camera import WebCameraAdapter
from src.view.view import Visualizer
from viewmodel import BotViewModel
from PyQt6.QtWidgets import QApplication

if __name__ == '__main__':

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Arena dimensions")
    parser.add_argument("width", type=int, help="Width of the arena (mm)")
    parser.add_argument("length", type=int, help="Length of the arena (mm)")
    parser.add_argument("bot_width", type=int, help="Width of the bot (mm)")
    parser.add_argument("bot_length", type=int, help="Length of the bot (mm)")
    parser.add_argument("aruco_size", type=int, help="Size of the ArUco markers (mm)")
    parser.add_argument("aruco_margin", type=int, help="Margin between the ArUco markers and the bot edge (mm)")
    parser.add_argument("-pw", "--preview_width", type=int, default=900, help="Width of the preview windows")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    arena_model = ArenaModel(args.width, args.length, args.bot_width, args.bot_length, args.aruco_size, args.aruco_margin)
    camera_model = WebCameraAdapter()
    view_model = BotViewModel(arena_model, camera_model)
    view = Visualizer(view_model)
    view.show()
    sys.exit(app.exec())