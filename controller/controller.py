from visualizer import Visualizer
from camera import WebCameraAdapter
from arena import ArenaModel
import cv2 
import numpy as np


class BotController:
    def __init__(self, args):
        self.camera_adapter = WebCameraAdapter()
        self.arena_model = ArenaModel(args.width, args.length, args.bot_width,
            args.bot_length, args.aruco_size, args.aruco_margin
        )
        self.visualizer = Visualizer(args.preview_width)
        self.corner_points = []

    def _select_arena_corners(self, event, x, y, flags, param):
        # Mouse callback function to capture corner points
        resize_factor = self.visualizer.resize_factor
        if event == cv2.EVENT_LBUTTONDOWN and len(self.corner_points) < 4:
            x = int(x / resize_factor)
            y = int(y / resize_factor)
            self.corner_points.append((x, y))
            # print(f"Corner {len(corner_points)}: ({x}, {y})")
        
        # If by adding the last point we complete the rectangle, then we can proceed
        if len(self.corner_points) == 4:
            frame = self.camera_adapter.get_frame()
            self.arena_model.capture_corners(self.corner_points, frame)

            transformed_frame = self.arena_model.get_state(frame).transformed_frame
            px_per_mm_width = transformed_frame.shape[1] / self.arena_model.arena_width
            # px_per_mm_length = transformed_frame.shape[0] / self.arena_model.arena_length

            # # Should be very close. Otherwise, warn the user that the selected points do not correspond to the preset width/height
            # if not np.isclose(px_per_mm_width, px_per_mm_length, rtol=0.01):
            #     print(f"Warning: The selected points do not correspond to the preset width/height ratio ({px_per_mm_width} vs {px_per_mm_length})")

            #     # Drop the corners, unset everything, and start over
            #     self.corner_points = []
            #     self.arena_model.capture_corners(None, None)
            # else:
            # We are good to go
            self.visualizer.set_original_view_mouse_callback(lambda *args: None)
            self.visualizer.set_px_per_mm(px_per_mm_width)

    def handle_user_input(self, user_input, current_frame) -> bool:
        """
        Handle user input. Returns True if the program should exit.
        """
        if user_input == 'quit':
            return True
        elif user_input == 'capture_baseline':
            self.arena_model.capture_baseline(current_frame)

        return False


    def run(self):
        # Set up initial frame properties
        first_frame = self.camera_adapter.get_frame()
        self.visualizer.set_resize_factor(first_frame.shape[1])

        # Initially, we don't have the corners selected
        self.visualizer.set_original_view_mouse_callback(self._select_arena_corners)

        while True:
            frame = self.camera_adapter.get_frame()
            arena_state = self.arena_model.get_state(frame)
            self.visualizer.visualize(frame, arena_state, self.corner_points)
            user_input = self.visualizer._collect_user_input()

            if self.handle_user_input(user_input, frame):
                break
