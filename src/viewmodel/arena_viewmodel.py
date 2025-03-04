from model.camera import WebCameraModel
from model.arena import ArenaModel, ArenaState
import cv2
import numpy as np
from time import perf_counter
from PyQt6.QtCore import pyqtSignal, QObject


class ArenaViewModel(QObject):
    webcam_list_changed = pyqtSignal(list)
    
    def __init__(self, arena_model: ArenaModel, camera_model: WebCameraModel):
        super().__init__()
        self.camera_adapter = camera_model
        self.arena_model = arena_model
        self.last_vis_time = None
        
    def update_webcam_list(self):
        webcam_ids = self.camera_adapter.list_available_ids()
        self.webcam_list_changed.emit(webcam_ids)
        
    def select_webcam(self, webcam_id):
        self.camera_adapter.set_camera_id(webcam_id)

    def get_original_view(self, frame, corner_points=None):
        if frame is None:
            return None
        vis_frame = frame.copy()
        if corner_points is not None:
            for corner_point in corner_points:
                cv2.circle(vis_frame, corner_point, 15, (255, 0, 0), -1)
        
        if self.last_vis_time is not None:
            fps = 1 / (perf_counter() - self.last_vis_time)
            cv2.putText(vis_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        self.last_vis_time = perf_counter()
        return vis_frame

    def get_transformed_view(self, arena_state: ArenaState):
        if arena_state.transformed_frame is not None:
            vis_transformed_frame = arena_state.transformed_frame.copy()
            self._draw_arena_state(vis_transformed_frame, arena_state, 0)
            return vis_transformed_frame
        return None

    def get_model_view(self, arena_state: ArenaState):
        if arena_state.transformed_frame is not None:
            model_view = np.zeros((arena_state.transformed_frame.shape[0], arena_state.transformed_frame.shape[1], 3), dtype=np.uint8)
            arena_border_margin = 0.1
            arena_top_left = (int(arena_border_margin * model_view.shape[1]), int(arena_border_margin * model_view.shape[0]))
            arena_bottom_right = (int((1 - arena_border_margin) * model_view.shape[1]), int((1 - arena_border_margin) * model_view.shape[0]))
            cv2.rectangle(model_view, arena_top_left, arena_bottom_right, (0, 0, 255), 2)
            self._draw_arena_state(model_view, arena_state, arena_border_margin)
            return model_view
        return None

    def get_baseline_diff_view(self, arena_state: ArenaState):
        if arena_state.baseline_frame is not None:
            baseline_diff = cv2.absdiff(arena_state.transformed_frame, arena_state.baseline_frame)
            baseline_diff = cv2.cvtColor(baseline_diff, cv2.COLOR_BGR2GRAY)
            _, baseline_diff = cv2.threshold(baseline_diff, 30, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            baseline_diff = cv2.morphologyEx(baseline_diff, cv2.MORPH_OPEN, kernel)
            return baseline_diff
        return None

    def _draw_arena_state(self, frame, arena_state: ArenaState, border_margin):
        target_area = [
            border_margin,
            border_margin,
            1 - border_margin,
            1 - border_margin
        ]
        self._draw_bot(arena_state.bot,
                       frame[
                           int(target_area[1] * frame.shape[0]):int(target_area[3] * frame.shape[0]),
                           int(target_area[0] * frame.shape[1]):int(target_area[2] * frame.shape[1])
                       ],
                       arena_state.transformed_frame.shape,
                       (0, 255, 0)
        )

    def _denormalize(self, point, frame):
        x, y = point
        return int(x * frame.shape[1]), int(y * frame.shape[0])

    def _draw_bot(self, bot, frame, og_frame_shape, color):
        size_factor = frame.shape[0] / og_frame_shape[0]
        center = bot.x, bot.y
        theta = bot.theta
        center = self._denormalize(center, frame)
        arrow_end = center + frame.shape[0]/6 * np.array([np.cos(theta), np.sin(theta)])
        cv2.arrowedLine(frame, center, tuple(arrow_end.astype(int)), color, frame.shape[0]//100)
        if bot.width_px is not None and bot.length_px is not None:
            width, length = bot.width_px * size_factor, bot.length_px * size_factor
            bot_corners = np.array([
                [-length / 2, -width / 2],
                [length / 2, -width / 2],
                [length / 2, width / 2],
                [-length / 2, width / 2]
            ])
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            rotated_corners = (rotation_matrix @ bot_corners.T).T + center
            rotated_corners = rotated_corners.astype(int)
            cv2.polylines(frame, [rotated_corners], isClosed=True, color=color, thickness=frame.shape[0]//100)

