import cv2
import numpy as np
from arena import ArenaState


WINDOW_ORIGINAL = "Original view"
WINDOW_MODEL = "Model view"
WINDOW_TRANSFORMED = "Transformed view"
WINDOW_BASELINE_DIFF = "Baseline difference"


class Visualizer:
    def __init__(self, preview_width):
        self.preview_width = preview_width
        self.resize_factor = None
        self.px_per_mm = None

    def set_resize_factor(self, original_width):
        self.resize_factor = self.preview_width / original_width

    def set_px_per_mm(self, px_per_mm):
        self.px_per_mm = px_per_mm

    def visualize(self, frame, arena_state: ArenaState, corner_points=None):
        if arena_state.transformed_frame is not None:
            # Draw the arena state on top of the transformed state (full cover)
            vis_transformed_frame = arena_state.transformed_frame.copy()
            self._draw_arena_state(vis_transformed_frame, arena_state, 0)

            transformed_frame_resized = cv2.resize(vis_transformed_frame, 
                                                   dsize=None, fx=self.resize_factor, fy=self.resize_factor)
            cv2.imshow(WINDOW_TRANSFORMED, transformed_frame_resized)

            # Draw the arena state on a blank canvas with the arena drawn as a rectangle
            model_view = np.zeros((int(arena_state.transformed_frame.shape[0] * self.resize_factor),
                                   int(arena_state.transformed_frame.shape[1] * self.resize_factor), 3), dtype=np.uint8)
            arena_border_margin = 0.1
            arena_top_left = (int(arena_border_margin * model_view.shape[1]), int(arena_border_margin * model_view.shape[0]))
            arena_bottom_right = (int((1 - arena_border_margin) * model_view.shape[1]), int((1 - arena_border_margin) * model_view.shape[0]))
            cv2.rectangle(model_view, arena_top_left, arena_bottom_right, (0, 0, 255), 2)
            self._draw_arena_state(model_view, arena_state, arena_border_margin)

            cv2.imshow(WINDOW_MODEL, model_view)

            # Render the baseline difference for visualization
            if arena_state.baseline_frame is not None:
                baseline_diff = cv2.absdiff(arena_state.transformed_frame, arena_state.baseline_frame)
                baseline_diff = cv2.cvtColor(baseline_diff, cv2.COLOR_BGR2GRAY)

                # Apply thresholding 0-1 (background/foreground)
                _, baseline_diff = cv2.threshold(baseline_diff, 30, 255, cv2.THRESH_BINARY)

                # Apply opening to remove noise
                kernel = np.ones((5, 5), np.uint8)
                baseline_diff = cv2.morphologyEx(baseline_diff, cv2.MORPH_OPEN, kernel)

                baseline_diff = cv2.resize(baseline_diff, dsize=None, fx=self.resize_factor, fy=self.resize_factor)
                cv2.imshow(WINDOW_BASELINE_DIFF, baseline_diff)
            
        # Draw the corner points
        vis_frame = frame.copy()
        if corner_points is not None:
            for corner_point in corner_points:
                cv2.circle(vis_frame, corner_point, 15, (255, 0, 0), -1)
                # print(frame.shape, corner_point)

        # Draw the original frame for user to select the arena corners
        frame_resized = cv2.resize(vis_frame, dsize=None, fx=self.resize_factor, fy=self.resize_factor)

        cv2.imshow(WINDOW_ORIGINAL, frame_resized)

    def _draw_arena_state(self, frame, arena_state: ArenaState, border_margin):
        target_area = [
            border_margin,
            border_margin,
            1 - border_margin,
            1 - border_margin
        ]

        # Draw our bot green
        self._draw_bot(arena_state.bot,
                       frame[
                           int(target_area[1] * frame.shape[0]):int(target_area[3] * frame.shape[0]),
                           int(target_area[0] * frame.shape[1]):int(target_area[2] * frame.shape[1])
                       ],
                       arena_state.transformed_frame.shape,
                       (0, 255, 0)
        )

        # Draw enemy bots TODO: Implement this

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
                [-length / 2, -width / 2],   # bottom-left corner
                [length / 2, -width / 2],    # bottom-right corner
                [length / 2, width / 2],     # top-right corner
                [-length / 2, width / 2]     # top-left corner
            ])

            # Rotate and translate corners according to bot's center and orientation
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            rotated_corners = (rotation_matrix @ bot_corners.T).T + center

            # Convert to ints
            rotated_corners = rotated_corners.astype(int)

            # Draw a rectangle
            cv2.polylines(frame, [rotated_corners], isClosed=True, color=color, thickness=frame.shape[0]//100)

    def set_original_view_mouse_callback(self, callback):
        cv2.namedWindow(WINDOW_ORIGINAL)
        cv2.setMouseCallback(WINDOW_ORIGINAL, callback)

    def _collect_user_input(self) -> str:
        key = cv2.waitKey(1)
        if key == ord('q'):
            return "quit"
        elif key == ord(' '):
            return "capture_baseline"
        return None
    def __del__(self):
        cv2.destroyAllWindows()