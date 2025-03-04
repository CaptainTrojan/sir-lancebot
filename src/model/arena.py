from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class BotModel:
    x: float
    y: float
    vx: float
    vy: float
    width_px: float
    length_px: float
    theta: float
    omega: float

@dataclass
class ArenaState:
    arena_width: float
    arena_length: float
    bot: BotModel = None
    enemies: list[BotModel] = None
    transformed_frame: np.ndarray = None
    baseline_frame: np.ndarray = None

    def __post_init__(self):
        if self.enemies is None:
            self.enemies = []


# Corner mapping
CORNER_ARUCO_ID_MAPPING = [
    "front_left",
    "front_right",
    "back_left",
    "back_right"
]


class ArenaModel:
    def __init__(self, arena_width, arena_length, bot_width, bot_length, aruco_size, aruco_margin):
        self.arena_width = arena_width
        self.arena_length = arena_length
        self.bot_width = bot_width
        self.bot_length = bot_length
        self.aruco_size = aruco_size
        self.aruco_margin = aruco_margin
        self.kalman_bot = self._create_bot_kalman_filter()
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250),
            cv2.aruco.DetectorParameters()
        )

        self.corner_points = None
        self.tf_H = None
        self.tf_W = None
        self.tf_size = None
        self.baseline_frame = None

    def capture_corners(self, corner_points, some_frame):
        if corner_points is None:
            self.corner_points = None
            return
        self.corner_points = np.array(corner_points, dtype=np.float32)

        # Get any frame and transform it, then set expected transformed frame size
        transformed_frame = self._transform_view(some_frame)
        self.tf_H, self.tf_W = transformed_frame.shape[:2]
        self.tf_size = np.array([self.tf_W, self.tf_H])

    def capture_baseline(self, frame):
        # Transform the frame
        transformed_frame = self._transform_view(frame)
        self.baseline_frame = transformed_frame

    def update_state(self, frame):
        if self.corner_points is None:
            return ArenaState(self.arena_width, self.arena_length)

        # Transform the frame from the camera adapter
        transformed_frame = self._transform_view(frame)

        # Assert that the transformed frame has the expected size
        assert transformed_frame.shape[:2] == (self.tf_H, self.tf_W), f"Transformed frame has unexpected size {transformed_frame.shape[:2]}, expected {(self.tf_H, self.tf_W)}."

        # Detect the bot in the transformed frame
        bot_coords = self._detect_bot(transformed_frame) if transformed_frame is not None else None
        
        # Convert bot corners to center and orientation
        center, orientation = self._corners_to_center_orientation(bot_coords)
        c, theta, v, omega = self._update_bot_kalman(self.kalman_bot, center, orientation)
        
        if bot_coords is not None:
            width = np.linalg.norm(bot_coords[0] - bot_coords[1])
            length = np.linalg.norm(bot_coords[0] - bot_coords[2])
        else:
            width = None
            length = None

        # Normalize c and v to the arena dimensions
        c = c / self.tf_size
        v = v / self.tf_size

        # Build the model
        bot_model = BotModel(*c, *v, width, length, theta, omega)
        return ArenaState(self.arena_width, self.arena_length, bot_model, transformed_frame=transformed_frame, baseline_frame=self.baseline_frame)
    
    def _create_bot_kalman_filter(self):
        kf = cv2.KalmanFilter(6, 3)  # 6 states (x, y, theta, vx, vy, omega), 3 measurements (x, y, theta)
        
        # State transition matrix
        kf.transitionMatrix = np.array([[1, 0, 0, 1, 0, 0],
                                        [0, 1, 0, 0, 1, 0],
                                        [0, 0, 1, 0, 0, 1],
                                        [0, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 0, 1]], dtype=np.float32)
        
        # Measurement matrix
        kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                        [0, 1, 0, 0, 0, 0],
                                        [0, 0, 1, 0, 0, 0]], dtype=np.float32)
        
        # Process noise covariance
        kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03

        # Measurement noise covariance
        kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.1

        return kf

    def _transform_view(self, frame):
        if self.corner_points is None:
            return None

        src_points = self.corner_points

        # Calculate new dimensions to fit the window while preserving aspect ratio
        new_height = frame.shape[0]
        new_width = int(new_height * (self.arena_width / self.arena_length))

        dst_points = np.array([
                    [0, 0],
                    [new_width - 1, 0],
                    [new_width - 1, new_height - 1],
                    [0, new_height - 1]
                ], dtype="float32")
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        transformed_frame = cv2.warpPerspective(frame, matrix, (new_width, new_height))
        return transformed_frame

    def _detect_bot(self, transformed_frame):
        """ Returns FL, FR, BR, BL coordinates of the bot in the transformed frame or None if no bot is detected """

        # Detect 5x5 ArUcos ID 0-3 in the transformed frame
        corners, ids, _ = self.detector.detectMarkers(transformed_frame)

        if ids is None:
            return None
        
        ids = ids.flatten()
        
        # Drop all ids that are not in the range 0-3
        _c = []
        _i = []
        for c, i in zip(corners, ids):
            if i in range(4):
                _c.append(c)
                _i.append(i)
        corners = _c
        ids = np.array(ids)

        # Check if there is at least one aruco detected
        if len(ids) < 1:
            return None
        
        # Try finding each aruco corner
        true_corners = [None] * 4
        for corner, idx in zip(corners, ids):
            true_corners[idx] = corner[0][0]  # corner is a list of 4 corners, we only need the first one, which is the top left corner

        # Get first aruco detection 
        aruco_corners = corners[0][0]
        aruco_id = int(ids[0])
        corner = CORNER_ARUCO_ID_MAPPING[aruco_id]

        # Based on the aruco position and size, calculate the bot corners.
        # Each of the corner arucos are aruco_margin away from the bot corners (each edge by aruco_margin mm).

        vec_right = aruco_corners[1] - aruco_corners[0]
        vec_down = aruco_corners[3] - aruco_corners[0]
        # Scale them to account for the fact that we are working in px space not mm space
        uvec_right = vec_right / self.aruco_size # np.linalg.norm(vec_right)
        uvec_down = vec_down / self.aruco_size # np.linalg.norm(vec_down)

        origin = aruco_corners[0]
        if corner == "front_left":
            pass
        if corner == "front_right":
            uvec_right, uvec_down = -uvec_down, uvec_right
            origin = origin - uvec_right * (self.bot_width - 2 * self.aruco_margin)
        elif corner == "back_right":
            uvec_right, uvec_down = -uvec_right, -uvec_down
            origin = origin - uvec_right * (self.bot_width - 2 * self.aruco_margin) - uvec_down * (self.bot_length - 2 * self.aruco_margin)
        elif corner == "back_left":
            uvec_right, uvec_down = uvec_down, -uvec_right
            origin = origin - uvec_down * (self.bot_length - 2 * self.aruco_margin)

        # origin = front left aruco corner

        # Calculate the front left corner: top left aruco corner
        if true_corners[0] is None:
            true_corners[0] = origin

        # Calculate the front right corner: front left corner + width of the bot - 2 * aruco_margin
        if true_corners[1] is None:
            true_corners[1] = true_corners[0] + uvec_right * (self.bot_width - 2 * self.aruco_margin)

        # Calculate the back left corner: front left corner + length of the bot - 2 * aruco_margin
        if true_corners[2] is None:
            true_corners[2] = true_corners[0] + uvec_down * (self.bot_length - 2 * self.aruco_margin)

        # Calculate the back right corner: front right corner + length of the bot
        if true_corners[3] is None:
            true_corners[3] = true_corners[1] + uvec_down * (self.bot_length - 2 * self.aruco_margin)

        # Grow the aruco corners by aruco_margin to get the bot corners
        bot_fl = true_corners[0] - uvec_right * self.aruco_margin - uvec_down * self.aruco_margin
        bot_fr = true_corners[1] + uvec_right * self.aruco_margin - uvec_down * self.aruco_margin
        bot_bl = true_corners[2] - uvec_right * self.aruco_margin + uvec_down * self.aruco_margin
        bot_br = true_corners[3] + uvec_right * self.aruco_margin + uvec_down * self.aruco_margin

        return bot_fl, bot_fr, bot_br, bot_bl

    @staticmethod
    def _corners_to_center_orientation(bot_coords):
        if bot_coords is None:
            return None, None

        bot_fl, bot_fr, bot_br, bot_bl = bot_coords
        center = (bot_fl + bot_br) / 2
        theta_1 = np.arctan2(bot_fl[1] - bot_bl[1], bot_fl[0] - bot_bl[0])
        theta_2 = np.arctan2(bot_fr[1] - bot_br[1], bot_fr[0] - bot_br[0])
        theta = (theta_1 + theta_2) / 2  # Some easy smoothing

        # print(theta_1, theta_2, theta)

        return center, theta

    @staticmethod
    def _update_bot_kalman(kf, center, orientation):
        # Predict next state
        prediction = kf.predict()
        predicted_center = (prediction[0], prediction[1])
        predicted_orientation = prediction[2]
        predicted_velocity = (prediction[3], prediction[4])
        predicted_omega = prediction[5]

        if center is not None and orientation is not None:
            # If new measurements are available, correct the filter
            measured = np.array([[np.float32(center[0])], 
                                [np.float32(center[1])], 
                                [np.float32(orientation)]])
            corrected = kf.correct(measured)
            stabilized_center = (corrected[0], corrected[1])
            stabilized_orientation = corrected[2]
            stabilized_velocity = (corrected[3], corrected[4])
            stabilized_omega = corrected[5]
        else:
            # Use the prediction if no new measurements
            stabilized_center = predicted_center
            stabilized_orientation = predicted_orientation
            stabilized_velocity = predicted_velocity
            stabilized_omega = predicted_omega

        return np.array(stabilized_center).squeeze(), stabilized_orientation.squeeze(), np.array(stabilized_velocity).squeeze(), stabilized_omega.squeeze()
            
    @staticmethod
    def _center_orientation_to_corners(center, orientation, bot_width, bot_length):
        # Calculate half-width and half-length in pixel units
        half_width = bot_width / 2
        half_length = bot_length / 2

        # Compute corner offsets based on orientation
        cos_theta = np.cos(orientation)
        sin_theta = np.sin(orientation)

        front_left = (center[0] + half_length * cos_theta - half_width * sin_theta,
                    center[1] + half_length * sin_theta + half_width * cos_theta)
        front_right = (center[0] + half_length * cos_theta + half_width * sin_theta,
                    center[1] + half_length * sin_theta - half_width * cos_theta)
        back_left = (center[0] - half_length * cos_theta - half_width * sin_theta,
                    center[1] - half_length * sin_theta + half_width * cos_theta)
        back_right = (center[0] - half_length * cos_theta + half_width * sin_theta,
                    center[1] - half_length * sin_theta - half_width * cos_theta)

        return front_left, front_right, back_left, back_right