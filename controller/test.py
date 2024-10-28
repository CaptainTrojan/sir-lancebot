import cv2
import depthai as dai
import argparse
import numpy as np

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

WINDOW_TRANSFORMED = "Transformed view"
WINDOW_ORIGINAL = "Original view"
WINDOW_MODEL = "Model view"

# Initialize variables to store corner points
corner_points = []
arena_width_mm = args.width
arena_length_mm = args.length
bot_width = args.bot_width
bot_length = args.bot_length
aruco_size = args.aruco_size
aruco_margin = args.aruco_margin

# Corner mapping
CORNER_ARUCO_ID_MAPPING = [
    "front_left",
    "front_right",
    "back_left",
    "back_right"
]

# Mouse callback function to capture corner points
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(corner_points) < 4:
        x = int(x / resize_factor)
        y = int(y / resize_factor)
        corner_points.append((x, y))
        print(f"Corner {len(corner_points)}: ({x}, {y})")

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
xoutVideo = pipeline.create(dai.node.XLinkOut)

xoutVideo.setStreamName("video")

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
# camRgb.initialControl.setManualFocus(27)
# camRgb.setVideoSize(1920, 1080)

xoutVideo.input.setBlocking(False)
xoutVideo.input.setQueueSize(1)

# Linking
camRgb.video.link(xoutVideo.input)

# Connect to device and start pipeline
def transform_view(corner_points, arena_width_mm, arena_length_mm, frame):
    src_points = np.array(corner_points, dtype="float32")

            # Calculate new dimensions to fit the window while preserving aspect ratio
    new_height = frame.shape[0]
    new_width = int(new_height * (arena_width_mm / arena_length_mm))

    dst_points = np.array([
                [0, 0],
                [new_width - 1, 0],
                [new_width - 1, new_height - 1],
                [0, new_height - 1]
            ], dtype="float32")
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed_frame = cv2.warpPerspective(frame, matrix, (new_width, new_height))
    return transformed_frame


def detect_bot(transformed_frame, detector: cv2.aruco.ArucoDetector):
    """ Returns FL, FR, BR, BL coordinates of the bot in the transformed frame or None if no bot is detected """

    # Detect 5x5 ArUcos ID 0-3 in the transformed frame
    corners, ids, rejectedImgPoints = detector.detectMarkers(transformed_frame)

    if ids is None:
        return None
    
    ids = ids.flatten()
    
    # Drop all ids that are not in the range 0-3
    corners, ids = zip(*[(corner, id) for corner, id in zip(corners, ids) if id in range(4)])
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
    uvec_right = vec_right / aruco_size # np.linalg.norm(vec_right)
    uvec_down = vec_down / aruco_size # np.linalg.norm(vec_down)

    origin = aruco_corners[0]
    if corner == "front_left":
        pass
    if corner == "front_right":
        uvec_right, uvec_down = -uvec_down, uvec_right
        origin = origin - uvec_right * (bot_width - 2 * aruco_margin)
    elif corner == "back_right":
        uvec_right, uvec_down = -uvec_right, -uvec_down
        origin = origin - uvec_right * (bot_width - 2 * aruco_margin) - uvec_down * (bot_length - 2 * aruco_margin)
    elif corner == "back_left":
        uvec_right, uvec_down = uvec_down, -uvec_right
        origin = origin - uvec_down * (bot_length - 2 * aruco_margin)

    # origin = front left aruco corner

    # Calculate the front left corner: top left aruco corner
    if true_corners[0] is None:
        true_corners[0] = origin

    # Calculate the front right corner: front left corner + width of the bot - 2 * aruco_margin
    if true_corners[1] is None:
        true_corners[1] = true_corners[0] + uvec_right * (bot_width - 2 * aruco_margin)

    # Calculate the back left corner: front left corner + length of the bot - 2 * aruco_margin
    if true_corners[2] is None:
        true_corners[2] = true_corners[0] + uvec_down * (bot_length - 2 * aruco_margin)

    # Calculate the back right corner: front right corner + length of the bot
    if true_corners[3] is None:
        true_corners[3] = true_corners[1] + uvec_down * (bot_length - 2 * aruco_margin)

    # Grow the aruco corners by aruco_margin to get the bot corners
    bot_fl = true_corners[0] - uvec_right * aruco_margin - uvec_down * aruco_margin
    bot_fr = true_corners[1] + uvec_right * aruco_margin - uvec_down * aruco_margin
    bot_bl = true_corners[2] - uvec_right * aruco_margin + uvec_down * aruco_margin
    bot_br = true_corners[3] + uvec_right * aruco_margin + uvec_down * aruco_margin

    return bot_fl, bot_fr, bot_br, bot_bl


def corners_to_center_orientation(bot_coords):
    if bot_coords is None:
        return None, None

    bot_fl, bot_fr, bot_br, bot_bl = bot_coords
    center = (bot_fl + bot_br) / 2
    theta_1 = np.arctan2(bot_fl[1] - bot_bl[1], bot_fl[0] - bot_bl[0])
    theta_2 = np.arctan2(bot_fr[1] - bot_br[1], bot_fr[0] - bot_br[0])
    theta = (theta_1 + theta_2) / 2  # Some easy smoothing

    # print(theta_1, theta_2, theta)

    return center, theta


def create_bot_kalman_filter():
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


def update_bot_kalman(kf, center, orientation):
    # Predict next state
    prediction = kf.predict()
    predicted_center = (prediction[0], prediction[1])
    predicted_orientation = prediction[2]

    if center is not None and orientation is not None:
        # If new measurements are available, correct the filter
        measured = np.array([[np.float32(center[0])], 
                             [np.float32(center[1])], 
                             [np.float32(orientation)]])
        corrected = kf.correct(measured)
        stabilized_center = (corrected[0], corrected[1])
        stabilized_orientation = corrected[2]
    else:
        # Use the prediction if no new measurements
        stabilized_center = predicted_center
        stabilized_orientation = predicted_orientation

    return stabilized_center, stabilized_orientation
        
def center_orientation_to_corners(center, orientation, bot_width, bot_length):
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

with dai.Device(pipeline) as device:
    video = device.getOutputQueue(name="video", maxSize=1, blocking=False)
    cv2.namedWindow(WINDOW_ORIGINAL, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_ORIGINAL, mouse_callback)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    some_frame = video.get().getCvFrame()
    original_W = some_frame.shape[1]
    original_H = some_frame.shape[0]
    W = args.preview_width
    H = int(args.preview_width * (original_H / original_W))
    resize_factor = W / original_W
    kalman_bot = create_bot_kalman_filter()

    while True:
        videoIn = video.get()
        frame = videoIn.getCvFrame()

        # Draw corner points on the frame
        for point in corner_points:
            cv2.circle(frame, point, 25, (0, 255, 0), -1)

        # If four corner points are captured, compute perspective transform
        if len(corner_points) == 4:
            transformed_frame = transform_view(corner_points, arena_width_mm, arena_length_mm, frame)

            bot_coords = detect_bot(transformed_frame, detector)
            bot_center, bot_theta = corners_to_center_orientation(bot_coords)
            
            stabilized_center, stabilized_theta = update_bot_kalman(kalman_bot, bot_center, bot_theta)

            if stabilized_center is not None:
                cv2.arrowedLine(transformed_frame, tuple(map(int, stabilized_center)), 
                    (int(stabilized_center[0] + 450 * np.cos(stabilized_theta)), 
                    int(stabilized_center[1] + 450 * np.sin(stabilized_theta))), (255, 0, 0), 2)
                
                normalized_center = np.array(stabilized_center).squeeze() / np.array([transformed_frame.shape[1], transformed_frame.shape[0]])

            if bot_coords is not None:
                for i in range(4):
                    cv2.circle(transformed_frame, tuple(map(int, bot_coords[i])), 15, (0, 0, 255), -1)

            # Resize the transformed frame to fit a window with the specified width
            transformed_frame = cv2.resize(transformed_frame, dsize=None, fx=resize_factor, fy=resize_factor)

            # Display the transformed frame
            cv2.imshow(WINDOW_TRANSFORMED, transformed_frame)

            # Build the model view - arena as a rectangle and the bot as an arrow center -> orientation
            model_view = np.zeros((H, W, 3), dtype=np.uint8)
            model_arena_margin_percent = 0.1
            model_arena_width = W * (1 - 2 * model_arena_margin_percent)
            model_arena_length = H * (1 - 2 * model_arena_margin_percent)

            # Draw the arena as a red rectangle
            arena_top_left = (int(W * model_arena_margin_percent), int(H * model_arena_margin_percent))
            arena_bottom_right = (int(W * (1 - model_arena_margin_percent)), int(H * (1 - model_arena_margin_percent)))
            cv2.rectangle(model_view, arena_top_left, arena_bottom_right, (0, 0, 255), 2)

            # Get the bot center and theta
            if stabilized_center is not None:
                model_bot_center = normalized_center * np.array([model_arena_width, model_arena_length]) + np.array(arena_top_left)

                # Draw the bot as a thick, short arrow
                bot_arrow_length = 150
                bot_arrow_end = (int(model_bot_center[0] + bot_arrow_length * np.cos(stabilized_theta)),
                                int(model_bot_center[1] + bot_arrow_length * np.sin(stabilized_theta)))
                cv2.arrowedLine(model_view, tuple(map(int, model_bot_center)), bot_arrow_end, (255, 0, 0), 5)

            # Display the model view
            cv2.imshow(WINDOW_MODEL, model_view)


        # Resize the original frame to fit a window with the specified width
        frame = cv2.resize(frame, dsize=None, fx=resize_factor, fy=resize_factor)
            
        # Display the original frame 
        cv2.imshow(WINDOW_ORIGINAL, frame)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()