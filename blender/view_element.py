import argparse
import json
import cv2
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

def project_points(points, P):
    # Project 3D points to 2D
    projected_points = []
    for point in points:
        point = np.hstack((point, 1))
        projected = P @ point
        projected /= projected[2]
        projected_points.append((int(projected[0]), int(projected[1])))
    return np.array(projected_points)


def bot_to_BEV(points, bot, arena_corners_bev, bev_scale_factor, target_height):
    bot_points = points.copy()
    x = bot['x']
    y = bot['y']
    theta = bot['theta']

    # Rotate the bot points
    r = R.from_euler('z', theta)
    bot_points = r.apply(bot_points)

    # Translate the bot points to the bot's position
    bot_points += np.array([x, y, 0])
    bot_points = bot_points[:, :2]

    # Shift and scale the points to BEV
    bot_points = bot_points - arena_corners_bev[0]
    bot_points = bot_points * bev_scale_factor
    bot_points = bot_points.astype(np.int32)

    # Flip the Y-axis so that it points upwards
    bot_points[:, 1] = target_height - bot_points[:, 1]

    return bot_points

def render_element(image_path, metadata_path):
    # Load image
    image = cv2.imread(image_path)

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Camera P matrix
    P = np.array(metadata['camera_P'])

    # Arena corners
    arena_corners = np.array(metadata['arena_corners'])
    arena_corners_bev = arena_corners[:, :2]

    # Project 3D points to 2D
    projected_arena_corners = project_points(arena_corners, P)

    # Perform a perspective transformation on the arena corners -> BEV
    # Get the transformation matrix
    arena_bev_width = arena_corners_bev[1][0] - arena_corners_bev[0][0]
    arena_bev_height = arena_corners_bev[2][1] - arena_corners_bev[0][1]
    target_corners = np.array([
        [0, 0],
        [arena_bev_width, 0],
        [0, arena_bev_height],
        [arena_bev_width, arena_bev_height]
    ])
    target_height = cv2.getWindowImageRect("Image")[3]
    bev_scale_factor = target_height / arena_bev_height
    target_width = int(arena_bev_width * bev_scale_factor)
    target_corners = target_corners * bev_scale_factor
    M = cv2.getPerspectiveTransform(projected_arena_corners.astype(np.float32), target_corners.astype(np.float32))

    # Perform the transformation
    bev_image = cv2.warpPerspective(image, M, (target_width, target_height))

    # Flip the BEV image so that Y-axis points upwards
    bev_image = cv2.flip(bev_image, 0)

    # Resize the image so that its height is the same as BEV image
    image_scale_factor = target_height / image.shape[0]
    image = cv2.resize(image, (int(image_scale_factor * image.shape[1]), target_height))

    # Draw bot BEV bounding boxes on the BEV view
    for bot in metadata['bots']:
        width = bot['width']
        length = bot['length']

        # Define the 2D points of the bot relative to its center
        bot_points = np.array([
            [-width/2, -length/2, 0],
            [width/2, -length/2, 0],
            [width/2, length/2, 0],
            [-width/2, length/2, 0],
        ])

        # Transform the bot points to BEV
        bev_bot_points = bot_to_BEV(bot_points, bot, arena_corners_bev, bev_scale_factor, target_height)

        # Draw the bot points
        for i in range(4):
            cv2.line(bev_image, tuple(bev_bot_points[i]), tuple(bev_bot_points[(i+1)%4]), (0, 255, 255), 2)

        # Collect wheel coordinates
        wheel_points = []
        for wheel in bot['config']['wheels']:
            x = -width/2 - wheel['x_offset']
            y = wheel['y_coord']
            wheel_points.append([x, y, 0])

            x = width/2 + wheel['x_offset']
            wheel_points.append([x, y, 0])

        # Transform the wheel points to BEV
        bev_wheel_points = bot_to_BEV(np.array(wheel_points), bot, arena_corners_bev, bev_scale_factor, target_height)

        # Draw the wheel points
        for i in range(len(bev_wheel_points)):
            cv2.circle(bev_image, tuple(bev_wheel_points[i]), 5, (0, 255, 255), -1)

        # Draw a blue arrow pointing forward from the bot's center
        center = bot_to_BEV(np.array([[0, 0, 0]]), bot, arena_corners_bev, bev_scale_factor, target_height)
        forward = bot_to_BEV(np.array([[0, length, 0]]), bot, arena_corners_bev, bev_scale_factor, target_height)

        cv2.arrowedLine(bev_image, tuple(center[0]), tuple(forward[0]), (0, 255, 255), 2)

    # Concatenate the images
    concatenated_image = np.hstack((image, bev_image))

    # Draw the metadata path somewhere
    # cv2.putText(concatenated_image, metadata_path, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    cv2.imshow("Image", concatenated_image)

def main(dataset_path):
    image_dir = os.path.join(dataset_path, "images")
    metadata_dir = os.path.join(dataset_path, "metadata")

    image_files = sorted(os.listdir(image_dir))
    metadata_files = sorted(os.listdir(metadata_dir))

    if len(image_files) == 0:
        print(f"No images found in '{image_dir}'.")
        return
    
    # Display the images
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 1920, 1080)

    index = 0
    while True:
        image_path = os.path.join(image_dir, image_files[index])
        metadata_path = os.path.join(metadata_dir, metadata_files[index])
        render_element(image_path, metadata_path)

        key = cv2.waitKey(0)
        if key == ord('a'):
            index = (index - 1) % len(image_files)
        elif key == ord('d'):
            index = (index + 1) % len(image_files)
        elif key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render dataset elements.')
    parser.add_argument('-d', '--dataset_path', type=str, default='BB_SYNTH_DATA', help='Path to the dataset directory.')

    args = parser.parse_args()
    if os.path.exists(args.dataset_path):
        main(args.dataset_path)
    else:
        print(f"Dataset path '{args.dataset_path}' does not exist.")
        exit(1)