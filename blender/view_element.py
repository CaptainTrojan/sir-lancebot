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

    # Project [0, 0, 0] to 2D
    origin = np.array([[0.0, 0.0, 0.0]])
    projected_origin = project_points(origin, P)

    # Draw points on the image
    for image_point, world_point in zip(projected_arena_corners, arena_corners):
        x, y = image_point
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        # cv2.putText(image, f"({world_point[0]:.2f}, {world_point[1]:.2f})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # Draw the origin
    x, y = projected_origin[0]
    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

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

    # Draw bot 3D bounding boxes on the image view
    bots = metadata['bots']
    for bot in bots:
        x = bot['x']
        y = bot['y']
        z = bot['z']
        width = bot['width']
        length = bot['length']
        height = bot['height']
        theta = bot['theta']

        # Define the 3D points of the bot relative to its center
        bot_points = np.array([
            [-width/2, -length/2, 0],
            [width/2, -length/2, 0],
            [width/2, length/2, 0],
            [-width/2, length/2, 0],
            [-width/2, -length/2, height],
            [width/2, -length/2, height],
            [width/2, length/2, height],
            [-width/2, length/2, height]
        ])

        # Rotate the bot points
        r = R.from_euler('z', theta)
        bot_points = r.apply(bot_points)

        # Translate the bot points to the bot's position
        bot_points += np.array([x, y, z])

        # Project the bot points
        projected_bot_points = project_points(bot_points, P)

        # Scale them by the image scale factor
        projected_bot_points = projected_bot_points * image_scale_factor
        projected_bot_points = projected_bot_points.astype(np.int32)

        # Draw the bot points
        for i in range(4):
            cv2.line(image, tuple(projected_bot_points[i]), tuple(projected_bot_points[i+4]), (0, 255, 255), 2)
            cv2.line(image, tuple(projected_bot_points[i]), tuple(projected_bot_points[(i+1)%4]), (0, 255, 255), 2)
            cv2.line(image, tuple(projected_bot_points[i+4]), tuple(projected_bot_points[(i+1)%4 + 4]), (0, 255, 255), 2)

    # Draw bot BEV bounding boxes on the BEV view
    for bot in bots:
        x = bot['x']
        y = bot['y']
        width = bot['width']
        length = bot['length']
        theta = bot['theta']

        # Define the 2D points of the bot relative to its center
        bot_points = np.array([
            [-width/2, -length/2, 0],
            [width/2, -length/2, 0],
            [width/2, length/2, 0],
            [-width/2, length/2, 0],
        ])

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

        # Draw the bot points
        for i in range(4):
            cv2.line(bev_image, tuple(bot_points[i]), tuple(bot_points[(i+1)%4]), (0, 255, 255), 2)

    # Concatenate the images
    concatenated_image = np.hstack((image, bev_image))

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