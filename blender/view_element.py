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

def load_data(image_path, metadata_path, target_height=None, do_pad=False):
    ret = {}

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

    if target_height is None:
        target_height = cv2.getWindowImageRect("Image")[3]

    bev_scale_factor = target_height / arena_bev_height
    target_width = int(arena_bev_width * bev_scale_factor)
    target_corners = target_corners * bev_scale_factor
    M = cv2.getPerspectiveTransform(projected_arena_corners.astype(np.float32), target_corners.astype(np.float32))

    # Perform the transformation
    bev_image = cv2.warpPerspective(image, M, (target_width, target_height))

    # Flip the BEV image so that Y-axis points upwards
    bev_image = cv2.flip(bev_image, 0)

    ret['bev_image'] = bev_image

    # Resize the image so that its height is the same as BEV image
    image_scale_factor = target_height / image.shape[0]
    image = cv2.resize(image, (int(image_scale_factor * image.shape[1]), target_height))

    ret['image'] = image
    ret['bev_bots'] = []

    # Draw bot BEV bounding boxes on the BEV view
    for bot in metadata['bots']:
        ret['bev_bots'].append({})
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

        ret['bev_bots'][-1]['bot_points'] = bev_bot_points

        # Collect wheel coordinates
        wheel_points = []
        for wheel in bot['config']['wheels']:
            x = -width/2 - wheel['x_offset']
            y = wheel['y_coord']
            wheel_points.append([x, y, 0])

            x = width/2 + wheel['x_offset']
            wheel_points.append([x, y, 0])

        bev_wheel_points = bot_to_BEV(np.array(wheel_points), bot, arena_corners_bev, bev_scale_factor, target_height)
        ret['bev_bots'][-1]['wheel_points'] = bev_wheel_points

        forward = bot_to_BEV(np.array([[0, length/2, 0]]), bot, arena_corners_bev, bev_scale_factor, target_height)
        ret['bev_bots'][-1]['forward'] = forward

    if do_pad:
        # Ensure that the BEV image is square and the points are correctly scaled and offset

        # Calculate the scaling factor
        height, width = bev_image.shape[:2]
        if width > height:
            # The image will have to be scaled down, because the width is larger than the target width, which is the height
            pad_scale_factor = target_height / width

            # Scale the BEV image
            bev_image = cv2.resize(bev_image, (target_height, int(height * pad_scale_factor)))
        else:
            pad_scale_factor = 1.0  # No scaling is needed, the image already fits inside the square

        # Calculate the padding
        height, width = bev_image.shape[:2]
        required_width_padding = target_height - width
        required_height_padding = target_height - height

        # Only one of these should be non-zero
        assert required_width_padding * required_height_padding == 0
        # assert required_width_padding + required_height_padding > 0

        # Apply the padding to the BEV image, so that it becomes square
        left_pad_width = required_width_padding // 2
        left_pad_height = required_height_padding // 2
        right_pad_width = required_width_padding - left_pad_width
        right_pad_height = required_height_padding - left_pad_height
        ret['bev_image'] = cv2.copyMakeBorder(bev_image, left_pad_height, right_pad_height, left_pad_width, right_pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        assert ret['bev_image'].shape[0] == ret['bev_image'].shape[1] == target_height

        # Re-calculate the point coordinates
        for bot in ret['bev_bots']:
            bot['bot_points'] = bot['bot_points'].astype(np.float32)
            bot['bot_points'] *= pad_scale_factor
            bot['bot_points'] += np.array([left_pad_width, left_pad_height])
            bot['bot_points'] = bot['bot_points'].astype(np.int32)

            bot['wheel_points'] = bot['wheel_points'].astype(np.float32)
            bot['wheel_points'] *= pad_scale_factor
            bot['wheel_points'] += np.array([left_pad_width, left_pad_height])
            bot['wheel_points'] = bot['wheel_points'].astype(np.int32)

            bot['forward'] = bot['forward'].astype(np.float32)
            bot['forward'][0] *= pad_scale_factor
            bot['forward'][0] += np.array([left_pad_width, left_pad_height])
            bot['forward'] = bot['forward'].astype(np.int32)

    return ret

def render_element(image_path, metadata_path):
    # Load the data
    data = load_data(image_path, metadata_path, do_pad=True)
    image = data['image']
    bev_image = data['bev_image']

    # Draw bot BEV bounding boxes on the BEV view
    for bot in data['bev_bots']:
        bev_bot_points = bot['bot_points']

        # Draw the bot points
        for i in range(4):
            cv2.line(bev_image, tuple(bev_bot_points[i]), tuple(bev_bot_points[(i+1)%4]), (0, 255, 255), 2)

        bev_wheel_points = bot['wheel_points']

        # Draw the wheel points
        for i in range(len(bev_wheel_points)):
            cv2.circle(bev_image, tuple(bev_wheel_points[i]), 5, (0, 255, 255), -1)

        # Draw a blue arrow pointing forward from the bot's center
        forward = bot['forward']

        cv2.circle(bev_image, tuple(forward[0]), 5, (255, 255, 255), -1)

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
    parser.add_argument('-d', '--dataset_path', type=str, default='data/BB_SYNTH_DATA', help='Path to the dataset directory.')

    args = parser.parse_args()
    if os.path.exists(args.dataset_path):
        main(args.dataset_path)
    else:
        print(f"Dataset path '{args.dataset_path}' does not exist.")
        exit(1)