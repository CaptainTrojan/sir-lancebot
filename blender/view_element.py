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
    return projected_points

def transform_blender_to_opencv(coords):
    # Transform Blender coordinates to OpenCV coordinates
    transform_matrix = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])
    return np.dot(transform_matrix, coords)

def render_image_with_corners(image_path, metadata_path):
    # Load image
    image = cv2.imread(image_path)

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Camera P matrix
    P = np.array(metadata['camera_P'])

    # Arena corners
    arena_corners = np.array(metadata['arena_corners'])
    # arena_corners = np.array([transform_blender_to_opencv(corner) for corner in arena_corners])

    # Project 3D points to 2D
    projected_points = project_points(arena_corners, P)

    # Project [0, 0, 0] to 2D
    origin = np.array([[0.0, 0.0, 0.0]])
    projected_origin = project_points(origin, P)
    
    # print("=" * 50)
    # print("Point 3D coordinates:")
    # print(arena_corners)
    # print("Projected 2D coordinates:")
    # print(projected_points)
    # print("Projected origin:")
    # print(projected_origin)

    # Draw points on the image
    for point in projected_points:
        x, y = point
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    # Draw the origin
    x, y = projected_origin[0]
    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

    # cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

    # Display the image
    cv2.imshow('Image with Corners', image)

def main(dataset_path):
    image_dir = os.path.join(dataset_path, "images")
    metadata_dir = os.path.join(dataset_path, "metadata")

    image_files = sorted(os.listdir(image_dir))
    metadata_files = sorted(os.listdir(metadata_dir))

    if len(image_files) == 0:
        print(f"No images found in '{image_dir}'.")
        return

    index = 0
    while True:
        image_path = os.path.join(image_dir, image_files[index])
        metadata_path = os.path.join(metadata_dir, metadata_files[index])
        render_image_with_corners(image_path, metadata_path)

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