import argparse
import os
import json
import shutil
import cv2
import numpy as np
from view_element import load_data
from tqdm import tqdm
import yaml

def convert_to_yolo_format(dataset_path, output_path, val_split_size=0.1):
    image_dir = os.path.join(dataset_path, "images")
    metadata_dir = os.path.join(dataset_path, "metadata")
    target_size = 640

    image_files = sorted(os.listdir(image_dir))
    metadata_files = sorted(os.listdir(metadata_dir))

    if len(image_files) == 0:
        print(f"No images found in '{image_dir}'.")
        return

    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labels", "val"), exist_ok=True)

    # Export the yaml to the root of the output directory
    with open(os.path.join(output_path, "bb_synth.yaml"), 'w') as f:
        config = {
            "path": os.path.abspath(output_path),
            "train": "images/train",
            "val": "images/val",

            "nc": 1,
            "names": ["battlebot"],

            "kpt_shape": [5, 3],
            "flip_idx": [0, 2, 1, 4, 3],
        }
        yaml.safe_dump(config, f)

    # Split the data into training and validation sets
    num_val = int(len(image_files) * val_split_size)

    for idx, (image_file, metadata_file) in enumerate(tqdm(zip(image_files, metadata_files), total=len(image_files), desc="Processing images")):
        if idx < num_val:
            split = "val"
        else:
            split = "train"
        
        image_path = os.path.join(image_dir, image_file)
        metadata_path = os.path.join(metadata_dir, metadata_file)

        # Generate BEV image
        ret = load_data(image_path, metadata_path, target_height=target_size, do_pad=True)
        bev_image = ret['bev_image']

        # Save the padded BEV image
        output_image_path = os.path.join(output_path, "images", split, image_file)
        cv2.imwrite(output_image_path, bev_image)

        # Extract bounding boxes and keypoints
        annotations = []
        for bot in ret['bev_bots']:
            # Transform the bot points to BEV
            bev_bot_points = bot['bot_points']

            # Calculate the bounding box
            x_min = np.min(bev_bot_points[:, 0])
            y_min = np.min(bev_bot_points[:, 1])
            x_max = np.max(bev_bot_points[:, 0])
            y_max = np.max(bev_bot_points[:, 1])

            # Normalize the bounding box coordinates
            x_center = (x_min + x_max) / 2 / target_size
            y_center = (y_min + y_max) / 2 / target_size
            bbox_width = (x_max - x_min) / target_size
            bbox_height = (y_max - y_min) / target_size

            # Create the annotation string
            annotation = f"0 {x_center} {y_center} {bbox_width} {bbox_height}"

            # Transform the front point to BEV
            bev_forward = bot['forward']

            # Normalize the front point coordinates
            forward_x = bev_forward[0][0] / target_size
            forward_y = bev_forward[0][1] / target_size

            # Add the front point to the annotation
            annotation += f" {forward_x} {forward_y} 1"

            # Transform the wheel points to BEV
            bev_wheel_points = bot['wheel_points']

            # Normalize the wheel coordinates
            normalized_wheel_points = [(x / target_size, y / target_size) for x, y in bev_wheel_points[:, :2]]
            for x, y in normalized_wheel_points:
                annotation += f" {x} {y} 1"

            # Add -1s for missing wheels
            for _ in range(4 - len(normalized_wheel_points)):
                annotation += " 0 0 0"

            annotations.append(annotation)

        # Save the annotations
        output_label_path = os.path.join(output_path, "labels", split, os.path.splitext(image_file)[0] + ".txt")
        with open(output_label_path, 'w') as f:
            f.write("\n".join(annotations))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert raw synth data to YOLO format.')
    parser.add_argument('input_path', type=str, help='Path to the synth data directory.')
    parser.add_argument('output_path', type=str, help='Path to the output directory.')

    args = parser.parse_args()
    if os.path.exists(args.input_path):
        convert_to_yolo_format(args.input_path, args.output_path)
    else:
        print(f"Input path '{args.input_path}' does not exist.")
        exit(1)