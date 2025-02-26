import os
import shutil
import cv2
import numpy as np
import argparse

# Global variables
corners = []
bbox = []
keypoints = []
annotations = []
current_frame = None
bev_image = None
og_bev_image = None
arena_size = 1  # Arena size in meters
scaled_frame = None
scale_factor_for_video = None

def click_event_orig(event, x, y, flags, param):
    global corners, bbox, keypoints, current_frame, bev_image, annotations, scaled_frame, scale_factor_for_video

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(corners) < 4:
            cv2.circle(scaled_frame, (x, y), 5, (0, 0, 255), -1)

            x = int(x / scale_factor_for_video)
            y = int(y / scale_factor_for_video)

            corners.append((x, y))
            cv2.imshow("Frame", scaled_frame)
            if len(corners) == 4:
                transform_to_bev()

def click_event_bev(event, x, y, flags, param):
    global corners, bbox, keypoints, current_frame, bev_image, annotations, scaled_frame, scale_factor_for_video

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(bbox) < 2:
            bbox.append((x, y))
            if len(bbox) == 2:
                cv2.rectangle(bev_image, bbox[0], bbox[1], (0, 255, 0), 2)
                cv2.imshow("BEV", bev_image)
        elif len(keypoints) < 5:
            keypoints.append((x, y))
            cv2.circle(bev_image, (x, y), 5, (255, 0, 0), -1)
            cv2.imshow("BEV", bev_image)

def transform_to_bev():
    global corners, current_frame, bev_image, og_bev_image

    src_pts = np.array(corners, dtype="float32")
    dst_pts = np.array([[0, 0], [arena_size, 0], [arena_size, arena_size], [0, arena_size]], dtype="float32") * 640
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    bev_image = cv2.warpPerspective(current_frame, M, (640, 640))
    og_bev_image = bev_image.copy()
    cv2.imshow("BEV", bev_image)
    cv2.setMouseCallback("BEV", click_event_bev)

def save_annotation():
    global bbox, keypoints, annotations, bev_image, og_bev_image

    x_min, y_min = bbox[0]
    x_max, y_max = bbox[1]
    x_center = (x_min + x_max) / 2 / 640
    y_center = (y_min + y_max) / 2 / 640
    bbox_width = (x_max - x_min) / 640
    bbox_height = (y_max - y_min) / 640

    annotation = f"0 {x_center} {y_center} {bbox_width} {bbox_height}"
    for x, y in keypoints:
        annotation += f" {x / 640} {y / 640} 1"

    for _ in range(5 - len(keypoints)):
        annotation += " 0 0 0"

    annotations.append(annotation)
    bbox = []
    keypoints = []

    # Re-paint the BEV image - copy from the original and paint all existing annotations in black
    # so we can differentiate between new and existing annotations
    bev_image = og_bev_image.copy()
    for ann in annotations:
        ann = ann.split()
        x_center = float(ann[1]) * 640
        y_center = float(ann[2]) * 640
        bbox_width = float(ann[3]) * 640
        bbox_height = float(ann[4]) * 640
        x_min = int(x_center - bbox_width / 2)
        y_min = int(y_center - bbox_height / 2)
        x_max = int(x_center + bbox_width / 2)
        y_max = int(y_center + bbox_height / 2)
        cv2.rectangle(bev_image, (x_min, y_min), (x_max, y_max), (0, 0, 0), 2)
        for i in range(5, len(ann), 3):
            x = int(float(ann[i]) * 640)
            y = int(float(ann[i + 1]) * 640)
            cv2.circle(bev_image, (x, y), 5, (0, 0, 0), -1)

    cv2.imshow("BEV", bev_image)

def process_video(video_path, output_dir):
    global current_frame, annotations, scaled_frame, scale_factor_for_video, corners

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = frame.copy()
        height, width = frame.shape[:2]
        scale_factor_for_video = 800 / height
        scaled_frame = cv2.resize(frame, (int(width * scale_factor_for_video), 800))

        cv2.imshow("Frame", scaled_frame)
        cv2.setMouseCallback("Frame", click_event_orig)

        key = cv2.waitKey(0)
        while key == ord('s'):
            save_annotation()
            key = cv2.waitKey(0)

        if key == 27:  # ESC key to exit
            break
        elif key == 32:  # SPACE key to save annotations and move to next frame
            # Destroy the 'BEV' window if it exists
            cv2.destroyWindow("BEV")
            if annotations:
                save_annotations(video_path, frame_count, output_dir)
            annotations = []
            corners = []
            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

def save_annotations(video_path, frame_count, output_dir):
    global annotations, og_bev_image

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_image_path = os.path.join(output_dir, "images", f"{video_name}_{frame_count}.jpg")
    output_label_path = os.path.join(output_dir, "labels", f"{video_name}_{frame_count}.txt")

    cv2.imwrite(output_image_path, og_bev_image)
    with open(output_label_path, 'w') as f:
        f.write("\n".join(annotations))

def main(input_dir, output_dir):
    # shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    for file in os.listdir(input_dir):
        if file.lower().endswith(".mp4"):
            video_path = os.path.join(input_dir, file)
            process_video(video_path, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate DJI videos and convert to YOLO format.")
    parser.add_argument("input_dir", type=str, help="Directory containing .mp4 files.")
    parser.add_argument("output_dir", type=str, help="Directory to save annotated images and labels.")

    args = parser.parse_args()
    main(args.input_dir, args.output_dir)