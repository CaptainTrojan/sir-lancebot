import os
import cv2
import argparse

def draw_annotations(image, annotations):
    for annotation in annotations:
        parts = annotation.split()
        x_center, y_center, bbox_width, bbox_height = map(float, parts[1:5])
        keypoints = [(float(parts[i]), float(parts[i+1])) for i in range(5, len(parts), 3)]

        x_min = int((x_center - bbox_width / 2) * 640)
        y_min = int((y_center - bbox_height / 2) * 640)
        x_max = int((x_center + bbox_width / 2) * 640)
        y_max = int((y_center + bbox_height / 2) * 640)

        # Draw bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Draw keypoints
        for x, y in keypoints:
            if x != 0 and y != 0:
                cv2.circle(image, (int(x * 640), int(y * 640)), 5, (255, 0, 0), -1)

    return image

def visualize_dataset(image_dir, label_dir):
    image_files = sorted(os.listdir(image_dir))
    label_files = sorted(os.listdir(label_dir))

    for image_file, label_file in zip(image_files, label_files):
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, label_file)

        # Read image
        image = cv2.imread(image_path)

        # Read annotations
        with open(label_path, 'r') as f:
            annotations = f.readlines()

        # Draw annotations on the image
        annotated_image = draw_annotations(image, annotations)

        # Display the image
        cv2.imshow("Annotated Image", annotated_image)
        key = cv2.waitKey(0)
        if key == 27:  # ESC key to exit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize YOLO dataset.")
    parser.add_argument("image_dir", type=str, help="Directory containing images.")
    parser.add_argument("label_dir", type=str, help="Directory containing YOLO annotations.")

    args = parser.parse_args()
    visualize_dataset(args.image_dir, args.label_dir)