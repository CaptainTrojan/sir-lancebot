import numpy as np
import cv2
import bpy

P = np.loadtxt("projection_matrix.txt")

# Define the points
e1 = np.array([-6.11270189e-01, 0.00000000e+00, -6.52571499e-01, 1])
e2 = np.array([6.11270428e-01, 5.96046448e-08, -6.52571499e-01, 1])
e3 = np.array([-6.11270189e-01, 0.00000000e+00, 6.52571559e-01, 1])
e4 = np.array([6.11270428e-01, 5.96046448e-08, 6.52571559e-01, 1])
# Switch the Y and Z axes
flip_M = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])
e1 = flip_M @ e1
e2 = flip_M @ e2
e3 = flip_M @ e3
e4 = flip_M @ e4
O = np.array([0.0, 0.0, 0.0, 1])

# Project the points
points = [e1, e2, e3, e4, O]
projected_points = []

for point in points:
    projected = P @ point
    projected /= projected[2]
    projected_points.append((int(projected[0]), int(projected[1])))

# Load the image
image_path = "/home/captaintrojan/Projects/sir-lancebot/blender/untitled.png"
image = cv2.imread(image_path)

# Draw the points
for point in projected_points:
    cv2.circle(image, point, 5, (0, 0, 255), -1)
# Save the image
output_path = "/home/captaintrojan/Projects/sir-lancebot/blender/untitled2.png"
cv2.imwrite(output_path, image)
