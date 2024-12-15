from functools import cache
import json
import random
import bpy
from mathutils import *
import sys
from tqdm import tqdm
import os
import shutil
import uuid
import bpy_extras.object_utils
import numpy as np

D = bpy.data
C = bpy.context
O = bpy.ops


#---------------------------------------------------------------
# 3x4 P matrix from Blender camera
#---------------------------------------------------------------

# BKE_camera_sensor_size
def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in 
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
# as well as
# https://blender.stackexchange.com/a/120063/3581
def get_calibration_matrix_K_from_blender(camd):
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((s_u, skew, u_0),
        (   0,  s_v, v_0),
        (   0,    0,   1)))
    return K

# Returns camera rotation and translation matrices from Blender.
# 
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates 
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
        (0, -1, 0),
        (0, 0, -1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
        ))
    return RT

def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K@RT, K, RT
# ----------------------------------------------------------
# Alternate 3D coordinates to 2D pixel coordinate projection code
# adapted from https://blender.stackexchange.com/questions/882/how-to-find-image-coordinates-of-the-rendered-vertex?lq=1
# to have the y axes pointing up and origin at the top-left corner
def project_by_object_utils(cam, point):
    scene = bpy.context.scene
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, point)
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
            int(scene.render.resolution_x * render_scale),
            int(scene.render.resolution_y * render_scale),
            )
    return Vector((co_2d.x * render_size[0], render_size[1] - co_2d.y * render_size[1]))

def randomize_camera(camera):
    # Randomize the camera position
    camera.location.x += random.uniform(-0.2, 0.2)
    camera.location.y += random.uniform(-0.2, 0)
    camera.location.z += random.uniform(-0.2, 0.2)
    
    # Make the camera look at the point (0,0,0) with a deviation of ±0.1
    target_x = random.uniform(-0.1, 0.1)
    target_y = random.uniform(-0.1, 0.1)
    target_z = 0
    
    direction = D.objects.new("Empty", None)
    direction.location = (target_x, target_y, target_z)
    C.collection.objects.link(direction)
    
    camera.constraints.new(type='TRACK_TO')
    camera.constraints['Track To'].target = direction
    camera.constraints['Track To'].track_axis = 'TRACK_NEGATIVE_Z'
    camera.constraints['Track To'].up_axis = 'UP_Y'
    O.object.transform_apply(location=True, rotation=True, scale=True)

@cache
def get_vertex_group_vertices(obj, group_name):
    group_index = obj.vertex_groups[group_name].index
    vertices = [v for v in obj.data.vertices if group_index in [g.group for g in v.groups]]
    return vertices

def generate_and_render_image(image_dir: str, metadata_dir: str):
    metadata = {}

    # Randomize the camera position
    camera = D.objects['Camera']
    randomize_camera(camera)

    # Save the camera projection matrix
    metadata['camera_P'] = np.array(get_3x4_P_matrix_from_blender(camera)[0]).tolist()

    # Find arena corners
    arena = D.objects['Arena']
    corners = get_vertex_group_vertices(arena, 'Corners')
    
    # Save the arena corner 3D coordinates
    metadata['arena_corners'] = [list(arena.matrix_world @ c.co) for c in corners]
    # Save the arena corner 2D coordinates in the metadata
    # metadata['arena_corners_2d'] = np.array([project_by_object_utils(camera, arena.matrix_world @ c.co) for c in corners]).tolist()

    # Set the render output path and render the image
    element_id = str(uuid.uuid4())
    output_path = os.path.join(image_dir, f"{element_id}.png")
    C.scene.render.filepath = output_path
    # Reduce the amount of samples for faster rendering
    C.scene.eevee.taa_render_samples = 10
    O.render.render(write_still=True)

    # Save the metadata
    metadata_path = os.path.join(metadata_dir, f"{element_id}.json")
    with open(metadata_path, 'w') as f:
        f.write(json.dumps(metadata, indent=4))

if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:]
    num_images = int(args[0])

    # Clear the output directory
    output_dir = "BB_SYNTH_DATA"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir)

    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(metadata_dir)

    for i in tqdm(range(num_images), desc="Rendering images"):
        generate_and_render_image(image_dir, metadata_dir)