from copy import deepcopy
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


def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x


def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit


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


def get_3x4_RT_matrix_from_blender(cam):
    R_bcam2cv = Matrix(
        ((1, 0,  0),
        (0, -1, 0),
        (0, 0, -1)))

    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()
  
    T_world2bcam = -1*R_world2bcam @ location

    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

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
    # target_x = 0
    # target_y = 0
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


def randomize_arena(arena):
    # Find the arena corners
    corners = get_vertex_group_vertices(arena, 'Corners')
    dl = random.uniform(-0.2, 0.2)
    dw = random.uniform(-0.2, 0.2)
    arena["arena_dl"] = dl
    arena["arena_dw"] = dw
    # update the arena
    O.object.transform_apply(location=True, rotation=True, scale=True, properties=True)
    # Assert the arena world matrix is still eye
    assert arena.matrix_world == Matrix.Identity(4), f"Expected arena matrix to be identity {arena.matrix_world}"
    # Collect the corner coordinates and update them by dw and dl
    corner_coords = []
    og_x_span = abs(corners[0].co.x)
    og_y_span = abs(corners[0].co.y)
    for c in corners:
        co = np.array(c.co)
        x_factor = (og_x_span + dw*0.5) / og_x_span
        y_factor = (og_y_span + dl*0.5) / og_y_span
        co[0] *= x_factor
        co[1] *= y_factor
        corner_coords.append(co)
    return corner_coords


def generate_bots(corner_coords):
    mean_width = 0.1
    mean_length = 0.1
    mean_height = 0.035
    std_width = 0.02
    std_length = 0.02
    std_height = 0.005
    corner_margin = 0.1
    COUNT = random.randint(3, 6)

    bots = []
    def is_too_close(bot, bots, min_distance=0.1):
        for existing_bot in bots:
            distance = np.sqrt((bot['x'] - existing_bot['x'])**2 + (bot['y'] - existing_bot['y'])**2)
            if distance < min_distance:
                return True
        return False
    
    # Get the Bots collection (create it if it doesn't exist)
    if 'Bots' not in D.collections:
        bots_collection = D.collections.new(name='Bots')
    else:
        bots_collection = D.collections['Bots']

    for i in range(COUNT):
        bot = {}
        while True:
            bot['x'] = random.uniform(corner_coords[0][0] + corner_margin, corner_coords[1][0] - corner_margin)
            bot['y'] = random.uniform(corner_coords[0][1] + corner_margin, corner_coords[2][1] - corner_margin)
            if not is_too_close(bot, bots):
                break
        bot['z'] = random.uniform(0, 0.01)
        bot['width'] = random.gauss(mean_width, std_width)
        bot['length'] = random.gauss(mean_length, std_length)
        bot['height'] = random.gauss(mean_height, std_height)
        bot['theta'] = random.uniform(0, 2*np.pi)
        bots.append(bot)
        
        # Add a cube with the bot's dimensions to the scene
        bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, align='WORLD', location=(bot['x'], bot['y'], bot['z'] + bot['height']/2))
        bot_cube = bpy.context.active_object
        bot_cube.scale = (bot['width'], bot['length'], bot['height'])
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        bot_cube.rotation_euler = (0, 0, bot['theta'])
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        bot_cube.name = f"Bot_{i}"
        # Add the bot to the collection
        bots_collection.objects.link(bot_cube)

    return bots

def generate_and_render_image(i, image_dir: str, metadata_dir: str):
    metadata = {}

    # Randomize the camera position
    camera = D.objects['Camera']
    camera_loc = deepcopy(camera.location)
    randomize_camera(camera)

    # Save the camera projection matrix
    metadata['camera_P'] = np.array(get_3x4_P_matrix_from_blender(camera)[0]).tolist()

    # Manipulate the arena
    arena = D.objects['Arena']
    corner_coords = randomize_arena(arena)

    # Save the arena corner 3D coordinates
    metadata['arena_corners'] = np.array(corner_coords).tolist()

    # Generate a bunch of bots
    bots = generate_bots(corner_coords)
    # Save the bots
    metadata['bots'] = bots

    # Set the render output path and render the image
    element_id = f"{i}_{uuid.uuid4()}"
    output_path = os.path.join(image_dir, f"{element_id}.png")
    C.scene.render.filepath = output_path
    # Reduce the amount of samples for faster rendering
    C.scene.eevee.taa_render_samples = 10
    O.render.render(write_still=True)

    # Reset everything
    camera.location = camera_loc
    O.object.transform_apply(location=True, rotation=True, scale=True)
    # Delete the bots from the scene
    # print(D.collections)
    bots_collection = D.collections['Bots']
    with bpy.context.temp_override(selected_objects=bots_collection.objects):
        # print(bots_collection.objects)
        O.object.delete()

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
        generate_and_render_image(i, image_dir, metadata_dir)