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
import bmesh
import shapely as sp

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
    camera.location.z += random.uniform(0, 0.2)
    
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


def replace_array(orig, new, start, amount):
    if start < 0:
        start = len(orig) + start
    end = start + amount
    return orig[:start] + new + orig[end:]


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

def create_polygon_extrusion(bot, vertices, dimension, name):
    assert dimension in ['X', 'Y', 'Z'], f"Invalid dimension {dimension}"

    if dimension == "X":
        vertices = [(-bot['width']/2, v[0], v[1]) for v in vertices]
    elif dimension == "Y":
        vertices = [(v[0], -bot['length']/2, v[1]) for v in vertices]
    else:
        vertices = [(v[0], v[1], 0) for v in vertices]

    # Create a new mesh and object for the polygon
    mesh = bpy.data.meshes.new(name=f"{name}Mesh")
    obj = bpy.data.objects.new(name=name, object_data=mesh)

    # Link the object to the scene
    bpy.context.collection.objects.link(obj)

    # Create a bmesh object and add the polygon vertices
    bm = bmesh.new()
    verts = [bm.verts.new(v) for v in vertices]
    bm.verts.ensure_lookup_table()

    # Create a face from the polygon vertices
    face = bm.faces.new(verts)

    # Extrude the polygon across the specified dimension
    extrude_result = bmesh.ops.extrude_face_region(bm, geom=[face])
    extruded_verts = [v for v in extrude_result['geom'] if isinstance(v, bmesh.types.BMVert)]
    if dimension == "X":
        bmesh.ops.translate(bm, vec=(bot['width'], 0, 0), verts=extruded_verts)
    elif dimension == "Y":
        bmesh.ops.translate(bm, vec=(0, bot['length'], 0), verts=extruded_verts)
    else:
        bmesh.ops.translate(bm, vec=(0, 0, bot['height']), verts=extruded_verts)
    
    # Update the mesh with the new geometry
    bm.to_mesh(mesh)
    bm.free()

    return obj

def join_and_intersect(A_obj, B_obj):
    # Join the two objects
    join([A_obj, B_obj])
    # Enter edit mode
    bpy.ops.object.mode_set(mode='EDIT')
    # Select any vertex
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')
    A_obj.data.vertices[0].select = True
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_linked(delimit=set())
    # Apply the boolean intersection
    bpy.ops.mesh.intersect_boolean(operation='INTERSECT', solver='EXACT')
    bpy.ops.object.mode_set(mode='OBJECT')

def join(objs):
    # Join the objects
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.join()

def generate_random_bot_config(bots: list, corner_coords):
    min_width = 0.08
    max_width = 0.12
    min_length = 0.08
    max_length = 0.12
    min_height = 0.02
    max_height = 0.05
    corner_margin = 0.1

    def is_too_close(bot, bots, min_distance=0.1):
        for existing_bot in bots:
            distance = np.sqrt((bot['x'] - existing_bot['x'])**2 + (bot['y'] - existing_bot['y'])**2)
            if distance < min_distance:
                return True
        return False

    bot = {}
    while True:
        bot['x'] = random.uniform(corner_coords[0][0] + corner_margin, corner_coords[1][0] - corner_margin)
        bot['y'] = random.uniform(corner_coords[0][1] + corner_margin, corner_coords[2][1] - corner_margin)
        if not is_too_close(bot, bots):
            break
    bot['width'] = random.uniform(min_width, max_width)
    bot['length'] = random.uniform(min_length, max_length)
    bot['height'] = random.uniform(min_height, max_height)
    bot['theta'] = random.uniform(0, 2*np.pi)

    # Generate random front configuration
    bot_type_options = ["pusher", "vertical spinner", "horizontal spinner"]
    bot_type = random.choice(bot_type_options)

    # Generate random wheel configurations
    num_wheels = random.choice([1, 2]) if bot_type != "horizontal spinner" else 1
    wheels = []
    for i in range(num_wheels):
        y_coord = random.uniform(-bot['length']*0.4, -bot['length']*0.25) if i == 0 else random.uniform(bot['length']*0.25, bot['length']*0.4)
        x_coord = random.uniform(-bot['width']*0.25, bot['width']*0.25)
        radius = random.uniform(bot['height']*0.4, bot['height']*0.75)
        if i == 0:
            axle_height = random.uniform(bot['height']*0.1, bot['height']*0.7)
            bot['z'] = radius - axle_height
        else:
            # Ensure the bot is aligned with the ground
            tries = 0
            axle_height = radius - bot['z']
            while axle_height < bot['height']*0.1 or axle_height > bot['height']*0.7:
                # Re-genereate the radius and retry. We know this must succeed, because the first wheel also succeeded
                radius = random.uniform(bot['height']*0.4, bot['height']*0.75)
                axle_height = radius - bot['z']

                tries += 1

                # If too many tries, just set the same radius as the first wheel
                if tries > 10:
                    radius = wheels[0]['radius']
                    axle_height = radius - bot['z']
                    break

        thickness = random.uniform(bot['width']*0.1, bot['width']*0.2)
        darkness = random.uniform(0, 0.1)
        wheels.append({
            'y_coord': y_coord,
            'x_offset': x_coord,
            'axle_height': axle_height,
            'radius': radius,
            'thickness': thickness,
            'darkness': darkness
        })

    # Return the bot configuration as a dictionary
    bot['config'] = {
        'wheels': wheels,
        'wheels_covered': False, # random.choice([True, False]),
        'type': bot_type
    }

    bots.append(bot)

def create_wheel_object(bot_width, wheel, sign):
    axle_length = (bot_width + wheel['x_offset']) * 0.5
    x = sign * (axle_length)
    y = wheel['y_coord']
    radius = wheel['radius']
    thickness = wheel['thickness']
    axle_height = wheel['axle_height']

    # Create a cylinder mesh for the wheel
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=thickness, location=(x, y, axle_height))
    wheel_obj = bpy.context.object
    wheel_obj.name = "Wheel"

    # Rotate the wheel to have the X axis as the rotation axis
    wheel_obj.rotation_euler = (0, np.pi / 2, 0)

    # Add the "Tires" material to the wheel
    base_material = bpy.data.materials.get("Tires")
    tire_material = base_material.copy()
    # Find the Principled BSDF node and vary the color
    bsdf = tire_material.node_tree.nodes.get("Principled BSDF")
    color = wheel['darkness']
    bsdf.inputs['Base Color'].default_value = (color, color, color, 1)
    wheel_obj.data.materials.append(tire_material)

    x_sign = 1 if x > 0 else -1

    # Add the axle to the wheel
    bpy.ops.mesh.primitive_cylinder_add(radius=0.005, depth=axle_length, location=(x - x_sign * axle_length/2, y, axle_height))
    axle_obj = bpy.context.object
    axle_obj.name = "Axle"
    axle_obj.rotation_euler = (0, np.pi / 2, 0)

    # Join the wheel and the axle
    join([wheel_obj, axle_obj])

    # Add the wheel to the Bots collection
    bots_collection = bpy.data.collections.get('Bots')
    if bots_collection:
        bots_collection.objects.link(wheel_obj)

    return wheel_obj

def create_wheels(bot):
    wheel_objs = []
    for wheel in bot['config']['wheels']:
        # Left wheel
        wheel_obj = create_wheel_object(bot['width'], wheel, -1)
        wheel_objs.append(wheel_obj)

        # Right wheel
        wheel_obj = create_wheel_object(bot['width'], wheel, 1)
        wheel_objs.append(wheel_obj)
    return wheel_objs

def create_initial_bot_mesh(bot):
    # Y break point must come after the last wheel
    last_wheel = bot['config']['wheels'][-1]
    last_wheel_y = last_wheel['y_coord']
    last_wheel_r = last_wheel['radius']
    y_break_point = random.uniform(last_wheel_y + last_wheel_r, bot['length']/2)    

    # Add a big elbow only if the bot is a horizontal spinner
    should_add_elbow = bot['config']['type'] == "horizontal spinner"
    elbow_height = bot['height'] * 0.9

    side_view_vertices = [
        (-bot['length']/2, bot['height']),
        (-bot['length']/2, 0),
        (bot['length']/2, 0),
        (y_break_point, bot['height'])
    ]

    if should_add_elbow:
        side_view_vertices.insert(-1, (y_break_point, bot['height'] - elbow_height))
        side_view_vertices.insert(-1, (bot['length']/2, bot['height'] - elbow_height))

    return create_polygon_extrusion(bot, side_view_vertices, "X", "Bot")

def sp_diff_wheel(polygon, wheel, bot_width, sign):
    # Create a wheel polygon
    wheel_center = (sign * (wheel['x_offset'] + bot_width/2), wheel['y_coord'])
    wheel_polygon = sp.Polygon([
        (wheel_center[0] + sign, wheel_center[1] + wheel['radius']),
        (wheel_center[0] + sign, wheel_center[1] - wheel['radius']),
        (wheel_center[0] - sign * wheel['thickness']/2, wheel_center[1] - wheel['radius']),
        (wheel_center[0] - sign * wheel['thickness']/2, wheel_center[1] + wheel['radius'])
    ])

    # Subtract the wheel polygon from the main polygon
    return polygon.difference(wheel_polygon)

def create_bev_cutting_polygon(bot):
    polygon_verts = [
        (-bot['width']/2, bot['length']/2),     # Front left
        (-bot['width']/2, -bot['length']/2),    # Back left
        (bot['width']/2, -bot['length']/2),     # Back right
        (bot['width']/2, bot['length']/2)       # Front right
    ]
    polygon = sp.Polygon(polygon_verts)

    if not bot['config']['wheels_covered']:
        # If wheels are not covered and they are inset, we must add a cutout for each
        for wheel in bot['config']['wheels']:
            polygon = sp_diff_wheel(polygon, wheel, bot['width'], -1) # Left wheel
            polygon = sp_diff_wheel(polygon, wheel, bot['width'], 1)  # Right wheel

    polygon_verts = list(polygon.exterior.coords)

    # Create the cutting polygon in 2D (X, Y)
    return create_polygon_extrusion(bot, polygon_verts, "Z", "BEVCutPolygon")

def create_front_cutting_polygon(bot):
    # Create the cutting polygon in 2D (X, Z) (leave as-is for now)
    cutting_polygon_verts = [
        (-bot['width']/2, 0),
        (bot['width']/2, 0),
        (bot['width']/2, bot['height']),
        (-bot['width']/2, bot['height'])
    ]
    return create_polygon_extrusion(bot, cutting_polygon_verts, "Y", "FrontCutPolygon")

def create_bot_object(bot):
    # Create the bot mesh
    bot_obj = create_initial_bot_mesh(bot)

    # Create the cutting polygon from BEV
    cut_obj = create_bev_cutting_polygon(bot)

    # Apply the boolean intersection
    join_and_intersect(bot_obj, cut_obj)

    # Create the cutting polygon from front view
    cut_obj = create_front_cutting_polygon(bot)

    # Apply the boolean intersection
    join_and_intersect(bot_obj, cut_obj)

    # Create the wheels
    wheel_objs = create_wheels(bot)
    for wheel_obj in wheel_objs:
        wheel_obj.parent = bot_obj

    return bot_obj

def add_bot_to_scene(bot, bots_collection, i):
    # Create the bot mesh
    bot_obj = create_bot_object(bot)

    # Set the bot's location and rotation
    bot_obj.location = (bot['x'], bot['y'], bot['z'])
    bot_obj.rotation_euler = (0, 0, bot['theta'])
    O.object.transform_apply(location=True, rotation=True, scale=False)

    # Apply the material to the bot
    base_material = bpy.data.materials.get("PLA")
    material = base_material.copy()
    material.name = f"PLA_{i}"
    bot_obj.data.materials.append(material)
    material.use_nodes = True
    color_ramp = material.node_tree.nodes.get("Color Ramp")
    color_ramp.color_ramp.elements[1].color = (
        random.choice([random.uniform(0, 0.2), random.uniform(0.8, 1)]),
        random.choice([random.uniform(0, 0.2), random.uniform(0.8, 1)]),
        random.choice([random.uniform(0, 0.2), random.uniform(0.8, 1)]),
        1
    )

    # Name the bot
    bot_obj.name = f"Bot_{i}"

    # Add the bot to the collection
    bots_collection.objects.link(bot_obj)


def generate_bots(corner_coords):
    COUNT = random.randint(3, 6)
    # COUNT = 1

    bots = []
    
    # Get the Bots collection (create it if it doesn't exist)
    if 'Bots' not in D.collections:
        bots_collection = D.collections.new(name='Bots')
    else:
        bots_collection = D.collections['Bots']

    for i in range(COUNT):
        generate_random_bot_config(bots, corner_coords)
        add_bot_to_scene(bots[-1], bots_collection, i)

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

    # Set seed
    random.seed(1)

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