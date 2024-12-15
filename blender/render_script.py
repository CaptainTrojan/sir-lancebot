import bpy
from mathutils import *

D = bpy.data
C = bpy.context

# Adjust the camera position slightly
camera = D.objects['Camera']  # Replace 'Camera' if your camera has a different name
camera.location.x += 0.1  # Adjust as needed
camera.location.y += 0.1
camera.location.z += 0.1

# Specify the output path
output_path = "image.png"  # Replace with your desired output path

# Set the render output path
C.scene.render.filepath = output_path

# Render the scene
bpy.ops.render.render(write_still=True)

# Print success message
print(f"Rendered image saved to: {output_path}")