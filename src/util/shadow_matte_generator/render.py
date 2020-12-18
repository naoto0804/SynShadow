# ./blender-2.81a-linux-glibc217-x86_64/blender -b -P render.py

import argparse
import math
import random
from itertools import cycle
from pathlib import Path

import numpy as np
from numpy.random import uniform

import bpy

LIGHT_MIN, LIGHT_MAX = 0.1, 1.5
SCALE_MIN, SCALE_MAX = 1.0, 4.0
# SHIFT_MAX = 1.25  # for default(640x480)
SHIFT_MAX = 1.5  # for 640x640
MESH_LOC = [0, 0, 10]
CAMERA_LOC = [0, 0, 5]
random.seed(0)
np.random.seed(0)

output_root = Path("data/640x640_light_0p1_1p5/raw")
output_root.mkdir(parents=True, exist_ok=True)

with Path("human_mesh_paths.txt").open(mode='r') as fd:
    file_names = [t.strip() for t in fd.readlines()]
    human_mesh_iter = cycle(random.sample(file_names, len(file_names)))
with Path("object_mesh_paths.txt").open(mode='r') as fd:
    file_names = [t.strip() for t in fd.readlines()]
    object_mesh_iter = cycle(random.sample(file_names, len(file_names)))

# Global setup for blender
# Set frequently used chunk
C = bpy.context
D = bpy.data
O = bpy.ops

# Render
C.scene.render.engine = 'CYCLES'
C.scene.render.film_transparent = True  # change background as transparent
# C.scene.render.image_settings.file_format = 'PNG'
C.scene.render.image_settings.color_depth = '16'
C.scene.render.image_settings.file_format = 'OPEN_EXR'
C.scene.render.resolution_x = 640
C.scene.render.resolution_y = 640


def add_camera():
    O.object.camera_add(
        enter_editmode=False, align='VIEW', location=CAMERA_LOC)


def add_human(human_mesh_iter):
    filepath = str(next(human_mesh_iter))
    print(f"Loading human mesh {filepath} ..")
    O.import_mesh.stl(filepath=filepath)
    obj = C.active_object
    obj.location = \
        uniform(-SHIFT_MAX, SHIFT_MAX, size=(3, )) + MESH_LOC
    obj.scale = uniform(SCALE_MIN, SCALE_MAX, size=(1, )).repeat(3)
    obj.rotation_euler = [
        0,
        math.pi * uniform(low=-1.0, high=1.0),  # along y axis (plus: upright)
        math.pi * uniform(low=-0.25, high=0.25)  # along z axis (plus: front)
    ]


def add_object(object_mesh_iter):
    filepath = str(next(object_mesh_iter))
    print(f"Loading object mesh {filepath} ..")
    O.import_scene.obj(filepath=filepath)
    obj = D.objects[C.selected_objects[0].name]
    obj.location = uniform(-SHIFT_MAX, SHIFT_MAX, size=(3, )) + MESH_LOC
    obj.scale = uniform(SCALE_MIN, SCALE_MAX, size=(1, )).repeat(3)
    obj.rotation_euler = math.pi * uniform(-1.0, 1.0, size=(3, ))


def add_light():
    # light
    # O.object.light_add(type='SUN')
    radius = uniform(LIGHT_MIN, LIGHT_MAX)
    O.object.light_add(type='POINT', radius=radius, location=(0, 0, 50))
    C.active_object.data.energy = 10000  # set very strong light source

    # HDR Image Based Lighting
    # https://blender.stackexchange.com/questions/132271/blender-2-8-get-environment-texture-path-from-ui-to-python-script
    # world = C.scene.world
    # world.use_nodes = True
    # enode = world.node_tree.nodes['Environment Texture']
    # enode.image = D.images.load("//../../Downloads/urban_street_04_1k.hdr")


def add_shadow_catch_plane():
    # dummy plane for casting shadow
    O.mesh.primitive_plane_add(
        size=100, enter_editmode=False, location=(0, 0, 0))
    # angle_min, angle_max = -0.375, 0.125
    # angle = angle_min + np.random.random() * (angle_max - angle_min)
    # C.active_object.rotation_euler[0] = math.pi * angle
    C.active_object.cycles.is_shadow_catcher = True


def reset_scene():
    # diff_obj_names = list(set(bpy.data.objects.keys()) - set(base_objects))
    # if len(diff_obj_names) > 0:
    #     for obj_name in diff_obj_names:
    #         D.objects.remove(bpy.data.objects[obj_name])
    # Initialization
    O.object.select_all(action="SELECT")
    O.object.delete(use_global=False)

    data = [bpy.data.meshes, bpy.data.materials, bpy.data.textures,
            bpy.data.images]
    for d in data:
        for block in d:
            if block.users == 0:
                d.remove(block)


def render(human_mesh_iter, object_mesh_iter):
    reset_scene()
    add_camera()
    add_shadow_catch_plane()
    add_light()

    if np.random.random() > 0.5:
        add_human(human_mesh_iter)  # Assume one person at max
    else:
        add_object(object_mesh_iter)
    if np.random.random() > 0.5:
        add_object(object_mesh_iter)

    # Render
    random_id = np.random.randint(1e12)
    # C.scene.render.filepath = str(output_root / f"{random_id:012}.png")
    C.scene.render.filepath = str(output_root / f"{random_id:012}.exr")
    C.scene.camera = D.objects['Camera']
    O.render.render(write_still=True)
    print(C.scene.render.filepath)


for _ in range(10000):
    render(human_mesh_iter, object_mesh_iter)
