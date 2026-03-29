import bpy
import random
import math
import mitsuba as mi
import numpy as np
from mathutils import Vector
import sys
import os
#randomizes the position and orientation of an object in the scene and that of the tx and rx
def fix_all_mesh_normals():
    """Fix normals for all mesh objects in the scene"""
    
    fixed_count = 0
    
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            # Select object
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            
            # Fix normals
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.object.mode_set(mode='OBJECT')
            
            fixed_count += 1
            print(f"✓ Fixed normals: {obj.name}")
    
    print(f"\n{'='*50}")
    print(f"✓ Fixed {fixed_count} mesh objects")
    print(f"{'='*50}")


###################################################
def get_bounding_box_2d(obj):
    """Get the 2D bounding box (min/max X and Y) of an object"""
    # Get world matrix to transform local coordinates to world coordinates
    mat = obj.matrix_world

    
    # Get bounding box corners in world space
    bbox_corners = [mat @ Vector(corner) for corner in obj.bound_box]
    
    # Extract X and Y coordinates
    x_coords = [corner.x for corner in bbox_corners]
    y_coords = [corner.y for corner in bbox_corners]
    
    return {
        'min_x': min(x_coords),
        'max_x': max(x_coords),
        'min_y': min(y_coords),
        'max_y': max(y_coords),
        'min_z': 1,
        'max_z': 5
    }

def check_collision_2d(bbox1, bbox2, margin=.01):
    """Check if two 2D bounding boxes overlap (with margin for spacing)"""
    # Add margin to create spacing between objects
    return not (bbox1['max_x'] + margin < bbox2['min_x'] or
                bbox1['min_x'] - margin > bbox2['max_x'] or
                bbox1['max_y'] + margin < bbox2['min_y'] or
                bbox1['min_y'] - margin > bbox2['max_y'])
    
def place_antenna_randomly(floor_name, max_attempts=1000, margin=0.5, 
                        z_rotation_range=(0, 360),size=[1,1,1],name="TX"):
    """
    Place antenna randomly on the floor without overlapping other objects
    
    Parameters:
    - name: name of the antenna object
    - floor_name: name of the floor object to determine bounds
    - max_attempts: maximum number of random placement attempts
    - margin: minimum distance between objects (in Blender units)
    - z_rotation_range: tuple of (min, max) rotation in degrees for Z axis only
    """
    
    bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, size[2]/2))

    new_object = bpy.context.active_object
    new_object.name = name
    new_object.dimensions = (size[0], size[1], size[2]) # Example: 5 units wide, 2 deep, 10 high

# Optional: Apply scale to reset scale values to 1 (useful for modifiers)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    object_target = bpy.data.objects.get(name)
    if not object_target:
        print(f"ERROR: antenna '{name}' not found!")
        return False

    floor = bpy.data.objects.get(floor_name)
    ##############################
    if not floor:
        print(f"ERROR: Floor '{floor_name}' not found!")
        return False
    
    print(f"✓ Found floor: {floor_name}")
    
    # Set rotation mode to Euler
    object_target.rotation_mode = 'XYZ'
    
    # Get floor bounds
    floor_bbox = get_bounding_box_2d(floor)
    print(f"✓ Floor bounds: X({floor_bbox['min_x']:.2f} to {floor_bbox['max_x']:.2f}), Y({floor_bbox['min_y']:.2f} to {floor_bbox['max_y']:.2f})")
    
    # Get all other mesh objects (excluding object_target and floor)
    obstacles = [obj for obj in bpy.data.objects 
                 if obj.type == 'MESH' and obj.name != name and obj.name != 'ceilling' and obj.name != floor_name]
    
    print(f"✓ Found {len(obstacles)} obstacles to avoid:")
    for obs in obstacles:
        print(f"  - {obs.name}")
    
    # Get bounding boxes for all obstacles
    obstacle_bboxes = [get_bounding_box_2d(obj) for obj in obstacles]
    
    # Try to find a valid position
    print(f"\nAttempting to place object_target (max {max_attempts} attempts)...")
    
    for attempt in range(max_attempts):
        # Generate random X and Y position within floor bounds
        random_x = random.uniform(floor_bbox['min_x'] + margin, 
                                  floor_bbox['max_x'] - margin)
        random_y = random.uniform(floor_bbox['min_y'] + margin, 
                                  floor_bbox['max_y'] - margin)
        random_z = random.uniform(floor_bbox['min_z'] + margin, 
                                  floor_bbox['max_z'] - margin)
        
        # Generate random rotation for Z axis only (in radians)
        # X and Y axes are set to 0 (no rotation)
        rotation_x = 0  # No rotation on X axis
        rotation_y = 0  # No rotation on Y axis
        rotation_z = math.radians(random.uniform(z_rotation_range[0], z_rotation_range[1]))
        
        # Set new position (keep Z coordinate same)
        object_target.location.x = random_x
        object_target.location.y = random_y
        object_target.location.z=random_z
        
        # Set rotation: no rotation on X and Y, random rotation on Z
        object_target.rotation_euler[0] = rotation_x  # X axis (no rotation)
        object_target.rotation_euler[1] = rotation_y  # Y axis (no rotation)
        object_target.rotation_euler[2] = rotation_z  # Z axis (random rotation)
        
        # CRITICAL: Update the depsgraph to apply transformations
        bpy.context.view_layer.update()
        
        # Force viewport refresh
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        
        # Get object_target's new bounding box
        object_target_bbox = get_bounding_box_2d(object_target)
        
        # Check for collisions with all obstacles
        collision = False
        collision_with = None
        for i, obstacle_bbox in enumerate(obstacle_bboxes):
            if check_collision_2d(object_target_bbox, obstacle_bbox, margin):
                collision = True
                collision_with = obstacles[i].name
                break
        
        # If no collision, we found a valid position
        if not collision:
            print(f"\n{'=' * 50}")
            print(f"✓ SUCCESS! object_target placed at:")
            print(f"  Position:")
            print(f"    X: {random_x:.2f}")
            print(f"    Y: {random_y:.2f}")
            print(f"    Z: {object_target.location.z:.2f} (unchanged)")
            print(f"  Rotation:")
            print(f"    X: {math.degrees(rotation_x):.2f}° (no rotation)")
            print(f"    Y: {math.degrees(rotation_y):.2f}° (no rotation)")
            print(f"    Z: {math.degrees(rotation_z):.2f}° (random)")
            print(f"  Attempts needed: {attempt + 1}")
            print(f"{'=' * 50}")
            return random_x,random_y,random_z
        else:
            if attempt % 10 == 0 and attempt > 0:
                print(f"  Attempt {attempt + 1}: Collision with {collision_with}, retrying...")
    
    # Failed to find valid position
    print(f"\n{'=' * 50}")
    print(f"✗ FAILED: Could not place object_target after {max_attempts} attempts")
    print(f"Try: reducing margin, increasing max_attempts, or clearing space")
    print(f"{'=' * 50}")
    return random_x,random_y,random_z

txlocs=[]
rxlocs=[]



def move_obj_randomly( floor_name, max_attempts=100000, margin=0.5, 
                        z_rotation_range=(0, 360),size=[1,1,1],name="a"):
    """
    Place object_target randomly on the floor without overlapping other objects
    
    Parameters:
    - name: name of the  object
    - floor_name: name of the floor object to determine bounds
    - max_attempts: maximum number of random placement attempts
    - margin: minimum distance between objects (in Blender units)
    - z_rotation_range: tuple of (min, max) rotation in degrees for Z axis only
    """


    object_target = bpy.data.objects.get(name)
    
    
    if not object_target:
        print(f"ERROR: object_target '{name}' not found!")
        return False
    
    print(f"✓ Found object_target: {name}")
    
    # Get the floor object to determine placement bounds
    floor = bpy.data.objects.get(floor_name)
    ##############################
    if not floor:
        print(f"ERROR: Floor '{floor_name}' not found!")
        return False
    
    print(f"✓ Found floor: {floor_name}")
    
    # Set rotation mode to Euler
    object_target.rotation_mode = 'XYZ'
    
    # Get floor bounds
    floor_bbox = get_bounding_box_2d(floor)
    print(f"✓ Floor bounds: X({floor_bbox['min_x']:.2f} to {floor_bbox['max_x']:.2f}), Y({floor_bbox['min_y']:.2f} to {floor_bbox['max_y']:.2f})")
    
    # Get all other mesh objects (excluding object_target and floor)
    obstacles = [obj for obj in bpy.data.objects 
                 if obj.type == 'MESH' and obj.name != name and obj.name != 'ceilling' and obj.name != floor_name]
    print("obs"+str(obstacles))
    print(f"✓ Found {len(obstacles)} obstacles to avoid:")
    for obs in obstacles:
        print(f"  - {obs.name}")
    
    # Get bounding boxes for all obstacles
    obstacle_bboxes = [get_bounding_box_2d(obj) for obj in obstacles]
    
    # Try to find a valid position
    print(f"\nAttempting to place object_target (max {max_attempts} attempts)...")
    
    for attempt in range(max_attempts):
        # Generate random X and Y position within floor bounds
        random_x = random.uniform(floor_bbox['min_x'] + margin, 
                                  floor_bbox['max_x'] - margin)
        random_y = random.uniform(floor_bbox['min_y'] + margin, 
                                  floor_bbox['max_y'] - margin)
        
        # Generate random rotation for Z axis only (in radians)
        # X and Y axes are set to 0 (no rotation)
        rotation_x = 0  # No rotation on X axis
        rotation_y = 0  # No rotation on Y axis
        rotation_z = math.radians(random.uniform(z_rotation_range[0], z_rotation_range[1]))
        
        # Set new position (keep Z coordinate same)
        object_target.location.x = random_x
        object_target.location.y = random_y
        
        # Set rotation: no rotation on X and Y, random rotation on Z
        object_target.rotation_euler[0] = rotation_x  # X axis (no rotation)
        object_target.rotation_euler[1] = rotation_y  # Y axis (no rotation)
        object_target.rotation_euler[2] = rotation_z  # Z axis (random rotation)
        
        # CRITICAL: Update the depsgraph to apply transformations
        bpy.context.view_layer.update()
        
        # Force viewport refresh
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        
        # Get object_target's new bounding box
        object_target_bbox = get_bounding_box_2d(object_target)
        
        # Check for collisions with all obstacles
        collision = False
        collision_with = None
        for i, obstacle_bbox in enumerate(obstacle_bboxes):
            if check_collision_2d(object_target_bbox, obstacle_bbox, margin):
                collision = True
                collision_with = obstacles[i].name
                print(obstacles[i].name)
                break
        
        # If no collision, we found a valid position
        if not collision:
            print(f"\n{'=' * 50}")
            print(f"✓ SUCCESS! object_target placed at:")
            print(f"  Position:")
            print(f"    X: {random_x:.2f}")
            print(f"    Y: {random_y:.2f}")
            print(f"    Z: {object_target.location.z:.2f} (unchanged)")
            print(f"  Rotation:")
            print(f"    X: {math.degrees(rotation_x):.2f}° (no rotation)")
            print(f"    Y: {math.degrees(rotation_y):.2f}° (no rotation)")
            print(f"    Z: {math.degrees(rotation_z):.2f}° (random)")
            print(f"  Attempts needed: {attempt + 1}")
            print(f"{'=' * 50}")
            return True
        else:
            if attempt % 10 == 0 and attempt > 0:
                print(f"  Attempt {attempt + 1}: Collision with {collision_with}, retrying...")
    
    # Failed to find valid position
    print(f"\n{'=' * 50}")
    print(f"✗ FAILED: Could not place object_target after {max_attempts} attempts")
    print(f"Try: reducing margin, increasing max_attempts, or clearing space")
    print(f"{'=' * 50}")
    return False

#this will save a number of camera placements and an xml
fix_all_mesh_normals()
for i in range(100):
    
    move_obj_randomly(
    floor_name="floor",              # Your floor object name
    max_attempts=1000,                # Number of placement attempts
    margin=0.5,                      # Space between objects (in Blender units)
    z_rotation_range=(0, 360),
    size=[1,1,1],#objs[i][1],
    name="Cube.041"        # Full 360° rotation on Z axis
    )
    moved = bpy.data.objects["Cube.041"]
    moved.name="Cube.041-{0}".format(i)
    bpy.ops.export_scene.mitsuba(filepath="/roomsmoved/test{0}.xml".format(i))
    moved.name="Cube.041"
    
    tx_x,tx_y,tx_z=place_antenna_randomly(
    floor_name="floor",              # Your floor object name
    max_attempts=100,                # Number of placement attempts
    margin=0.5,                      # Space between objects (in Blender units)
    z_rotation_range=(0, 360),
    size=[1,1,1],#objs[i][1],
    name="TX"        # Full 360° rotation on Z axis
    )

    rx_x,rx_y,rx_z=place_antenna_randomly(
    floor_name="floor",              # Your floor object name
    max_attempts=100,                # Number of placement attempts
    margin=0.5,                      # Space between objects (in Blender units)
    z_rotation_range=(0, 360),
    size=[1,1,1],#objs[i][1],
    name="RX"        # Full 360° rotation on Z axis
    )
    obj = bpy.data.objects["TX"]
    bpy.data.objects.remove(obj, do_unlink=True)
    obj = bpy.data.objects["RX"]
    bpy.data.objects.remove(obj, do_unlink=True)
    # Run the fix
    txlocs.append([tx_x,tx_y,tx_z])
    rxlocs.append([rx_x,rx_y,rx_z])

np.save('txloc.npy', np.array(txlocs, dtype=object), allow_pickle=True)
np.save('rxloc.npy', np.array(rxlocs, dtype=object), allow_pickle=True)

