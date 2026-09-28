import bpy, math, os
from mathutils import Vector

OUT = os.path.abspath("render_out")
FRAMES = os.path.join(OUT, "frames")
os.makedirs(FRAMES, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
    try:
        scene.render.engine = eng
        break
    except Exception:
        pass
scene.render.resolution_x = 720
scene.render.resolution_y = 1280
scene.render.resolution_percentage = 100
scene.render.fps = 30
scene.frame_start = 1
scene.frame_end = 150
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.film_transparent = False
try:
    scene.render.use_motion_blur = False
except Exception:
    pass
try:
    scene.eevee.taa_render_samples = 24
    scene.eevee.taa_samples = 16
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = 3
    scene.eevee.gtao_factor = 1.3
except Exception:
    pass
try:
    scene.view_settings.look = "AgX - Medium High Contrast"
except Exception:
    try:
        scene.view_settings.look = "Medium High Contrast"
    except Exception:
        pass
scene.view_settings.exposure = 0.15

world = bpy.data.worlds.new("World")
world.use_nodes = True
scene.world = world
bg = world.node_tree.nodes.get("Background")
bg.inputs["Color"].default_value = (0.012, 0.020, 0.035, 1)
bg.inputs["Strength"].default_value = 0.38

def material(name, color, metallic=0.0, roughness=0.45):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    m.diffuse_color = (*color, 1)
    return m

RED = material("stapler_red", (0.62, 0.018, 0.028), 0.15, 0.22)
RED_DARK = material("stapler_red_dark", (0.19, 0.008, 0.012), 0.2, 0.25)
METAL = material("steel", (0.22, 0.27, 0.32), 0.82, 0.2)
GOLD = material("staple_gold", (0.95, 0.52, 0.08), 0.9, 0.18)
PAPER = material("paper", (0.92, 0.96, 1.0), 0.0, 0.82)
SKIN = material("skin", (0.72, 0.36, 0.21), 0.0, 0.48)
SKIN2 = material("skin_light", (0.88, 0.55, 0.34), 0.0, 0.5)
DARK = material("rubber", (0.018, 0.024, 0.032), 0.1, 0.45)
FLOOR = material("floor", (0.028, 0.055, 0.075), 0.05, 0.3)
BACK = material("backdrop", (0.024, 0.050, 0.074), 0.0, 0.7)
CYAN = material("accent", (0.02, 0.42, 0.58), 0.25, 0.3)

def smooth(obj):
    if hasattr(obj.data, "polygons"):
        for p in obj.data.polygons:
            p.use_smooth = True

def cube(name, loc, scale, mat, bevel=0.08, rot=(0,0,0), parent=None):
    bpy.ops.mesh.primitive_cube_add(location=loc, rotation=rot)
    o = bpy.context.object
    o.name = name
    o.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel > 0:
        mod = o.modifiers.new("Bevel", "BEVEL")
        mod.width = bevel
        mod.segments = 4
    if mat:
        o.data.materials.append(mat)
    if parent:
        o.parent = parent
        o.matrix_parent_inverse = parent.matrix_world.inverted()
    return o

def sphere(name, loc, scale, mat, parent=None):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=36, ring_count=20, location=loc)
    o = bpy.context.object
    o.name = name
    o.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    smooth(o)
    if mat:
        o.data.materials.append(mat)
    if parent:
        o.parent = parent
        o.matrix_parent_inverse = parent.matrix_world.inverted()
    return o

def cyl(name, loc, radius, depth, mat, rot=(0,0,0), parent=None, verts=32):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=radius, depth=depth, location=loc, rotation=rot)
    o = bpy.context.object
    o.name = name
    smooth(o)
    if mat:
        o.data.materials.append(mat)
    if parent:
        o.parent = parent
        o.matrix_parent_inverse = parent.matrix_world.inverted()
    return o

def empty(name, loc=(0,0,0), parent=None):
    o = bpy.data.objects.new(name, None)
    scene.collection.objects.link(o)
    o.location = loc
    if parent:
        o.parent = parent
    return o

def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

def ease(t):
    t = max(0.0, min(1.0, t))
    return t*t*(3.0-2.0*t)

def lerp(a,b,t):
    return a + (b-a)*t

# Permanent stage
floor = cube("Floor", (0,0,-0.78), (7,6,0.10), FLOOR, 0.12)
back = cube("Backdrop", (0,3.9,3.0), (7,0.10,4.4), BACK, 0.1)
accent = cube("AccentStrip", (0,3.72,1.6), (4.3,0.04,0.035), CYAN, 0.02)

# Camera
bpy.ops.object.camera_add(location=(4.6,-8.2,3.45))
cam = bpy.context.object
cam.name = "Camera"
scene.camera = cam
cam.data.lens = 48

# Lighting
bpy.ops.object.light_add(type="AREA", location=(-4.2,-4.8,6.5))
key = bpy.context.object
key.data.energy = 1100
key.data.shape = "DISK"
key.data.size = 5.5
look_at(key, (0,0,0.9))

bpy.ops.object.light_add(type="AREA", location=(4.5,-1.5,4.0))
rim = bpy.context.object
rim.data.energy = 950
rim.data.color = (0.25,0.65,1.0)
rim.data.size = 4.0
look_at(rim, (0,0,0.8))

bpy.ops.object.light_add(type="AREA", location=(0,2.2,5.0))
fill = bpy.context.object
fill.data.energy = 700
fill.data.color = (1.0,0.28,0.20)
fill.data.size = 3.0
look_at(fill, (0,0,0.8))

# ---------- SHOT 1: exterior ----------
ext = []
base = cube("Ext_Base", (0,0,0.02), (2.55,0.62,0.18), RED, 0.18); ext.append(base)
pad = cube("Ext_BasePad", (-0.25,0,-0.22), (2.0,0.52,0.07), DARK, 0.06); ext.append(pad)
anvil = cube("Ext_Anvil", (1.34,-0.01,0.25), (0.48,0.48,0.055), METAL, 0.035); ext.append(anvil)
paper1 = cube("Ext_Paper1", (1.2,-0.01,0.36), (1.55,0.54,0.035), PAPER, 0.02); ext.append(paper1)
paper2 = cube("Ext_Paper2", (1.18,-0.01,0.44), (1.50,0.53,0.035), PAPER, 0.02, rot=(0,0,math.radians(-1.8))); ext.append(paper2)
hinge = cyl("Ext_Hinge", (-1.83,0,0.62), 0.18, 1.38, METAL, rot=(math.radians(90),0,0)); ext.append(hinge)
top_root = empty("Ext_TopRoot", (-1.83,0,0.62))
shell = cube("Ext_TopShell", (0.16,0,1.18), (2.20,0.57,0.20), RED, 0.18, parent=top_root); ext.append(shell)
shell2 = cube("Ext_TopInset", (0.22,0,0.91), (1.95,0.45,0.085), RED_DARK, 0.07, parent=top_root); ext.append(shell2)
mag = cube("Ext_Magazine", (0.28,0,0.74), (1.86,0.31,0.075), METAL, 0.04, parent=top_root); ext.append(mag)
nose = cube("Ext_Nose", (2.03,0,0.83), (0.18,0.35,0.12), METAL, 0.04, parent=top_root); ext.append(nose)
# stylized hand
hand_root = empty("HandRoot", (0,0,0))
palm = sphere("Palm", (0.35,-0.10,2.35), (0.92,0.58,0.36), SKIN, hand_root); ext.append(palm)
wrist = cyl("Wrist", (-0.35,-0.10,2.62), 0.33, 1.15, SKIN2, rot=(0,math.radians(72),0), parent=hand_root); ext.append(wrist)
finger = cube("IndexFinger", (0.78,-0.08,1.75), (0.22,0.24,0.68), SKIN2, 0.20, rot=(0,math.radians(-8),0), parent=hand_root); ext.append(finger)
tip = sphere("FingerTip", (0.88,-0.08,1.11), (0.24,0.25,0.28), SKIN2, hand_root); ext.append(tip)
thumb = cube("Thumb", (-0.18,-0.42,2.05), (0.38,0.20,0.22), SKIN, 0.16, rot=(0,math.radians(20),math.radians(18)), parent=hand_root); ext.append(thumb)
ext += [top_root, hand_root]

# ---------- SHOT 2: cross-section ----------
cross = []
cross_root = empty("CrossRoot")
c_back = cube("Cross_BackPlate", (0,0.38,0.45), (2.55,0.10,1.55), RED_DARK, 0.12, parent=cross_root); cross.append(c_back)
c_shell = cube("Cross_Shell", (0,0.08,1.42), (2.35,0.38,0.14), RED, 0.12, parent=cross_root); cross.append(c_shell)
c_mag = cube("Cross_Mag", (0,0.02,1.12), (1.82,0.23,0.055), METAL, 0.03, parent=cross_root); cross.append(c_mag)
cp1 = cube("Cross_Paper1", (0,0,0.28), (1.95,0.50,0.045), PAPER, 0.025, parent=cross_root); cross.append(cp1)
cp2 = cube("Cross_Paper2", (0,0,0.40), (1.88,0.49,0.045), PAPER, 0.025, parent=cross_root); cross.append(cp2)
canvil = cube("Cross_Anvil", (0,0,-0.28), (1.25,0.48,0.09), METAL, 0.06, parent=cross_root); cross.append(canvil)
grooveL = cube("GrooveL", (-0.52,-0.47,-0.18), (0.26,0.05,0.08), DARK, 0.02, rot=(0,math.radians(12),0), parent=cross_root); cross.append(grooveL)
grooveR = cube("GrooveR", (0.52,-0.47,-0.18), (0.26,0.05,0.08), DARK, 0.02, rot=(0,math.radians(-12),0), parent=cross_root); cross.append(grooveR)

staple_root = empty("StapleRoot", parent=cross_root)
bridge = cube("StapleBridge", (0,-0.04,0.94), (0.78,0.12,0.055), GOLD, 0.025, parent=staple_root); cross.append(bridge)
left_upper = cube("StapleLeftUpper", (-0.72,-0.04,0.48), (0.055,0.12,0.46), GOLD, 0.02, parent=staple_root); cross.append(left_upper)
right_upper = cube("StapleRightUpper", (0.72,-0.04,0.48), (0.055,0.12,0.46), GOLD, 0.02, parent=staple_root); cross.append(right_upper)
left_piv = empty("LeftTipPivot", (-0.72,-0.04,0.04), staple_root)
right_piv = empty("RightTipPivot", (0.72,-0.04,0.04), staple_root)
left_tip = cube("StapleLeftTip", (-0.72,-0.04,-0.22), (0.055,0.12,0.28), GOLD, 0.02, parent=left_piv); cross.append(left_tip)
right_tip = cube("StapleRightTip", (0.72,-0.04,-0.22), (0.055,0.12,0.28), GOLD, 0.02, parent=right_piv); cross.append(right_tip)
cross += [cross_root, staple_root, left_piv, right_piv]

# ---------- SHOT 3: underside macro ----------
res = []
res_root = empty("ResultRoot")
rp1 = cube("ResultPaper1", (0,0,0.62), (1.95,0.58,0.055), PAPER, 0.025, parent=res_root); res.append(rp1)
rp2 = cube("ResultPaper2", (0,0,0.76), (1.90,0.56,0.055), PAPER, 0.025, parent=res_root); res.append(rp2)
rplate = cube("ResultAnvil", (0,0,-0.24), (1.35,0.55,0.10), METAL, 0.07, parent=res_root); res.append(rplate)
# fixed upper parts entering paper
rlu = cube("ResultLeftUpper", (-0.72,0,0.38), (0.055,0.12,0.32), GOLD, 0.02, parent=res_root); res.append(rlu)
rru = cube("ResultRightUpper", (0.72,0,0.38), (0.055,0.12,0.32), GOLD, 0.02, parent=res_root); res.append(rru)
rlp = empty("ResultLeftPivot", (-0.72,0,0.08), res_root)
rrp = empty("ResultRightPivot", (0.72,0,0.08), res_root)
rlt = cube("ResultLeftTip", (-0.72,0,-0.20), (0.055,0.12,0.30), GOLD, 0.02, parent=rlp); res.append(rlt)
rrt = cube("ResultRightTip", (0.72,0,-0.20), (0.055,0.12,0.30), GOLD, 0.02, parent=rrp); res.append(rrt)
res += [res_root, rlp, rrp]

all_groups = [ext, cross, res]

def show_only(group_index):
    for gi, group in enumerate(all_groups):
        vis = gi == group_index
        for o in group:
            o.hide_render = not vis
            o.hide_viewport = not vis

def setup_external(f):
    floor.hide_render = False
    floor.hide_viewport = False
    t = ease((f-1)/35.0)
    top_root.rotation_euler = (0, math.radians(lerp(-12.0, 0.0, t)), 0)
    hand_root.location.z = lerp(0.18, -0.20, t)
    # subtle hand anticipation then press
    hand_root.rotation_euler[1] = math.radians(lerp(-2.0, 2.5, t))
    cam.location = (lerp(4.9,4.0,t), lerp(-8.8,-7.15,t), lerp(3.75,3.05,t))
    cam.data.lens = lerp(46, 54, t)
    look_at(cam, (0.45,0,0.92))

def setup_cross(f):
    floor.hide_render = False
    floor.hide_viewport = False
    t = (f-37)/59.0
    drop = ease(min(1.0, t/0.68))
    bend = ease(max(0.0, (t-0.62)/0.38))
    staple_root.location.z = lerp(0.78, -0.46, drop)
    left_piv.rotation_euler[1] = math.radians(lerp(0, -74, bend))
    right_piv.rotation_euler[1] = math.radians(lerp(0, 74, bend))
    cam.location = (lerp(0.30,-0.18,t), lerp(-7.2,-5.5,t), lerp(0.78,0.28,t))
    cam.data.lens = lerp(58, 72, t)
    look_at(cam, (0,0,lerp(0.50,0.05,t)))

def setup_result(f):
    floor.hide_render = True
    floor.hide_viewport = True
    t = (f-97)/53.0
    bend = ease(min(1.0, t/0.56))
    reveal = ease(max(0.0, (t-0.46)/0.54))
    rlp.rotation_euler[1] = math.radians(lerp(0, -82, bend))
    rrp.rotation_euler[1] = math.radians(lerp(0, 82, bend))
    rplate.location.z = lerp(0, -0.62, reveal)
    cam.location = (lerp(0.18,0.0,t), lerp(-5.8,-4.65,t), lerp(-0.95,-0.68,t))
    cam.data.lens = lerp(66, 82, t)
    look_at(cam, (0,0,0.26))

# Save a reusable .blend before rendering
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "zack_stapler_v3.blend"))

start_frame = int(os.environ.get("START_FRAME", "1"))
end_frame = int(os.environ.get("END_FRAME", "150"))
frame_list_env = os.environ.get("FRAME_LIST", "").strip()
frames_to_render = [int(x) for x in frame_list_env.split(",") if x.strip()] if frame_list_env else list(range(start_frame, end_frame+1))
for f in frames_to_render:
    scene.frame_set(f)
    if f <= 36:
        show_only(0)
        setup_external(f)
    elif f <= 96:
        show_only(1)
        setup_cross(f)
    else:
        show_only(2)
        setup_result(f)
    bpy.context.view_layer.update()
    scene.render.filepath = os.path.join(FRAMES, f"frame_{f:04d}.png")
    bpy.ops.render.render(write_still=True)
    print(f"RENDERED {f}/{end_frame} chunk={start_frame}-{end_frame}", flush=True)

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "zack_stapler_v3.blend"))
print("DONE", flush=True)
