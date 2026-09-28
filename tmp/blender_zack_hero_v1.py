import bpy, math, os
from mathutils import Vector

OUT = os.path.abspath("hero_out")
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

scene.render.resolution_x = 1080
scene.render.resolution_y = 1920
scene.render.resolution_percentage = 100
scene.render.fps = 30
scene.frame_start = 1
scene.frame_end = 60
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.film_transparent = False

try:
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.35
except Exception:
    pass

try:
    scene.eevee.taa_render_samples = 16
    scene.eevee.taa_samples = 16
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = 3.0
    scene.eevee.gtao_factor = 1.45
except Exception:
    pass

try:
    scene.view_settings.look = "AgX - Medium High Contrast"
except Exception:
    try:
        scene.view_settings.look = "Medium High Contrast"
    except Exception:
        pass
scene.view_settings.exposure = 0.05

world = bpy.data.worlds.new("World")
world.use_nodes = True
scene.world = world
wbg = world.node_tree.nodes.get("Background")
wbg.inputs["Color"].default_value = (0.004, 0.012, 0.021, 1)
wbg.inputs["Strength"].default_value = 0.18

def mat(name, color, metallic=0.0, rough=0.4, coat=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bs = m.node_tree.nodes.get("Principled BSDF")
    bs.inputs["Base Color"].default_value = (*color, 1)
    bs.inputs["Metallic"].default_value = metallic
    bs.inputs["Roughness"].default_value = rough
    if "Coat Weight" in bs.inputs:
        bs.inputs["Coat Weight"].default_value = coat
    if "Coat Roughness" in bs.inputs:
        bs.inputs["Coat Roughness"].default_value = max(0.08, rough*0.45)
    m.diffuse_color = (*color,1)
    return m

RED       = mat("Deep_Crimson_Plastic", (0.48,0.009,0.015), 0.05, 0.19, 0.45)
RED2      = mat("Crimson_Edge", (0.16,0.004,0.008), 0.10, 0.23, 0.25)
STEEL     = mat("Brushed_Steel", (0.28,0.34,0.40), 0.88, 0.22)
STEEL_D   = mat("Dark_Steel", (0.08,0.11,0.14), 0.82, 0.26)
GOLD      = mat("Staple_Gold", (0.88,0.43,0.055), 0.96, 0.14)
PAPER     = mat("Paper", (0.93,0.965,1.0), 0.0, 0.74)
SKIN      = mat("Skin", (0.77,0.39,0.22), 0.0, 0.46, 0.08)
SKIN_HI   = mat("Skin_Highlight", (0.92,0.60,0.38), 0.0, 0.43, 0.08)
RUBBER    = mat("Rubber", (0.012,0.018,0.024), 0.08, 0.38)
FLOOR_M   = mat("Floor", (0.018,0.042,0.058), 0.18, 0.24, 0.20)
BACK_M    = mat("Backdrop", (0.012,0.030,0.043), 0.0, 0.82)
CYAN_M    = mat("CyanAccent", (0.015,0.28,0.42), 0.22, 0.22)
WHITE_M   = mat("White", (0.97,0.99,1.0), 0.05, 0.20, 0.18)

def smooth(o):
    if hasattr(o.data, "polygons"):
        for p in o.data.polygons:
            p.use_smooth = True

def cube(name, loc, scale, material, bevel=0.06, rot=(0,0,0), parent=None):
    bpy.ops.mesh.primitive_cube_add(location=loc, rotation=rot)
    o=bpy.context.object
    o.name=name
    o.scale=scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        b=o.modifiers.new("PrecisionBevel","BEVEL"); b.width=bevel; b.segments=6; b.profile=0.64
    if material: o.data.materials.append(material)
    if parent:
        o.parent=parent
        o.matrix_parent_inverse=parent.matrix_world.inverted()
    return o

def sphere(name, loc, scale, material, parent=None):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=28, location=loc)
    o=bpy.context.object; o.name=name; o.scale=scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    smooth(o)
    if material: o.data.materials.append(material)
    if parent:
        o.parent=parent
        o.matrix_parent_inverse=parent.matrix_world.inverted()
    return o

def cyl(name, loc, radius, depth, material, rot=(0,0,0), parent=None, verts=48):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=radius, depth=depth, location=loc, rotation=rot)
    o=bpy.context.object; o.name=name
    smooth(o)
    if material: o.data.materials.append(material)
    if parent:
        o.parent=parent
        o.matrix_parent_inverse=parent.matrix_world.inverted()
    return o

def empty(name, loc=(0,0,0), parent=None):
    o=bpy.data.objects.new(name,None); scene.collection.objects.link(o); o.location=loc
    if parent: o.parent=parent
    return o

def look_at(o,target):
    o.rotation_euler=(Vector(target)-o.location).to_track_quat("-Z","Y").to_euler()

def clamp01(x): return max(0.0,min(1.0,x))
def smoothstep(x):
    x=clamp01(x)
    return x*x*(3-2*x)
def smoother(x):
    x=clamp01(x)
    return x*x*x*(x*(x*6-15)+10)
def lerp(a,b,t): return a+(b-a)*t

# Stage / sweep
floor=cube("Floor",(0,0,-0.44),(7.5,6.0,0.08),FLOOR_M,0.08)
back=cube("Backdrop",(0,4.2,3.5),(7.5,0.08,4.8),BACK_M,0.08)
# subtle luminous-ish strips
strip1=cube("AccentA",(-3.2,4.07,1.4),(1.4,0.025,0.025),CYAN_M,0.015,rot=(0,0,math.radians(18)))
strip2=cube("AccentB",(3.1,4.07,4.2),(1.2,0.025,0.025),RED,0.015,rot=(0,0,math.radians(-16)))

# Main stapler base
base_root=empty("BaseRoot")
base=cube("Base",(0.0,0,0.10),(2.75,0.62,0.19),RED,0.16,parent=base_root)
base_insert=cube("BaseRubber",(-0.18,0,-0.16),(2.22,0.53,0.055),RUBBER,0.045,parent=base_root)
anvil=cube("Anvil",(1.25,-0.02,0.36),(0.58,0.50,0.055),STEEL,0.03,parent=base_root)
# anvil grooves
grooveL=cube("AnvilGrooveL",(1.05,-0.505,0.405),(0.23,0.018,0.028),STEEL_D,0.012,rot=(0,math.radians(10),0),parent=base_root)
grooveR=cube("AnvilGrooveR",(1.45,-0.505,0.405),(0.23,0.018,0.028),STEEL_D,0.012,rot=(0,math.radians(-10),0),parent=base_root)

# Paper stack, slightly staggered
paper1=cube("PaperBottom",(1.05,-0.01,0.475),(1.78,0.56,0.036),PAPER,0.016,rot=(0,0,math.radians(-1.2)))
paper2=cube("PaperTop",(1.10,-0.015,0.555),(1.72,0.55,0.036),PAPER,0.016,rot=(0,0,math.radians(1.0)))

# Hinge and top assembly
hinge_loc=(-1.82,0,0.66)
hinge=cyl("Hinge",hinge_loc,0.20,1.26,STEEL,rot=(math.radians(90),0,0))
hinge_cap1=cyl("HingeCapNear",(-1.82,-0.66,0.66),0.25,0.08,RED2,rot=(math.radians(90),0,0))
hinge_cap2=cyl("HingeCapFar",(-1.82,0.66,0.66),0.25,0.08,RED2,rot=(math.radians(90),0,0))

top_root=empty("TopRoot",hinge_loc)
# back half shell provides silhouette
shell_back=cube("ShellBack",(0.16,0.26,1.45),(2.18,0.31,0.21),RED,0.17,parent=top_root)
shell_rear=cube("ShellRear",(-1.25,0.0,1.26),(0.58,0.54,0.18),RED2,0.13,parent=top_root)
# removable/front cutaway shell
panel_root=empty("PanelRoot",parent=top_root)
shell_front=cube("ShellFront",(0.28,-0.31,1.45),(2.06,0.27,0.21),RED,0.16,parent=panel_root)
nose_cap=cube("NoseCap",(2.17,-0.30,1.35),(0.24,0.26,0.20),RED2,0.09,parent=panel_root)
# grip inset
grip=cube("GripInset",(0.12,-0.595,1.45),(1.18,0.035,0.095),RUBBER,0.04,parent=panel_root)

# Mechanism
magazine=cube("Magazine",(0.35,0.00,1.17),(1.82,0.29,0.072),STEEL,0.045,parent=top_root)
rail1=cube("RailNear",(0.30,-0.31,1.17),(1.70,0.045,0.10),STEEL_D,0.022,parent=top_root)
rail2=cube("RailFar",(0.30,0.31,1.17),(1.70,0.045,0.10),STEEL_D,0.022,parent=top_root)
spring=cyl("SpringGuide",(-0.58,0,1.28),0.065,1.2,STEEL,rot=(0,math.radians(90),0),parent=top_root)
pusher=cube("StaplePusher",(-0.75,0,1.17),(0.16,0.24,0.10),STEEL_D,0.03,parent=top_root)
blade=cube("DriveBlade",(1.83,0,0.95),(0.07,0.27,0.36),STEEL,0.025,parent=top_root)

# Staple strip
for i in range(11):
    x=-0.55+i*0.20
    bridge=cube(f"StripBridge{i}",(x,0,1.08),(0.078,0.22,0.025),GOLD,0.012,parent=top_root)
    cube(f"StripLegL{i}",(x,-0.19,0.98),(0.025,0.025,0.11),GOLD,0.008,parent=top_root)
    cube(f"StripLegR{i}",(x,0.19,0.98),(0.025,0.025,0.11),GOLD,0.008,parent=top_root)

# Active staple – separate hero element near nose
staple_root=empty("ActiveStapleRoot",(1.80,0,0.0))
active_bridge=cube("ActiveBridge",(0,0,0.92),(0.34,0.19,0.036),GOLD,0.014,parent=staple_root)
left_upper=cube("ActiveLeftUpper",(-0.30,0,0.68),(0.035,0.19,0.25),GOLD,0.012,parent=staple_root)
right_upper=cube("ActiveRightUpper",(0.30,0,0.68),(0.035,0.19,0.25),GOLD,0.012,parent=staple_root)
left_piv=empty("LeftBendPivot",(-0.30,0,0.44),staple_root)
right_piv=empty("RightBendPivot",(0.30,0,0.44),staple_root)
left_tip=cube("ActiveLeftTip",(-0.30,0,0.24),(0.035,0.19,0.22),GOLD,0.012,parent=left_piv)
right_tip=cube("ActiveRightTip",(0.30,0,0.24),(0.035,0.19,0.22),GOLD,0.012,parent=right_piv)

# Screws/details
for x in (-1.15,1.35):
    cyl(f"Screw{x}",(x,-0.635,1.42),0.085,0.035,STEEL,rot=(math.radians(90),0,0),parent=top_root)

# Stylized but more constructed hand, mostly above frame
hand_root=empty("HandRoot")
palm=sphere("Palm",(0.20,-0.20,2.58),(0.86,0.56,0.34),SKIN,hand_root)
thenar=sphere("Thenar",(0.64,-0.24,2.34),(0.42,0.35,0.25),SKIN_HI,hand_root)
wrist=cyl("Wrist",(-0.72,-0.16,2.83),0.31,1.45,SKIN,rot=(0,math.radians(70),0),parent=hand_root)
# index finger segments pressing shell
idx1=cyl("Index1",(0.88,-0.12,2.05),0.17,0.78,SKIN_HI,rot=(0,math.radians(12),0),parent=hand_root)
idx2=cyl("Index2",(0.99,-0.12,1.52),0.155,0.46,SKIN_HI,rot=(0,math.radians(-4),0),parent=hand_root)
tip=sphere("IndexTip",(0.98,-0.12,1.29),(0.19,0.18,0.18),SKIN_HI,hand_root)
# hints of other fingers
for j,(x,y,z,ang) in enumerate([(0.15,-0.48,2.23,70),(-0.08,-0.50,2.35,75),(-0.30,-0.46,2.44,80)]):
    cyl(f"FingerHint{j}",(x,y,z),0.135,0.70,SKIN,rot=(0,math.radians(ang),0),parent=hand_root)

# Camera and DOF
focus=empty("Focus",(1.75,0,0.73))
bpy.ops.object.camera_add(location=(4.35,-7.45,3.10))
cam=bpy.context.object
cam.name="HeroCamera"
cam.data.lens=58
cam.data.dof.use_dof=True
cam.data.dof.focus_object=focus
cam.data.dof.aperture_fstop=3.2
scene.camera=cam
look_at(cam,focus.location)

# Lighting - cinematic product setup
bpy.ops.object.light_add(type="AREA", location=(-4.5,-4.0,6.8))
key=bpy.context.object; key.name="KeySoftbox"; key.data.energy=1500; key.data.shape="RECTANGLE"; key.data.size=5.0; key.data.size_y=6.5; key.data.color=(1.0,0.68,0.48); look_at(key,(0.6,0,1.0))

bpy.ops.object.light_add(type="AREA", location=(4.8,-1.2,4.6))
rim=bpy.context.object; rim.name="CyanRim"; rim.data.energy=1250; rim.data.shape="RECTANGLE"; rim.data.size=3.0; rim.data.size_y=5.0; rim.data.color=(0.16,0.62,1.0); look_at(rim,(0.7,0,1.0))

bpy.ops.object.light_add(type="AREA", location=(0,3.0,5.8))
top=bpy.context.object; top.name="TopFill"; top.data.energy=1100; top.data.shape="DISK"; top.data.size=4.0; top.data.color=(1.0,0.22,0.14); look_at(top,(0.6,0,1.0))

bpy.ops.object.light_add(type="AREA", location=(0,-1.0,6.8))
white=bpy.context.object; white.name="WhiteEdge"; white.data.energy=650; white.data.size=3.0; white.data.color=(0.92,0.97,1.0); look_at(white,(1.3,0,0.8))

# subtle practical highlight card
card=cube("LightCard",(3.7,2.8,3.4),(1.4,0.03,2.1),WHITE_M,0.02,rot=(0,math.radians(-20),0))
card.hide_render=True

# Save scene source
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT,"zack_hero_v1.blend"))

def setup_frame(f):
    # 0-20: press; 20-35: shell opens/reveal; 26-52: staple drives; 48-60: bend + final macro
    t_press=smoother((f-1)/22.0)
    t_reveal=smoother((f-18)/18.0)
    t_drive=smoother((f-25)/27.0)
    t_bend=smoother((f-47)/13.0)

    top_root.rotation_euler=(0, math.radians(lerp(-11.0,-1.0,t_press)), 0)
    hand_root.location=(0,0,lerp(0.18,-0.12,t_press))
    hand_root.rotation_euler=(math.radians(lerp(-1.5,1.6,t_press)), math.radians(lerp(-2,1,t_press)), math.radians(lerp(-1,1.2,t_press)))

    # front shell slides left and slightly forward, deliberately exposing mechanism
    panel_root.location=(lerp(0,-3.25,t_reveal), lerp(0,-0.25,t_reveal), lerp(0,0.10,t_reveal))
    panel_root.rotation_euler=(math.radians(lerp(0,4,t_reveal)), math.radians(lerp(0,-7,t_reveal)), math.radians(lerp(0,-7,t_reveal)))

    # drive blade + active staple
    blade.location.z=lerp(0,-0.45,t_drive)
    staple_root.location.z=lerp(0,-0.58,t_drive)
    left_piv.rotation_euler[1]=math.radians(lerp(0,-78,t_bend))
    right_piv.rotation_euler[1]=math.radians(lerp(0,78,t_bend))

    # camera: macro push and slight orbit
    tc=smoother((f-1)/59.0)
    cam.location=(lerp(4.35,2.15,tc), lerp(-7.45,-4.05,tc), lerp(3.10,1.35,tc))
    cam.data.lens=lerp(58,82,tc)
    focus.location=(lerp(1.10,1.76,tc), 0, lerp(0.94,0.48,tc))
    look_at(cam,focus.location)

    # final micro parallax
    if f>45:
        tt=smoothstep((f-45)/15.0)
        cam.location.x += 0.16*math.sin(tt*math.pi)
        cam.location.y += 0.10*tt
        look_at(cam,focus.location)

start=int(os.environ.get("START_FRAME","1"))
end=int(os.environ.get("END_FRAME","60"))
for f in range(start,end+1):
    scene.frame_set(f)
    setup_frame(f)
    bpy.context.view_layer.update()
    scene.render.filepath=os.path.join(FRAMES,f"frame_{f:04d}.png")
    bpy.ops.render.render(write_still=True)
    print(f"RENDERED {f}/{end} chunk={start}-{end}",flush=True)

print("DONE",flush=True)
