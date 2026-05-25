extends Node3D
# ─────────────────────────────────────────────────────────────────────────────
# MAP MAKER  v5
#
# Placement grids:
#   CELL grid  (16×16)  — road tiles, ducks (on road), start point
#   CORNER grid (17×17) — signs snap to grid-line intersections (road edge)
#
# Signs are invisible from top-down because they're vertical, so each sign
# gets a flat colored disc + direction arrow overlay on the ground.
# ─────────────────────────────────────────────────────────────────────────────

const TILE_SIZE: float = 0.600
const GRID_W:    int   = 16
const GRID_H:    int   = 16

const MAPS_DIR:        String = "user://maps"
const ACTIVE_MAP_PATH: String = "user://custom_map.json"

const _SCENE_LABELS: Array = ["Model Map Maker", "Braitenberg", "Lane Follower"]
const _SCENE_PATHS:  Array = [
	"res://modelmapmaker.tscn",
	"res://scenes/braitenberg.tscn",
	"res://scenes/lane_follower.tscn",
]

enum TileType { EMPTY = 0, STRAIGHT = 1, CURVE = 2, CROSS = 3, CROSS3 = 4 }

const OBJ_DUCK:    String = "duck"
const OBJ_STOP:    String = "stop_sign"
const OBJ_PARKING: String = "parking_sign"

# ── Selection ─────────────────────────────────────────────────────────────────
var _selected_item: String = "straight"
var _selected_rot:  int    = 0

# ── Road tile layer  (cell grid) ──────────────────────────────────────────────
var _grid:       Dictionary = {}
var _tile_nodes: Dictionary = {}

# ── Duck layer  (cell grid — duck stands on the road) ─────────────────────────
var _duck_cells: Dictionary = {}   # Vector2i(col,row) -> {rot:int}
var _duck_nodes: Dictionary = {}

# ── Sign layer  (corner grid — signs sit at road edges) ───────────────────────
# Corner (cx,cy) world pos = (cx*TILE_SIZE, 0, cy*TILE_SIZE)
var _sign_corners:        Dictionary = {}   # Vector2i(cx,cy) -> {type:String, rot:int}
var _sign_nodes:          Dictionary = {}   # the actual 3-D sign model
var _sign_indicator_nodes: Dictionary = {}  # flat top-down disc + arrow overlay

# ── Start point  (cell grid) ──────────────────────────────────────────────────
var _start_cell: Vector2i = Vector2i(-1, -1)
var _start_rot:  int      = 0
var _start_node: Node3D   = null

# ── 3-D scene refs ────────────────────────────────────────────────────────────
var _camera:          Camera3D
var _cell_hover:      MeshInstance3D
var _corner_hover:    MeshInstance3D
var _ghost_node:      Node3D
var _tiles_root:      Node3D
var _props_root:      Node3D
var _indicators_root: Node3D
var _tile_scenes:     Dictionary = {}
var _obj_scenes:      Dictionary = {}

# ── UI refs ───────────────────────────────────────────────────────────────────
var _name_edit:         LineEdit
var _map_dropdown:      OptionButton
var _scene_dropdown:    OptionButton
var _active_label:      Label
var _sel_label:         Label
var _rot_label:         Label
var _start_coord_label: Label
var _status_label:      Label
var _item_buttons:      Dictionary = {}


# ══════════════════════════════════════════════════════════════════════════════
#  SETUP
# ══════════════════════════════════════════════════════════════════════════════

func _ready() -> void:
	_ensure_maps_dir()
	_load_tile_scenes()
	_load_obj_scenes()
	_build_3d_scene()
	_build_ui()
	_refresh_map_list()
	_update_active_label()
	_select_item("straight")
	print("[MapMaker] Ready")


func _ensure_maps_dir() -> void:
	var abs := ProjectSettings.globalize_path(MAPS_DIR)
	if not DirAccess.dir_exists_absolute(abs):
		DirAccess.make_dir_recursive_absolute(abs)


func _load_tile_scenes() -> void:
	var paths := {
		TileType.STRAIGHT: "res://scenes/tiles/tile_straight.tscn",
		TileType.CURVE:    "res://scenes/tiles/tile_curve.tscn",
		TileType.CROSS:    "res://scenes/tiles/tile_cross.tscn",
		TileType.CROSS3:   "res://scenes/tiles/tile_cross3.tscn",
	}
	for t in paths:
		_tile_scenes[t] = load(paths[t])
		if _tile_scenes[t] == null:
			push_warning("[MapMaker] Missing tile scene: " + paths[t])


func _load_obj_scenes() -> void:
	var paths := {
		OBJ_DUCK:    "res://scenes/objects/obj_duck.tscn",
		OBJ_STOP:    "res://scenes/objects/obj_stop_sign.tscn",
		OBJ_PARKING: "res://scenes/objects/obj_parking_sign.tscn",
	}
	for t in paths:
		_obj_scenes[t] = load(paths[t])
		if _obj_scenes[t] == null:
			push_warning("[MapMaker] Missing object scene: " + paths[t])


# ── 3-D scene ─────────────────────────────────────────────────────────────────

func _build_3d_scene() -> void:
	_add_camera()
	_add_environment()
	_add_ground_plane()
	_add_grid_lines()
	_add_hover_highlights()

	_tiles_root = Node3D.new()
	_tiles_root.name = "Tiles"
	add_child(_tiles_root)

	_props_root = Node3D.new()
	_props_root.name = "Props"
	add_child(_props_root)

	# Sign overlays live here — separate so they don't get serialized with props
	_indicators_root = Node3D.new()
	_indicators_root.name = "Indicators"
	add_child(_indicators_root)


func _add_camera() -> void:
	_camera = Camera3D.new()
	_camera.projection = Camera3D.PROJECTION_ORTHOGONAL
	_camera.size  = GRID_H * TILE_SIZE * 1.25
	_camera.near  = 0.1
	_camera.far   = 100.0
	_camera.position = Vector3(GRID_W * TILE_SIZE * 0.5, 20.0, GRID_H * TILE_SIZE * 0.5)
	_camera.rotation_degrees = Vector3(-90.0, 0.0, 0.0)
	add_child(_camera)
	_camera.current = true


func _add_environment() -> void:
	var ew  := WorldEnvironment.new()
	var env := Environment.new()
	env.background_mode      = Environment.BG_COLOR
	env.background_color     = Color(0.18, 0.18, 0.18)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color  = Color(1.0, 1.0, 1.0)
	env.ambient_light_energy = 1.8
	ew.environment = env
	add_child(ew)


func _add_ground_plane() -> void:
	var body := StaticBody3D.new()
	var col  := CollisionShape3D.new()
	col.shape = WorldBoundaryShape3D.new()
	body.add_child(col)
	add_child(body)

	var mi  := MeshInstance3D.new()
	var pm  := PlaneMesh.new()
	pm.size  = Vector2(GRID_W * TILE_SIZE * 3.0, GRID_H * TILE_SIZE * 3.0)
	mi.mesh  = pm
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(0.14, 0.22, 0.10)
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mi.material_override = mat
	mi.position = Vector3(GRID_W * TILE_SIZE * 0.5, 0.084, GRID_H * TILE_SIZE * 0.5)
	add_child(mi)


func _add_grid_lines() -> void:
	var imm := ImmediateMesh.new()
	var mi  := MeshInstance3D.new()
	mi.mesh = imm
	var mat := StandardMaterial3D.new()
	mat.albedo_color  = Color(0.6, 0.6, 0.6, 0.7)
	mat.transparency  = BaseMaterial3D.TRANSPARENCY_ALPHA
	mat.shading_mode  = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.cull_mode     = BaseMaterial3D.CULL_DISABLED
	mi.material_override = mat
	add_child(mi)

	var y := 0.086
	imm.surface_begin(Mesh.PRIMITIVE_LINES)
	for c in range(GRID_W + 1):
		var x: float = c * TILE_SIZE
		imm.surface_add_vertex(Vector3(x, y, 0.0))
		imm.surface_add_vertex(Vector3(x, y, GRID_H * TILE_SIZE))
	for r in range(GRID_H + 1):
		var z: float = r * TILE_SIZE
		imm.surface_add_vertex(Vector3(0.0,                y, z))
		imm.surface_add_vertex(Vector3(GRID_W * TILE_SIZE, y, z))
	imm.surface_end()


func _add_hover_highlights() -> void:
	# Large yellow quad — cell mode (tiles, ducks, start)
	_cell_hover = MeshInstance3D.new()
	var pm  := PlaneMesh.new()
	pm.size  = Vector2(TILE_SIZE * 0.92, TILE_SIZE * 0.92)
	_cell_hover.mesh = pm
	var mat := StandardMaterial3D.new()
	mat.albedo_color  = Color(1.0, 1.0, 0.2, 0.30)
	mat.transparency  = BaseMaterial3D.TRANSPARENCY_ALPHA
	mat.shading_mode  = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.cull_mode     = BaseMaterial3D.CULL_DISABLED
	_cell_hover.material_override = mat
	_cell_hover.position.y = 0.087
	_cell_hover.visible    = false
	add_child(_cell_hover)

	# Small green dot — corner mode (signs)
	_corner_hover = MeshInstance3D.new()
	var cpm  := PlaneMesh.new()
	cpm.size  = Vector2(TILE_SIZE * 0.20, TILE_SIZE * 0.20)
	_corner_hover.mesh = cpm
	var cmat := StandardMaterial3D.new()
	cmat.albedo_color  = Color(0.2, 1.0, 0.35, 0.95)
	cmat.transparency  = BaseMaterial3D.TRANSPARENCY_ALPHA
	cmat.shading_mode  = BaseMaterial3D.SHADING_MODE_UNSHADED
	cmat.cull_mode     = BaseMaterial3D.CULL_DISABLED
	_corner_hover.material_override = cmat
	_corner_hover.position.y = 0.089
	_corner_hover.visible    = false
	add_child(_corner_hover)


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

func _build_ui() -> void:
	var cl := CanvasLayer.new()
	add_child(cl)

	var panel := PanelContainer.new()
	panel.position = Vector2(10.0, 10.0)
	cl.add_child(panel)

	var scroll := ScrollContainer.new()
	scroll.custom_minimum_size = Vector2(268.0, get_viewport().get_visible_rect().size.y - 30.0)
	scroll.vertical_scroll_mode   = ScrollContainer.SCROLL_MODE_AUTO
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	panel.add_child(scroll)

	var vb := VBoxContainer.new()
	vb.custom_minimum_size      = Vector2(248.0, 0.0)
	vb.size_flags_horizontal    = Control.SIZE_EXPAND_FILL
	scroll.add_child(vb)

	_add_label(vb, "─── MAP MAKER ───")
	_add_sep(vb)

	# Map name
	_add_label(vb, "MAP NAME")
	_name_edit = LineEdit.new()
	_name_edit.text = "my_map"
	_name_edit.placeholder_text = "enter map name..."
	vb.add_child(_name_edit)
	_add_button(vb, "Save Map", _save_map)
	_add_button(vb, "⬇ Export as .tscn Scene", _export_as_scene)
	_add_sep(vb)

	# Saved maps
	_add_label(vb, "SAVED MAPS")
	_map_dropdown = OptionButton.new()
	_map_dropdown.custom_minimum_size = Vector2(230, 0)
	vb.add_child(_map_dropdown)
	var hb_maps := HBoxContainer.new()
	vb.add_child(hb_maps)
	var load_btn := Button.new()
	load_btn.text = "Load"
	load_btn.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	load_btn.pressed.connect(_load_selected_map)
	hb_maps.add_child(load_btn)
	var refresh_btn := Button.new()
	refresh_btn.text = "↻"
	refresh_btn.pressed.connect(_refresh_map_list)
	hb_maps.add_child(refresh_btn)
	var del_btn := Button.new()
	del_btn.text = "Del"
	del_btn.pressed.connect(_delete_selected_map)
	hb_maps.add_child(del_btn)
	_add_button(vb, "▶ Set Active for Simulation", _set_active_map)
	_add_sep(vb)

	# Scene selector
	_add_label(vb, "SPAWN IN SCENE")
	_scene_dropdown = OptionButton.new()
	for lbl in _SCENE_LABELS:
		_scene_dropdown.add_item(lbl)
	vb.add_child(_scene_dropdown)
	_add_button(vb, "▶ Run in Scene", _run_in_scene)
	_add_sep(vb)

	# Road tiles
	_add_label(vb, "ROAD TILES")
	_add_swatch_toggle(vb, "straight",  "Straight",        Color(0.40, 0.40, 0.40))
	_add_swatch_toggle(vb, "curve",     "Curve",           Color(0.50, 0.45, 0.35))
	_add_swatch_toggle(vb, "cross3",    "3-Way Crossing",  Color(0.35, 0.30, 0.30))
	_add_swatch_toggle(vb, "cross",     "4-Way Crossing",  Color(0.30, 0.30, 0.30))
	_add_sep(vb)

	# Props
	_add_label(vb, "PROPS")
	_add_swatch_toggle(vb, OBJ_DUCK,    "Duck  (on road, cell center)",   Color(0.95, 0.80, 0.10))
	_add_swatch_toggle(vb, OBJ_STOP,    "Stop Sign  (road edge, corner)", Color(0.90, 0.08, 0.08))
	_add_swatch_toggle(vb, OBJ_PARKING, "Parking Sign  (road edge)",      Color(0.08, 0.20, 0.85))
	_add_sep(vb)

	# Start point + erase
	_add_swatch_toggle(vb, "start", "Set Start Point", Color(1.0, 0.82, 0.0))
	_add_swatch_toggle(vb, "",      "Erase",           Color(0.7, 0.15, 0.15))
	_add_sep(vb)

	# Rotation
	_sel_label = _add_label(vb, "Selected: Straight")
	_rot_label = _add_label(vb, "Rotation: 0°")
	_add_button(vb, "Rotate 90°", _rotate_cw)
	_add_sep(vb)

	# Clear
	var hb_clear := HBoxContainer.new()
	vb.add_child(hb_clear)
	var cr := Button.new()
	cr.text = "Clear Roads"
	cr.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	cr.pressed.connect(_clear_roads)
	hb_clear.add_child(cr)
	var cp := Button.new()
	cp.text = "Clear Props"
	cp.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	cp.pressed.connect(_clear_props)
	hb_clear.add_child(cp)
	_add_sep(vb)

	# Status
	_active_label      = _add_label(vb, "Active: none")
	_start_coord_label = _add_label(vb, "Start: not set")
	_status_label      = _add_label(vb, "Roads: 0  Ducks: 0  Signs: 0")
	_add_sep(vb)
	_add_label(vb, "LClick: place\nRClick: erase")


func _add_label(parent: Control, text: String) -> Label:
	var l := Label.new()
	l.text = text
	parent.add_child(l)
	return l


func _add_sep(parent: Control) -> void:
	parent.add_child(HSeparator.new())


func _add_button(parent: Control, text: String, cb: Callable) -> Button:
	var b := Button.new()
	b.text = text
	b.pressed.connect(cb)
	parent.add_child(b)
	return b


func _add_swatch_toggle(parent: Control, key: String, label_text: String, color: Color) -> void:
	var hb := HBoxContainer.new()
	parent.add_child(hb)
	var sw := ColorRect.new()
	sw.custom_minimum_size = Vector2(14, 20)
	sw.color = color
	hb.add_child(sw)
	var b := Button.new()
	b.text = label_text
	b.toggle_mode = true
	b.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	b.pressed.connect(func(): _select_item(key))
	_item_buttons[key] = b
	hb.add_child(b)


# ══════════════════════════════════════════════════════════════════════════════
#  SELECTION
# ══════════════════════════════════════════════════════════════════════════════

func _select_item(key: String) -> void:
	_selected_item = key
	for k in _item_buttons:
		_item_buttons[k].button_pressed = (k == key)
	_sel_label.text = "Selected: " + _item_display(key)
	_rebuild_ghost()


func _rebuild_ghost() -> void:
	if is_instance_valid(_ghost_node):
		_ghost_node.queue_free()
	_ghost_node = null

	var scene: PackedScene = null
	match _selected_item:
		"straight":  scene = _tile_scenes.get(TileType.STRAIGHT)
		"curve":     scene = _tile_scenes.get(TileType.CURVE)
		"cross3":    scene = _tile_scenes.get(TileType.CROSS3)
		"cross":     scene = _tile_scenes.get(TileType.CROSS)
		OBJ_DUCK:    scene = _obj_scenes.get(OBJ_DUCK)
		OBJ_STOP:    scene = _obj_scenes.get(OBJ_STOP)
		OBJ_PARKING: scene = _obj_scenes.get(OBJ_PARKING)

	if scene == null:
		return

	_ghost_node = scene.instantiate() as Node3D
	_ghost_node.rotation_degrees.y = _selected_rot
	_ghost_node.visible = false
	_apply_ghost_material(_ghost_node)
	add_child(_ghost_node)


func _apply_ghost_material(node: Node) -> void:
	if node is MeshInstance3D:
		var mi := node as MeshInstance3D
		if mi.mesh != null:
			for i in range(mi.mesh.get_surface_count()):
				var orig: Material = mi.get_surface_override_material(i)
				if orig == null:
					orig = mi.mesh.surface_get_material(i)
				var mat: StandardMaterial3D
				if orig is StandardMaterial3D:
					mat = orig.duplicate() as StandardMaterial3D
					mat.albedo_color.a = 0.5
				else:
					mat = StandardMaterial3D.new()
					mat.albedo_color = Color(0.8, 0.8, 0.8, 0.5)
				mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
				mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
				mat.cull_mode    = BaseMaterial3D.CULL_DISABLED
				mi.set_surface_override_material(i, mat)
	for child in node.get_children():
		_apply_ghost_material(child)


func _item_display(key: String) -> String:
	match key:
		"straight":  return "Straight"
		"curve":     return "Curve"
		"cross3":    return "3-Way Crossing"
		"cross":     return "4-Way Crossing"
		OBJ_DUCK:    return "Duck"
		OBJ_STOP:    return "Stop Sign"
		OBJ_PARKING: return "Parking Sign"
		"start":     return "Start Point"
		"":          return "Erase"
	return key


# Signs use corner snap; everything else uses cell snap
func _is_sign_item(item: String) -> bool:
	return item in [OBJ_STOP, OBJ_PARKING]


func _rotate_cw() -> void:
	_selected_rot = (_selected_rot + 90) % 360
	_rot_label.text = "Rotation: %d°" % _selected_rot
	if is_instance_valid(_ghost_node):
		_ghost_node.rotation_degrees.y = _selected_rot


# ══════════════════════════════════════════════════════════════════════════════
#  INPUT
# ══════════════════════════════════════════════════════════════════════════════

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == KEY_R:
			_rotate_cw()
			return

	if event is InputEventMouseMotion:
		if _is_sign_item(_selected_item):
			_cell_hover.visible = false
			var corner := _mouse_to_corner()
			var valid  := _corner_valid(corner)
			_corner_hover.visible = valid
			if valid:
				var wp := _corner_world_pos(corner)
				_corner_hover.position = Vector3(wp.x, 0.009, wp.z)
				if is_instance_valid(_ghost_node):
					_ghost_node.position = Vector3(wp.x, 0.091, wp.z)
					_ghost_node.visible  = true
			else:
				if is_instance_valid(_ghost_node):
					_ghost_node.visible = false
		else:
			_corner_hover.visible = false
			var cell  := _mouse_to_cell()
			var valid := _cell_valid(cell)
			_cell_hover.visible = valid
			if valid:
				var wp := _cell_world_center(cell)
				_cell_hover.position   = wp
				_cell_hover.position.y = 0.087
				if is_instance_valid(_ghost_node):
					_ghost_node.position = Vector3(wp.x, 0.091, wp.z)
					_ghost_node.visible  = true
			else:
				if is_instance_valid(_ghost_node):
					_ghost_node.visible = false
		return

	if event is InputEventMouseButton and event.pressed:
		if _is_sign_item(_selected_item):
			var corner := _mouse_to_corner()
			if not _corner_valid(corner):
				return
			if event.button_index == MOUSE_BUTTON_LEFT:
				print("[MapMaker] Sign click: item=%s rot=%d corner=%s" % [_selected_item, _selected_rot, str(corner)])
				_place_sign(corner, _selected_item, _selected_rot)
			elif event.button_index == MOUSE_BUTTON_RIGHT:
				_erase_sign(corner)
		else:
			var cell := _mouse_to_cell()
			if not _cell_valid(cell):
				return
			if event.button_index == MOUSE_BUTTON_LEFT:
				_handle_cell_left(cell)
			elif event.button_index == MOUSE_BUTTON_RIGHT:
				_handle_cell_right(cell)


func _handle_cell_left(cell: Vector2i) -> void:
	match _selected_item:
		"straight": _place_tile(cell, TileType.STRAIGHT, _selected_rot)
		"curve":    _place_tile(cell, TileType.CURVE,    _selected_rot)
		"cross3":   _place_tile(cell, TileType.CROSS3,   _selected_rot)
		"cross":    _place_tile(cell, TileType.CROSS,    _selected_rot)
		OBJ_DUCK:   _place_duck(cell, _selected_rot)
		"start":    _set_start(cell)
		"":
			_erase_tile(cell)
			_erase_duck(cell)
			_maybe_erase_start(cell)


func _handle_cell_right(cell: Vector2i) -> void:
	match _selected_item:
		"straight", "curve", "cross3", "cross": _erase_tile(cell)
		OBJ_DUCK:                     _erase_duck(cell)
		"start":                      _maybe_erase_start(cell)
		"":
			_erase_tile(cell)
			_erase_duck(cell)
			_maybe_erase_start(cell)


# ══════════════════════════════════════════════════════════════════════════════
#  GRID MATH
# ══════════════════════════════════════════════════════════════════════════════

func _mouse_to_cell() -> Vector2i:
	var mp     := get_viewport().get_mouse_position()
	var origin := _camera.project_ray_origin(mp)
	var dir    := _camera.project_ray_normal(mp)
	if abs(dir.y) < 0.0001:
		return Vector2i(-1, -1)
	var t  := -origin.y / dir.y
	var wp := origin + dir * t
	return Vector2i(int(floor(wp.x / TILE_SIZE)), int(floor(wp.z / TILE_SIZE)))


func _cell_valid(c: Vector2i) -> bool:
	return c.x >= 0 and c.x < GRID_W and c.y >= 0 and c.y < GRID_H


func _cell_world_center(c: Vector2i) -> Vector3:
	return Vector3((c.x + 0.5) * TILE_SIZE, 0.0, (c.y + 0.5) * TILE_SIZE)


func _mouse_to_corner() -> Vector2i:
	var mp     := get_viewport().get_mouse_position()
	var origin := _camera.project_ray_origin(mp)
	var dir    := _camera.project_ray_normal(mp)
	if abs(dir.y) < 0.0001:
		return Vector2i(-1, -1)
	var t  := -origin.y / dir.y
	var wp := origin + dir * t
	return Vector2i(int(round(wp.x / TILE_SIZE)), int(round(wp.z / TILE_SIZE)))


func _corner_valid(c: Vector2i) -> bool:
	return c.x >= 0 and c.x <= GRID_W and c.y >= 0 and c.y <= GRID_H


func _corner_world_pos(c: Vector2i) -> Vector3:
	return Vector3(c.x * TILE_SIZE, 0.0, c.y * TILE_SIZE)


# ══════════════════════════════════════════════════════════════════════════════
#  ROAD TILES
# ══════════════════════════════════════════════════════════════════════════════

func _place_tile(cell: Vector2i, type: int, rot: int) -> void:
	_erase_tile(cell)
	var scene: PackedScene = _tile_scenes.get(type)
	if scene == null:
		return
	var node: Node3D = scene.instantiate()
	var twp := _cell_world_center(cell)
	node.position           = Vector3(twp.x, 0.091, twp.z)
	node.rotation_degrees.y = rot
	_tiles_root.add_child(node)
	_tile_nodes[cell] = node
	_grid[cell] = {"type": type, "rot": rot}
	_refresh_status()


func _erase_tile(cell: Vector2i) -> void:
	if _tile_nodes.has(cell):
		_tile_nodes[cell].queue_free()
		_tile_nodes.erase(cell)
	_grid.erase(cell)
	_refresh_status()


func _clear_roads() -> void:
	for n in _tile_nodes.values():
		n.queue_free()
	_tile_nodes.clear()
	_grid.clear()
	_refresh_status()


# ══════════════════════════════════════════════════════════════════════════════
#  DUCKS  (cell grid — on the road)
# ══════════════════════════════════════════════════════════════════════════════

func _place_duck(cell: Vector2i, rot: int) -> void:
	_erase_duck(cell)
	var scene: PackedScene = _obj_scenes.get(OBJ_DUCK)
	if scene == null:
		return
	var node: Node3D = scene.instantiate()
	var wp := _cell_world_center(cell)
	node.position           = Vector3(wp.x, 0.091, wp.z)
	node.rotation_degrees.y = rot
	_props_root.add_child(node)
	_duck_nodes[cell] = node
	_duck_cells[cell] = {"rot": rot}
	_refresh_status()


func _erase_duck(cell: Vector2i) -> void:
	if _duck_nodes.has(cell):
		_duck_nodes[cell].queue_free()
		_duck_nodes.erase(cell)
	_duck_cells.erase(cell)
	_refresh_status()


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNS  (corner grid — at road edges)
# Each sign gets a visible top-down indicator: colored disc + direction arrow.
# Without this the vertical sign model is invisible from top-down camera.
# ══════════════════════════════════════════════════════════════════════════════

func _place_sign(corner: Vector2i, type: String, rot: int) -> void:
	_erase_sign(corner)
	# Actual 3-D sign model (shows in simulation)
	var scene: PackedScene = _obj_scenes.get(type)
	if scene != null:
		var node: Node3D = scene.instantiate()
		var swp := _corner_world_pos(corner)
		node.position           = Vector3(swp.x, 0.091, swp.z)
		node.rotation_degrees.y = rot
		_props_root.add_child(node)
		_sign_nodes[corner] = node

	_sign_corners[corner] = {"type": type, "rot": rot}

	# Top-down overlay so you can see the sign in the editor
	var icwp := _corner_world_pos(corner)
	var indicator := _make_sign_indicator(Vector3(icwp.x, 0.086, icwp.z), type, rot)
	_indicators_root.add_child(indicator)
	_sign_indicator_nodes[corner] = indicator

	_refresh_status()


func _erase_sign(corner: Vector2i) -> void:
	if _sign_nodes.has(corner):
		_sign_nodes[corner].queue_free()
		_sign_nodes.erase(corner)
	if _sign_indicator_nodes.has(corner):
		_sign_indicator_nodes[corner].queue_free()
		_sign_indicator_nodes.erase(corner)
	_sign_corners.erase(corner)
	_refresh_status()


# Builds a flat ground overlay: colored disc + white directional arrow.
# The root is y-rotated to match the sign, so the arrow always points
# toward the sign's face direction.
func _make_sign_indicator(world_pos: Vector3, type: String, rot_y: int) -> Node3D:
	var root := Node3D.new()
	root.position         = world_pos
	root.rotation_degrees = Vector3(0, rot_y, 0)

	var sign_color := Color(0.92, 0.07, 0.07) if type == OBJ_STOP else Color(0.07, 0.18, 0.88)

	# ── Colored disc ──────────────────────────────────────────────────────────
	var disc := MeshInstance3D.new()
	var disc_mesh := PlaneMesh.new()
	disc_mesh.size = Vector2(0.14, 0.14)
	disc.mesh = disc_mesh
	var disc_mat := StandardMaterial3D.new()
	disc_mat.albedo_color  = sign_color
	disc_mat.shading_mode  = BaseMaterial3D.SHADING_MODE_UNSHADED
	disc_mat.cull_mode     = BaseMaterial3D.CULL_DISABLED
	disc.material_override = disc_mat
	disc.position.y        = 0.010
	root.add_child(disc)

	# ── Direction arrow (points toward sign face = +Z in local space) ─────────
	# Arrow stem: thin elongated plane offset along +Z from the disc edge.
	var arrow := MeshInstance3D.new()
	var arrow_mesh := PlaneMesh.new()
	arrow_mesh.size = Vector2(0.04, 0.18)
	arrow.mesh = arrow_mesh
	var arrow_mat := StandardMaterial3D.new()
	arrow_mat.albedo_color  = Color(1.0, 1.0, 1.0, 0.95)
	arrow_mat.shading_mode  = BaseMaterial3D.SHADING_MODE_UNSHADED
	arrow_mat.cull_mode     = BaseMaterial3D.CULL_DISABLED
	arrow.material_override = arrow_mat
	arrow.position = Vector3(0.0, 0.012, 0.16)
	root.add_child(arrow)

	# Arrowhead: wider triangle-like tip so direction is unmistakable
	var tip := MeshInstance3D.new()
	var tip_mesh := PlaneMesh.new()
	tip_mesh.size = Vector2(0.10, 0.06)
	tip.mesh = tip_mesh
	var tip_mat := StandardMaterial3D.new()
	tip_mat.albedo_color  = Color(1.0, 1.0, 1.0, 0.95)
	tip_mat.shading_mode  = BaseMaterial3D.SHADING_MODE_UNSHADED
	tip_mat.cull_mode     = BaseMaterial3D.CULL_DISABLED
	tip.material_override = tip_mat
	tip.position = Vector3(0.0, 0.012, 0.28)
	root.add_child(tip)

	return root


# ══════════════════════════════════════════════════════════════════════════════
#  START POINT
# ══════════════════════════════════════════════════════════════════════════════

func _set_start(cell: Vector2i) -> void:
	if _start_cell == cell:
		_clear_start_visual()
		_start_cell = Vector2i(-1, -1)
		_refresh_status()
		return
	_clear_start_visual()
	_start_cell = cell
	_start_rot  = _selected_rot

	var wp := _cell_world_center(cell)
	var bot_scene := load("res://models/VehicleCorected.glb") as PackedScene
	if bot_scene != null:
		var bot := bot_scene.instantiate() as Node3D
		bot.position           = Vector3(wp.x, 0.122, wp.z)
		bot.rotation_degrees.y = _selected_rot
		add_child(bot)
		_start_node = bot
	else:
		# Fallback: yellow diamond if model not found
		var mi  := MeshInstance3D.new()
		var pm  := PlaneMesh.new()
		pm.size = Vector2(TILE_SIZE * 0.72, TILE_SIZE * 0.72)
		mi.mesh = pm
		var mat := StandardMaterial3D.new()
		mat.albedo_color     = Color(1.0, 0.82, 0.0, 0.92)
		mat.transparency     = BaseMaterial3D.TRANSPARENCY_ALPHA
		mat.shading_mode     = BaseMaterial3D.SHADING_MODE_UNSHADED
		mat.cull_mode        = BaseMaterial3D.CULL_DISABLED
		mi.material_override = mat
		mi.rotation_degrees.y = 45.0
		mi.position = Vector3(wp.x, 0.009, wp.z)
		add_child(mi)
		_start_node = mi
	_refresh_status()


func _maybe_erase_start(cell: Vector2i) -> void:
	if _start_cell == cell:
		_clear_start_visual()
		_start_cell = Vector2i(-1, -1)
		_refresh_status()


func _clear_start_visual() -> void:
	if _start_node != null:
		_start_node.queue_free()
		_start_node = null


# ══════════════════════════════════════════════════════════════════════════════
#  CLEAR ALL
# ══════════════════════════════════════════════════════════════════════════════

func _clear_props() -> void:
	for n in _duck_nodes.values():
		n.queue_free()
	_duck_nodes.clear()
	_duck_cells.clear()

	for n in _sign_nodes.values():
		n.queue_free()
	_sign_nodes.clear()

	for n in _sign_indicator_nodes.values():
		n.queue_free()
	_sign_indicator_nodes.clear()
	_sign_corners.clear()

	_refresh_status()


# ══════════════════════════════════════════════════════════════════════════════
#  STATUS
# ══════════════════════════════════════════════════════════════════════════════

func _refresh_status() -> void:
	_status_label.text = "Roads: %d  Ducks: %d  Signs: %d" % [
		_grid.size(), _duck_cells.size(), _sign_corners.size()
	]
	if _start_cell.x >= 0:
		var wx := (_start_cell.x + 0.5) * TILE_SIZE
		var wz := (_start_cell.y + 0.5) * TILE_SIZE
		_start_coord_label.text = "Start: col %d, row %d\n  pos (%.3f, 0, %.3f)" % [
			_start_cell.x, _start_cell.y, wx, wz
		]
	else:
		_start_coord_label.text = "Start: not set"


# ══════════════════════════════════════════════════════════════════════════════
#  MAP LIST
# ══════════════════════════════════════════════════════════════════════════════

func _refresh_map_list() -> void:
	_map_dropdown.clear()
	var dir := DirAccess.open(MAPS_DIR)
	if dir == null:
		return
	dir.list_dir_begin()
	var names: Array = []
	var fname := dir.get_next()
	while fname != "":
		if not dir.current_is_dir() and fname.ends_with(".json"):
			names.append(fname.get_basename())
		fname = dir.get_next()
	dir.list_dir_end()
	names.sort()
	for n in names:
		_map_dropdown.add_item(n)


func _get_map_path(map_name: String) -> String:
	return MAPS_DIR + "/" + map_name + ".json"


func _update_active_label(name: String = "") -> void:
	if name != "":
		_active_label.text = "Active: " + name
		return
	if not FileAccess.file_exists(ACTIVE_MAP_PATH):
		_active_label.text = "Active: none"
		return
	var f := FileAccess.open(ACTIVE_MAP_PATH, FileAccess.READ)
	if f == null:
		_active_label.text = "Active: ?"
		return
	var data = JSON.parse_string(f.get_as_text())
	f.close()
	if data is Dictionary and data.has("name"):
		_active_label.text = "Active: " + str(data["name"])
	else:
		_active_label.text = "Active: custom_map"


func _run_in_scene() -> void:
	var idx := _scene_dropdown.selected
	if idx < 0 or idx >= _SCENE_PATHS.size():
		return
	get_tree().change_scene_to_file(_SCENE_PATHS[idx])


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE / LOAD
# ══════════════════════════════════════════════════════════════════════════════

func _save_map() -> void:
	var map_name := _name_edit.text.strip_edges()
	if map_name.is_empty():
		_status_label.text = "Enter a map name first!"
		return

	var data := {
		"name":      map_name,
		"tile_size": TILE_SIZE,
		"grid_w":    GRID_W,
		"grid_h":    GRID_H,
		"tiles":     [],
		"ducks":     [],
		"signs":     [],
	}

	for cell in _grid.keys():
		var e: Dictionary = _grid[cell]
		data["tiles"].append({"col": cell.x, "row": cell.y, "type": e["type"], "rot": e["rot"]})

	for cell in _duck_cells.keys():
		var e: Dictionary = _duck_cells[cell]
		data["ducks"].append({"col": cell.x, "row": cell.y, "rot": e["rot"]})

	for corner in _sign_corners.keys():
		var e: Dictionary = _sign_corners[corner]
		data["signs"].append({"cx": corner.x, "cy": corner.y, "type": e["type"], "rot": e["rot"]})

	if _start_cell.x >= 0:
		data["start_point"] = {"col": _start_cell.x, "row": _start_cell.y, "rot": _start_rot}

	var json_str := JSON.stringify(data, "\t")

	var f := FileAccess.open(_get_map_path(map_name), FileAccess.WRITE)
	if f == null:
		_status_label.text = "Save FAILED!"
		return
	f.store_string(json_str)
	f.close()

	var fa := FileAccess.open(ACTIVE_MAP_PATH, FileAccess.WRITE)
	if fa:
		fa.store_string(json_str)
		fa.close()

	_status_label.text = "Saved '%s'!" % map_name
	_update_active_label(map_name)
	_refresh_map_list()
	for i in range(_map_dropdown.item_count):
		if _map_dropdown.get_item_text(i) == map_name:
			_map_dropdown.select(i)
			break


func _load_selected_map() -> void:
	if _map_dropdown.item_count == 0:
		_status_label.text = "No saved maps found"
		return
	var map_name := _map_dropdown.get_item_text(_map_dropdown.selected)
	_load_from_file(_get_map_path(map_name), map_name)


func _load_from_file(path: String, display_name: String) -> void:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		_status_label.text = "Cannot open map file"
		return
	var data = JSON.parse_string(f.get_as_text())
	f.close()
	if data == null:
		_status_label.text = "Load FAILED!"
		return

	_clear_roads()
	_clear_props()
	_clear_start_visual()
	_start_cell = Vector2i(-1, -1)

	for td in data.get("tiles", []):
		_place_tile(Vector2i(int(td["col"]), int(td["row"])), int(td["type"]), int(td["rot"]))

	for dd in data.get("ducks", []):
		_place_duck(Vector2i(int(dd["col"]), int(dd["row"])), int(dd["rot"]))

	for sd in data.get("signs", []):
		_place_sign(Vector2i(int(sd["cx"]), int(sd["cy"])), str(sd["type"]), int(sd["rot"]))

	var sp = data.get("start_point", null)
	if sp != null:
		var prev_rot := _selected_rot
		_selected_rot = int(sp.get("rot", 0))
		_set_start(Vector2i(int(sp["col"]), int(sp["row"])))
		_selected_rot = prev_rot

	_name_edit.text = display_name
	_status_label.text = "Loaded '%s'" % display_name


func _set_active_map() -> void:
	if _map_dropdown.item_count == 0:
		_status_label.text = "No saved maps found"
		return
	var map_name := _map_dropdown.get_item_text(_map_dropdown.selected)
	var f := FileAccess.open(_get_map_path(map_name), FileAccess.READ)
	if f == null:
		_status_label.text = "Cannot read: " + map_name
		return
	var content := f.get_as_text()
	f.close()
	var fa := FileAccess.open(ACTIVE_MAP_PATH, FileAccess.WRITE)
	if fa == null:
		_status_label.text = "Cannot write active map!"
		return
	fa.store_string(content)
	fa.close()
	_status_label.text = "Active: '%s'" % map_name
	_update_active_label(map_name)


func _delete_selected_map() -> void:
	if _map_dropdown.item_count == 0:
		_status_label.text = "No maps to delete"
		return
	var map_name := _map_dropdown.get_item_text(_map_dropdown.selected)
	var abs_path := ProjectSettings.globalize_path(_get_map_path(map_name))
	if DirAccess.remove_absolute(abs_path) == OK:
		_status_label.text = "Deleted: " + map_name
		_refresh_map_list()
	else:
		_status_label.text = "Delete failed!"


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORT AS .tscn SCENE
# Builds a real Godot scene file with all tiles, ducks, and signs instantiated
# at their correct world positions. Open it directly in Godot or instance it.
# ══════════════════════════════════════════════════════════════════════════════

func _export_as_scene() -> void:
	var map_name := _name_edit.text.strip_edges()
	if map_name.is_empty():
		_status_label.text = "Enter a map name first!"
		return
	if _grid.is_empty() and _duck_cells.is_empty() and _sign_corners.is_empty():
		_status_label.text = "Nothing to export!"
		return

	var root := Node3D.new()
	root.name = map_name.replace(" ", "_")

	# ── Road tiles ─────────────────────────────────────────────────────────────
	var tile_root := Node3D.new()
	tile_root.name = "Tiles"
	root.add_child(tile_root)
	tile_root.owner = root

	for cell in _grid.keys():
		var e: Dictionary = _grid[cell]
		var scene: PackedScene = _tile_scenes.get(e["type"])
		if scene == null:
			continue
		var node: Node3D = scene.instantiate()
		node.name       = "Tile_%d_%d" % [cell.x, cell.y]
		var etwp := _cell_world_center(cell)
		node.position           = Vector3(etwp.x, 0.091, etwp.z)
		node.rotation_degrees.y = e["rot"]
		tile_root.add_child(node)
		node.owner = root
		_set_owner_recursive(node, root)

	# ── Ducks ──────────────────────────────────────────────────────────────────
	var duck_root := Node3D.new()
	duck_root.name = "Ducks"
	root.add_child(duck_root)
	duck_root.owner = root

	var duck_scene: PackedScene = _obj_scenes.get(OBJ_DUCK)
	for cell in _duck_cells.keys():
		if duck_scene == null:
			break
		var e: Dictionary = _duck_cells[cell]
		var node: Node3D = duck_scene.instantiate()
		node.name = "Duck_%d_%d" % [cell.x, cell.y]
		var wp := _cell_world_center(cell)
		node.position           = Vector3(wp.x, 0.176, wp.z)
		node.rotation_degrees.y = e["rot"]
		duck_root.add_child(node)
		node.owner = root
		_set_owner_recursive(node, root)

	# ── Signs ──────────────────────────────────────────────────────────────────
	var sign_root := Node3D.new()
	sign_root.name = "Signs"
	root.add_child(sign_root)
	sign_root.owner = root

	for corner in _sign_corners.keys():
		var e: Dictionary = _sign_corners[corner]
		var scene: PackedScene = _obj_scenes.get(e["type"])
		if scene == null:
			continue
		var node: Node3D = scene.instantiate()
		node.name               = "Sign_%d_%d" % [corner.x, corner.y]
		var eswp := _corner_world_pos(corner)
		node.position           = Vector3(eswp.x, 0.091, eswp.z)
		node.rotation_degrees.y = e["rot"]
		sign_root.add_child(node)
		node.owner = root
		_set_owner_recursive(node, root)

	# ── Start point (bot) ────────────────────────────────────────────────────
	if _start_cell.x >= 0:
		var bot_scene := load("res://models/VehicleCorected.glb") as PackedScene
		if bot_scene != null:
			var bot := bot_scene.instantiate() as Node3D
			bot.name = "StartBot"
			var bwp := _cell_world_center(_start_cell)
			bot.position           = Vector3(bwp.x, 0.122, bwp.z)
			bot.rotation_degrees.y = _start_rot
			root.add_child(bot)
			bot.owner = root
			_set_owner_recursive(bot, root)

	# ── Green ground plane ────────────────────────────────────────────────────
	var ground_size := Vector2(GRID_W * TILE_SIZE * 3.0, GRID_H * TILE_SIZE * 3.0)
	var ground_center := Vector3(GRID_W * TILE_SIZE * 0.5, 0.084, GRID_H * TILE_SIZE * 0.5)

	var ground_body := StaticBody3D.new()
	ground_body.name = "Ground"
	var col := CollisionShape3D.new()
	col.shape = WorldBoundaryShape3D.new()
	ground_body.add_child(col)
	col.owner = root
	root.add_child(ground_body)
	ground_body.owner = root

	var ground_mi := MeshInstance3D.new()
	ground_mi.name = "GroundMesh"
	var ground_pm := PlaneMesh.new()
	ground_pm.size = ground_size
	ground_mi.mesh = ground_pm
	var ground_mat := StandardMaterial3D.new()
	ground_mat.albedo_color = Color(0.14, 0.22, 0.10)
	ground_mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	ground_mi.material_override = ground_mat
	ground_mi.position = ground_center
	root.add_child(ground_mi)
	ground_mi.owner = root

	# ── Pack and save ──────────────────────────────────────────────────────────
	# ResourceSaver can't write to res:// at runtime, so:
	# 1. Save to user:// (always writable)
	# 2. Copy the bytes to the real res://scenes/maps/ folder on disk

	var packed := PackedScene.new()
	var err    := packed.pack(root)
	root.free()

	if err != OK:
		_status_label.text = "Pack failed! (%d)" % err
		return

	var tmp_path := "user://_%s_tmp.tscn" % map_name
	err = ResourceSaver.save(packed, tmp_path)
	if err != OK:
		_status_label.text = "Pack save failed! (%d)" % err
		return

	# Copy from user:// to res://scenes/maps/ via absolute filesystem paths
	var dst_abs_dir := ProjectSettings.globalize_path("res://") + "scenes/maps"
	DirAccess.make_dir_recursive_absolute(dst_abs_dir)
	var dst_abs := dst_abs_dir + "/" + map_name + ".tscn"
	var src_abs := ProjectSettings.globalize_path(tmp_path)

	var src_f := FileAccess.open(src_abs, FileAccess.READ)
	if src_f == null:
		_status_label.text = "Read tmp failed!"
		return
	var bytes := src_f.get_buffer(src_f.get_length())
	src_f.close()

	var dst_f := FileAccess.open(dst_abs, FileAccess.WRITE)
	if dst_f == null:
		_status_label.text = "Write failed! Check folder permissions."
		return
	dst_f.store_buffer(bytes)
	dst_f.close()

	# Clean up temp file
	DirAccess.remove_absolute(src_abs)

	_status_label.text = "Saved!"
	print("[MapMaker] Exported scene: " + dst_abs)


func _set_owner_recursive(node: Node, owner: Node) -> void:
	for child in node.get_children():
		child.owner = owner
		_set_owner_recursive(child, owner)
