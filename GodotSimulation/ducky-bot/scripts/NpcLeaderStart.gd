extends PathFollow3D
## NPC leader for the convoy follow scene. Unlike path_follow_3d.gd it stays
## parked until the camera stream to Python is up (the follower's agent starts
## right after that accept) plus a short grace, so the follower departs with
## the leader's back marker in view instead of an empty road.

@export var speed: float = 0.12
@export var start_grace_s: float = 4.0

var _connected_for: float = -1.0

@onready var _camera_streamer: Node = get_tree().current_scene.get_node_or_null("DuckieBot/CameraStreamer")

func _ready() -> void:
	rotation_mode = PathFollow3D.ROTATION_Y
	loop = true
	add_to_group("npc_leader")

func reset_leader() -> void:
	# Called (via group) when Python resets the game: re-park at the start of
	# the path and re-run the departure grace so the respawned follower gets
	# the same leader-ahead geometry as a fresh boot.
	progress = 0.0
	_connected_for = -1.0

func _camera_up() -> bool:
	if _camera_streamer == null:
		return true  # no networked bot in the scene; just drive
	var tcp: StreamPeerTCP = _camera_streamer._tcp
	return tcp != null and tcp.get_status() == StreamPeerTCP.STATUS_CONNECTED

func _process(delta: float) -> void:
	if not _camera_up():
		_connected_for = -1.0
		return
	if _connected_for < 0.0:
		_connected_for = 0.0
		print("[NPC] camera stream up; leader departs in ", start_grace_s, "s")
	_connected_for += delta
	if _connected_for < start_grace_s:
		return
	progress += speed * delta
