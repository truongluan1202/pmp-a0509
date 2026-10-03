"""Read ZED skeleton and depth from a ROS 2 bag.

All knowledge of the ZED message layout is isolated in _decode_skeleton().
If the SDK message layout changes, that is the only function to touch.

Requires a sourced ROS 2 environment with zed_msgs available. Run inside the
container (dev@ prompt), never on the host.
"""
import os
import sqlite3
import numpy as np

from .types import Body, Frame, N_KP

SKEL_TOPIC = "/zed_external/zed_node/body_trk/skeletons"
DEPTH_TOPIC = "/zed_external/zed_node/depth/depth_registered"
DEPTH_INFO_TOPIC = "/zed_external/zed_node/depth/camera_info"


def _import_ros():
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    return deserialize_message, get_message


def _db3_path(bag_dir):
    if os.path.isfile(bag_dir) and bag_dir.endswith(".db3"):
        return bag_dir
    hits = [f for f in os.listdir(bag_dir) if f.endswith(".db3")]
    if not hits:
        raise FileNotFoundError(f"no .db3 file in {bag_dir}")
    return os.path.join(bag_dir, sorted(hits)[0])


def topic_table(bag_dir):
    """Return {topic_name: (type_string, message_count)}."""
    con = sqlite3.connect(_db3_path(bag_dir))
    rows = con.execute(
        "SELECT t.name, t.type, COUNT(m.id) FROM topics t "
        "LEFT JOIN messages m ON m.topic_id = t.id GROUP BY t.id").fetchall()
    con.close()
    return {n: (ty, c) for n, ty, c in rows}


def _decode_skeleton(msg):
    """zed_msgs/ObjectsStamped -> list[Body].

    ZED packs BODY_38 as `skeleton_3d.keypoints`, each with `.kp` (xyz).
    Older layouts expose `keypoint_3d`. Both are handled. Invalid keypoints
    come through as NaN or as exact zeros, and both are treated as missing.
    """
    bodies = []
    for obj in msg.objects:
        kp = np.full((N_KP, 3), np.nan)
        conf = np.full(N_KP, np.nan)

        raw = None
        if hasattr(obj, "skeleton_3d") and hasattr(obj.skeleton_3d, "keypoints"):
            raw = obj.skeleton_3d.keypoints
        elif hasattr(obj, "keypoint_3d"):
            raw = obj.keypoint_3d

        if raw is not None:
            for i, k in enumerate(raw):
                if i >= N_KP:
                    break
                v = getattr(k, "kp", k)
                kp[i] = (float(v[0]), float(v[1]), float(v[2])) \
                    if not hasattr(v, "x") else (float(v.x), float(v.y), float(v.z))

        kc = getattr(obj, "keypoint_confidence", None)
        if kc is not None:
            for i, c in enumerate(kc):
                if i >= N_KP:
                    break
                conf[i] = float(c)

        finite = np.isfinite(kp).all(axis=1)
        nonzero = np.abs(kp).sum(axis=1) > 1e-9
        valid = finite & nonzero
        kp[~valid] = np.nan

        bodies.append(Body(
            id=int(getattr(obj, "label_id", -1)),
            keypoints=kp, valid=valid, confidence=conf,
            tracking_state=int(getattr(obj, "tracking_state", -1)),
            label_id=int(getattr(obj, "label_id", -1)),
        ))
    return bodies


def read_frames(bag_dir, topic=SKEL_TOPIC, max_frames=None):
    """Yield Frame objects in timestamp order."""
    deserialize_message, get_message = _import_ros()
    con = sqlite3.connect(_db3_path(bag_dir))
    row = con.execute("SELECT id, type FROM topics WHERE name = ?",
                      (topic,)).fetchone()
    if row is None:
        con.close()
        raise KeyError(f"topic {topic} not in bag. "
                       f"Available: {list(topic_table(bag_dir))}")
    tid, tstr = row
    msg_type = get_message(tstr)

    cur = con.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? "
        "ORDER BY timestamp", (tid,))
    t0 = None
    n = 0
    for ts, blob in cur:
        msg = deserialize_message(bytes(blob), msg_type)
        t_abs = ts * 1e-9
        if t0 is None:
            t0 = t_abs
        yield Frame(t=t_abs - t0, bodies=_decode_skeleton(msg), t_abs=t_abs)
        n += 1
        if max_frames and n >= max_frames:
            break
    con.close()


def read_depth(bag_dir, topic=DEPTH_TOPIC, max_frames=None):
    """Yield (t, HxW float32 metres) depth images."""
    deserialize_message, get_message = _import_ros()
    con = sqlite3.connect(_db3_path(bag_dir))
    row = con.execute("SELECT id, type FROM topics WHERE name = ?",
                      (topic,)).fetchone()
    if row is None:
        con.close()
        return
    tid, tstr = row
    msg_type = get_message(tstr)
    cur = con.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? "
        "ORDER BY timestamp", (tid,))
    t0 = None
    n = 0
    for ts, blob in cur:
        msg = deserialize_message(bytes(blob), msg_type)
        t_abs = ts * 1e-9
        if t0 is None:
            t0 = t_abs
        arr = np.frombuffer(msg.data, dtype=np.float32).reshape(
            msg.height, msg.width)
        yield t_abs - t0, arr
        n += 1
        if max_frames and n >= max_frames:
            break
    con.close()


def read_camera_info(bag_dir, topic=DEPTH_INFO_TOPIC):
    """Return (fx, fy, cx, cy) from the first camera_info message, or None."""
    deserialize_message, get_message = _import_ros()
    con = sqlite3.connect(_db3_path(bag_dir))
    row = con.execute("SELECT id, type FROM topics WHERE name = ?",
                      (topic,)).fetchone()
    if row is None:
        con.close()
        return None
    tid, tstr = row
    msg_type = get_message(tstr)
    r = con.execute("SELECT data FROM messages WHERE topic_id = ? LIMIT 1",
                    (tid,)).fetchone()
    con.close()
    if r is None:
        return None
    msg = deserialize_message(bytes(r[0]), msg_type)
    k = msg.k
    return float(k[0]), float(k[4]), float(k[2]), float(k[5])


def observed_mask(body, conf_min=None):
    """The paper's definition of 'observed': geometrically valid AND, where
    per-keypoint confidence is available, meeting the capture threshold.

    The SDK's internal prediction window cannot be disabled, so keypoints it
    predicts can arrive looking like detections. Gating on per-keypoint
    confidence excludes low-confidence predictions from resetting observation
    age. Where confidence is unavailable (NaN), geometric validity is the
    only signal and is used alone; the residual contamination is bounded by
    the 0.5 s window and is stated in the paper.
    """
    v = body.valid.copy()
    if conf_min is None:
        return v
    c = body.confidence
    has_conf = np.isfinite(c)
    v[has_conf] &= c[has_conf] >= float(conf_min)
    return v
