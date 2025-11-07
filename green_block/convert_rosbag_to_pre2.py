from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, cast
import json

import cv2
import numpy as np
import tqdm
import tyro

from rosbags.highlevel import AnyReader
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float64

PROMPT = ""
NS2S = 1e-9


@dataclass(frozen=True)
class PrepConfig:
    frame_rate: float = 30.0
    decode_flags_cam_mid: int = cv2.IMREAD_COLOR
    decode_flags_wrist: int = cv2.IMREAD_COLOR
    image_ext: Literal["jpg", "png"] = "jpg"
    image_quality: int = 95


# ---------- utils ----------
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _decode_img_from_bytes(buf: bytes, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    np_arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(np_arr, flags)
    if img is None:
        raise ValueError("cv2.imdecode failed")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _imwrite_rgb(path: Path, rgb: np.ndarray, ext: str, quality: int):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if ext == "jpg":
        cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    else:
        cv2.imwrite(str(path), bgr)


def _stack_pairs(pairs: Sequence[tuple[float, np.ndarray | float]]):
    if not pairs:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)
    pairs_sorted = sorted(pairs, key=lambda x: x[0])
    t = np.asarray([p[0] for p in pairs_sorted], dtype=np.float64)
    v0 = pairs_sorted[0][1]
    if isinstance(v0, np.ndarray):
        v = np.stack([p[1] for p in pairs_sorted], axis=0)
    else:
        v = np.asarray([p[1] for p in pairs_sorted], dtype=np.float64)[:, None]
    return t, v


def _last_le_index(tq: float, t: np.ndarray) -> int:
    """
    返回 t 中满足 t[idx] <= tq 的最后一个 idx；若不存在返回 -1
    假设 t 已升序
    """
    k = np.searchsorted(t, tq, side="right") - 1
    return int(k) if k >= 0 else -1


# ---------- pass 1: collect all topics ----------
def _collect_all(ep_path: Path):
    images_topics = {
        "/cam_mid/cam_mid_realsense/color/image_raw/compressed",
        "/cam_left/cam_left_realsense/color/image_raw/compressed",
        "/cam_right/cam_right_realsense/color/image_raw/compressed",
    }
    joints_topics = {"/left/joint_states", "/right/joint_states"}
    hands_topics = {"/left/teleop/hand_val", "/right/teleop/hand_val"}
    wanted = images_topics | joints_topics | hands_topics

    t_mid_ns, t_left_ns, t_right_ns = [], [], []
    tlq_pairs: list[tuple[float, np.ndarray]] = []
    trq_pairs: list[tuple[float, np.ndarray]] = []
    tlh_pairs: list[tuple[float, float]] = []
    trh_pairs: list[tuple[float, float]] = []

    with AnyReader([ep_path]) as reader:
        conns = [c for c in reader.connections if c.topic in wanted]
        for conn, t_ns, raw in reader.messages(connections=conns):
            topic = conn.topic
            mt = conn.msgtype
            if topic in images_topics and mt == "sensor_msgs/msg/CompressedImage":
                if topic.startswith("/cam_mid/"):
                    t_mid_ns.append(t_ns)
                elif topic.startswith("/cam_left/"):
                    t_left_ns.append(t_ns)
                elif topic.startswith("/cam_right/"):
                    t_right_ns.append(t_ns)
            elif topic in joints_topics and mt == "sensor_msgs/msg/JointState":
                msg = reader.deserialize(raw, mt)
                arr = np.asarray(cast(JointState, msg).position, dtype=np.float64)
                if arr.shape != (7,):
                    raise RuntimeError("JointState.position must be length 7")
                if topic == "/left/joint_states":
                    tlq_pairs.append((t_ns * NS2S, arr))
                else:
                    trq_pairs.append((t_ns * NS2S, arr))
            elif topic in hands_topics and mt == "std_msgs/msg/Float64":
                msg = reader.deserialize(raw, mt)
                val = float(cast(Float64, msg).data)
                if topic == "/left/teleop/hand_val":
                    tlh_pairs.append((t_ns * NS2S, val))
                elif topic == "/right/teleop/hand_val":
                    tlh_pairs.append((t_ns * NS2S, val))
                    trh_pairs.append((t_ns * NS2S, val))

    # sort & stack
    t_mid_ns = np.sort(np.asarray(t_mid_ns, dtype=np.int64))
    t_left_ns = np.sort(np.asarray(t_left_ns, dtype=np.int64))
    t_right_ns = np.sort(np.asarray(t_right_ns, dtype=np.int64))

    tlq, lq = _stack_pairs(tlq_pairs)
    trq, rq = _stack_pairs(trq_pairs)
    tlh, lh = _stack_pairs(tlh_pairs)
    trh, rh = _stack_pairs(trh_pairs)

    return (t_mid_ns, t_left_ns, t_right_ns, tlq, lq, trq, rq, tlh, lh, trh, rh)


# ---------- build 30fps plan with ZOH ----------
def _plan_zoh_30fps(
    t_mid_ns: np.ndarray,
    t_left_ns: np.ndarray,
    t_right_ns: np.ndarray,
    tlq: np.ndarray,
    lq: np.ndarray,
    trq: np.ndarray,
    rq: np.ndarray,
    tlh: np.ndarray,
    lh: np.ndarray,
    trh: np.ndarray,
    rh: np.ndarray,
    *,
    frame_rate: float,
):
    # 基本可用性检查：至少要有相机与数值四路
    if min(len(t_mid_ns), len(t_left_ns), len(t_right_ns)) == 0:
        raise RuntimeError("相机帧时间戳为空")
    if min(len(tlq), len(trq), len(tlh), len(trh)) == 0:
        raise RuntimeError("关节或手势时间序列为空")

    # 转换为秒
    t_mid = t_mid_ns * NS2S
    t_left = t_left_ns * NS2S
    t_right = t_right_ns * NS2S

    # 统一时间窗：所有七路都至少覆盖
    t_start = max(t_mid[0], t_left[0], t_right[0], tlq[0], trq[0], tlh[0], trh[0])
    t_end = min(t_mid[-1], t_left[-1], t_right[-1], tlq[-1], trq[-1], tlh[-1], trh[-1])
    if t_end <= t_start:
        raise RuntimeError("多话题无共同时间窗")

    step = 1.0 / float(frame_rate)
    targets = np.arange(t_start, t_end + 1e-9, step, dtype=np.float64)

    # 计划：每个 tq 选择 <= tq 的最近项（若不存在则跳过该 tq）
    plan = []  # (tid, tq, ns_mid, ns_left, ns_right, state16[np.float32])
    tid = 0
    for tq in targets:
        im = _last_le_index(tq, t_mid)
        il = _last_le_index(tq, t_left)
        ir = _last_le_index(tq, t_right)
        ilq = _last_le_index(tq, tlq)
        irq = _last_le_index(tq, trq)
        ilh = _last_le_index(tq, tlh)
        irh = _last_le_index(tq, trh)

        if min(im, il, ir, ilq, irq, ilh, irh) < 0:
            # 有任何一路在 tq 时刻尚未出现，跳过该采样点
            continue

        ns_mid = int(t_mid_ns[im])
        ns_left = int(t_left_ns[il])
        ns_right = int(t_right_ns[ir])

        state = np.concatenate([lq[ilq].ravel(), [lh[ilh].item()], rq[irq].ravel(), [rh[irh].item()]]).astype(
            np.float32
        )
        assert state.shape == (16,)

        plan.append((tid, float(tq), ns_mid, ns_left, ns_right, state))
        tid += 1

    return plan


# ---------- pass 2: write frames & steps ----------
def _write_episode(ep_path: Path, out_ep_dir: Path, cfg: PrepConfig):
    (t_mid_ns, t_left_ns, t_right_ns, tlq, lq, trq, rq, tlh, lh, trh, rh) = _collect_all(ep_path)

    plan = _plan_zoh_30fps(
        t_mid_ns,
        t_left_ns,
        t_right_ns,
        tlq,
        lq,
        trq,
        rq,
        tlh,
        lh,
        trh,
        rh,
        frame_rate=cfg.frame_rate,
    )
    n = len(plan)
    if n == 0:
        print(f"[WARN] {ep_path} 无可写入帧（30fps + ZOH）")
        return 0

    frames_dir = out_ep_dir / "frames"
    _ensure_dir(frames_dir)

    # 为第二遍读取构建命中表
    wanted_mid: dict[int, list[tuple[int, float]]] = {}
    wanted_left: dict[int, list[tuple[int, float]]] = {}
    wanted_right: dict[int, list[tuple[int, float]]] = {}
    state_by_tid: dict[int, np.ndarray] = {}

    for tid, tq, ns_m, ns_l, ns_r, st in plan:
        wanted_mid.setdefault(ns_m, []).append((tid, tq))
        wanted_left.setdefault(ns_l, []).append((tid, tq))
        wanted_right.setdefault(ns_r, []).append((tid, tq))
        state_by_tid[tid] = st

    # 缓冲
    buf_mid: dict[int, bytes] = {}
    buf_left: dict[int, bytes] = {}
    buf_right: dict[int, bytes] = {}

    meta_f = (out_ep_dir / "steps.jsonl").open("w", encoding="utf-8")

    def _flush_if_ready(tid: int, tq: float):
        if tid in buf_mid and tid in buf_left and tid in buf_right:
            img_mid = _decode_img_from_bytes(buf_mid.pop(tid), cfg.decode_flags_cam_mid)
            img_left = _decode_img_from_bytes(buf_left.pop(tid), cfg.decode_flags_wrist)
            img_right = _decode_img_from_bytes(buf_right.pop(tid), cfg.decode_flags_wrist)

            stem = f"{tid:06d}"
            p_mid = frames_dir / f"{stem}_cam_mid.{cfg.image_ext}"
            p_lw = frames_dir / f"{stem}_cam_left_wrist.{cfg.image_ext}"
            p_rw = frames_dir / f"{stem}_cam_right_wrist.{cfg.image_ext}"
            _imwrite_rgb(p_mid, img_mid, cfg.image_ext, cfg.image_quality)
            _imwrite_rgb(p_lw, img_left, cfg.image_ext, cfg.image_quality)
            _imwrite_rgb(p_rw, img_right, cfg.image_ext, cfg.image_quality)

            state = state_by_tid[tid].tolist()
            step = {
                "index_in_episode": int(tid),
                "timestamp": float(tq),
                "observation": {
                    "state": state,
                    "mid": str(p_mid.name),
                    "left": str(p_lw.name),
                    "right": str(p_rw.name),
                },
                "action": state,  # 与原版保持一致：这里复用 observation state
                "reward": 0.0,
                "discount": 1.0,
                "is_first": bool(tid == 0),
                "is_last": bool(tid == n - 1),
                "is_terminal": False,
                "language_instruction": PROMPT,
            }
            meta_f.write(json.dumps(step, ensure_ascii=False) + "\n")

    topics = {
        "/cam_mid/cam_mid_realsense/color/image_raw/compressed",
        "/cam_left/cam_left_realsense/color/image_raw/compressed",
        "/cam_right/cam_right_realsense/color/image_raw/compressed",
    }
    with AnyReader([ep_path]) as reader:
        conns = [c for c in reader.connections if c.topic in topics]
        for conn, t_ns, raw in reader.messages(connections=conns):
            if conn.msgtype != "sensor_msgs/msg/CompressedImage":
                continue
            if conn.topic.startswith("/cam_mid/"):
                lst = wanted_mid.get(int(t_ns))
                if lst:
                    msg = reader.deserialize(raw, conn.msgtype)
                    for tid, tq in lst:
                        buf_mid[tid] = bytes(cast(CompressedImage, msg).data)
                        _flush_if_ready(tid, tq)
            elif conn.topic.startswith("/cam_left/"):
                lst = wanted_left.get(int(t_ns))
                if lst:
                    msg = reader.deserialize(raw, conn.msgtype)
                    for tid, tq in lst:
                        buf_left[tid] = bytes(cast(CompressedImage, msg).data)
                        _flush_if_ready(tid, tq)
            elif conn.topic.startswith("/cam_right/"):
                lst = wanted_right.get(int(t_ns))
                if lst:
                    msg = reader.deserialize(raw, conn.msgtype)
                    for tid, tq in lst:
                        buf_right[tid] = bytes(cast(CompressedImage, msg).data)
                        _flush_if_ready(tid, tq)

    meta_f.close()

    with (out_ep_dir / "episode_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "num_steps": int(n),
                "n_bad_interp": 0,  # 不做插值与容差判定
                "file_path": str(ep_path),
                "frame_rate": float(cfg.frame_rate),
                "sampling": "ZOH(last<=tq) on all topics",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return n


# ---------- CLI ----------
def main(
    raw_dir: Path,
    out_root: Path,
    prompt: str,
    episodes: list[int] | None = None,
    image_ext: Literal["jpg", "png"] = "jpg",
    frame_rate: float = 30.0,
):
    """
    重放所有话题；按 30fps 统一时间轴采样；
    对每个采样时刻 tq 为各话题选择最后一条 (time<=tq) —— 零阶保持；
    输出格式/命名保持不变。
    """
    global PROMPT
    PROMPT = prompt
    cfg = PrepConfig(image_ext=image_ext, frame_rate=frame_rate)

    rosbag_files = sorted(raw_dir.glob("episode_0*"))
    print(f"Find {len(rosbag_files)} in {raw_dir}: {rosbag_files}")
    if episodes is None:
        episodes = list(range(len(rosbag_files)))

    out_root = out_root.resolve()
    _ensure_dir(out_root)

    all_index = []
    for ep_idx in tqdm.tqdm(episodes, desc="replay zoh 30fps"):
        ep_path = rosbag_files[ep_idx]
        ep_dir = out_root / f"ep_{ep_idx:06d}"
        _ensure_dir(ep_dir)
        n = _write_episode(ep_path, ep_dir, cfg)
        if n > 0:
            all_index.append({"episode_id": f"ep_{ep_idx:06d}", "num_steps": int(n)})

    with (out_root / "index.json").open("w", encoding="utf-8") as f:
        json.dump(all_index, f, ensure_ascii=False, indent=2)
    print(f"Done. Episodes written: {len(all_index)} at {out_root}")


# 示例：
# uv run green_block/convert_rosbag_to_pre2.py --raw_dir /home/charles/tmp/green_block --out_root runtime/pre_green2 --prompt 'Put the green block on the cloth.' --frame_rate 30 --cam_sync_tol_s 0.1
if __name__ == "__main__":
    tyro.cli(main)
