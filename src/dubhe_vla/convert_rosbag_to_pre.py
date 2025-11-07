# rosbag_to_rlds_prep.py
from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import json
from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import tqdm
import tyro

PROMPT = ""


# ===========================
# 与你已有配置保持一致（稍有精简）
# ===========================
@dataclasses.dataclass(frozen=True)
class PrepConfig:
    cam_sync_tol_s: float = 0.1  # 三相机对齐容差
    interp_max_gap_s: float = 0.030  # 数值插值最大间隔
    decode_flags_cam_mid: int = cv2.IMREAD_COLOR
    decode_flags_wrist: int = cv2.IMREAD_COLOR
    image_ext: Literal["jpg", "png"] = "jpg"
    image_quality: int = 95  # jpg 质量（若 png 则忽略）


NS2S = 1e-9


# ============== 基础工具 ==============
def _decode_img_from_bytes(buf: bytes, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    np_arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(np_arr, flags)
    if img is None:
        raise ValueError("cv2.imdecode failed for a CompressedImage frame.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _linear_interp_at(tq: float, t: np.ndarray, y: np.ndarray):
    if t.ndim != 1:
        raise ValueError("t must be 1-D")
    y2 = y if y.ndim == 2 else y[:, None]
    if tq <= t[0]:
        return y2[0].copy(), False, float(abs(tq - t[0]))
    if tq >= t[-1]:
        return y2[-1].copy(), False, float(abs(tq - t[-1]))
    k = np.searchsorted(t, tq, side="left")
    t0, t1 = t[k - 1], t[k]
    y0, y1 = y2[k - 1], y2[k]
    alpha = (tq - t0) / (t1 - t0)
    yq = (1.0 - alpha) * y0 + alpha * y1
    nearest_gap = float(min(abs(tq - t0), abs(tq - t1)))
    return yq, True, nearest_gap


def _sync_triplets(t_mid: np.ndarray, t_left: np.ndarray, t_right: np.ndarray, tol_s: float = 0.010):
    i = j = k = 0
    out: list[tuple[int, int, int, float]] = []
    while i < len(t_mid) and j < len(t_left) and k < len(t_right):
        tm, tl, tr = t_mid[i], t_left[j], t_right[k]
        tmin, tmax = min(tm, tl, tr), max(tm, tl, tr)
        skew = tmax - tmin
        if skew <= tol_s:
            tref = float(np.median([tm, tl, tr]))
            out.append((i, j, k, tref))
            i += 1
            j += 1
            k += 1
        elif tm == tmin:
            i += 1
        elif tl == tmin:
            j += 1
        else:
            k += 1
    return out


# ============== 第 1 遍：只收集时间序列 ==============
def _collect_timeseries_only(ep_path: Path):
    images_topics = {
        "/cam_right/cam_right_realsense/color/image_raw/compressed",
        "/cam_mid/cam_mid_realsense/color/image_raw/compressed",
        "/cam_left/cam_left_realsense/color/image_raw/compressed",
    }
    joints_topics = {"/left/joint_states", "/right/joint_states"}
    hands_topics = {"/left/teleop/hand_val", "/right/teleop/hand_val"}
    wanted_topics = images_topics | joints_topics | hands_topics

    t_mid_ns, t_lw_ns, t_rw_ns = [], [], []
    tlq_pairs, trq_pairs, tlh_pairs, trh_pairs = [], [], [], []

    with AnyReader([ep_path]) as reader:
        connections = [c for c in reader.connections if c.topic in wanted_topics]
        for connection, t_ns, raw in reader.messages(connections=connections):
            topic = connection.topic
            msg_type = connection.msgtype
            if topic in images_topics and msg_type == "sensor_msgs/msg/CompressedImage":
                if topic.endswith("/cam_mid_realsense/color/image_raw/compressed"):
                    t_mid_ns.append(t_ns)
                elif topic.startswith("/cam_left/"):
                    t_lw_ns.append(t_ns)
                else:
                    t_rw_ns.append(t_ns)
            elif topic in joints_topics and msg_type == "sensor_msgs/msg/JointState":
                msg = reader.deserialize(raw, msg_type)
                arr = np.asarray(cast(JointState, msg).position, dtype=np.float64)
                if arr.shape != (7,):
                    raise RuntimeError("JointState.position must be length 7")
                if topic == "/left/joint_states":
                    tlq_pairs.append((t_ns * NS2S, arr))
                else:
                    trq_pairs.append((t_ns * NS2S, arr))
            elif topic in hands_topics and msg_type == "std_msgs/msg/Float64":
                msg = reader.deserialize(raw, msg_type)
                val = float(cast(Float64, msg).data)
                # 如你原注释，这里可根据需要区分左右手；默认同值写入
                tlh_pairs.append((t_ns * NS2S, val))
                trh_pairs.append((t_ns * NS2S, val))

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

    tlq, lq = _stack_pairs(tlq_pairs)
    trq, rq = _stack_pairs(trq_pairs)
    tlh, lh = _stack_pairs(tlh_pairs)
    trh, rh = _stack_pairs(trh_pairs)

    return (
        np.asarray(t_mid_ns, dtype=np.int64),
        np.asarray(t_lw_ns, dtype=np.int64),
        np.asarray(t_rw_ns, dtype=np.int64),
        tlq,
        lq,
        trq,
        rq,
        tlh,
        lh,
        trh,
        rh,
    )


def _build_triplets_and_interp_states(
    t_mid_ns: np.ndarray,
    t_lw_ns: np.ndarray,
    t_rw_ns: np.ndarray,
    tlq: np.ndarray,
    lq: np.ndarray,
    trq: np.ndarray,
    rq: np.ndarray,
    tlh: np.ndarray,
    lh: np.ndarray,
    trh: np.ndarray,
    rh: np.ndarray,
    *,
    cam_sync_tol_s: float,
    interp_max_gap_s: float,
):
    t_mid = t_mid_ns * NS2S
    t_lw = t_lw_ns * NS2S
    t_rw = t_rw_ns * NS2S

    if min(len(t_mid), len(t_lw), len(t_rw)) == 0:
        raise RuntimeError("相机帧时间戳为空，请检查数据")
    if min(len(tlq), len(trq), len(tlh), len(trh)) == 0:
        raise RuntimeError("机械臂或手势流为空，请检查数据")

    triplets = _sync_triplets(t_mid, t_lw, t_rw, tol_s=cam_sync_tol_s)
    if not triplets:
        raise RuntimeError(f"三相机在容差 {cam_sync_tol_s}s 内无可匹配三元组")

    mid_idx_to_tid = -np.ones(len(t_mid), dtype=np.int32)
    lw_idx_to_tid = -np.ones(len(t_lw), dtype=np.int32)
    rw_idx_to_tid = -np.ones(len(t_rw), dtype=np.int32)
    tref = np.empty(len(triplets), dtype=np.float64)
    for tid, (i, j, k, t) in enumerate(triplets):
        mid_idx_to_tid[i] = tid
        lw_idx_to_tid[j] = tid
        rw_idx_to_tid[k] = tid
        tref[tid] = t

    states = np.empty((len(triplets), 16), dtype=np.float32)
    n_bad = 0
    for n, t in enumerate(tref):
        lq_t, _, g1 = _linear_interp_at(t, tlq, lq)
        rq_t, _, g2 = _linear_interp_at(t, trq, rq)
        lh_t, _, g3 = _linear_interp_at(t, tlh, lh)
        rh_t, _, g4 = _linear_interp_at(t, trh, rh)
        ok = (
            (g1 <= interp_max_gap_s)
            and (g2 <= interp_max_gap_s)
            and (g3 <= interp_max_gap_s)
            and (g4 <= interp_max_gap_s)
        )
        if not ok:
            n_bad += 1
        states[n, :] = np.concatenate([lq_t.ravel(), lh_t.ravel(), rq_t.ravel(), rh_t.ravel()]).astype(np.float32)

    mid_ns_to_idx = {int(ns): idx for idx, ns in enumerate(t_mid_ns)}
    lw_ns_to_idx = {int(ns): idx for idx, ns in enumerate(t_lw_ns)}
    rw_ns_to_idx = {int(ns): idx for idx, ns in enumerate(t_rw_ns)}
    return (
        triplets,
        tref,
        states,
        n_bad,
        mid_idx_to_tid,
        lw_idx_to_tid,
        rw_idx_to_tid,
        mid_ns_to_idx,
        lw_ns_to_idx,
        rw_ns_to_idx,
    )


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _imwrite_rgb(path: Path, rgb: np.ndarray, ext: str, quality: int):
    if ext == "jpg":
        # OpenCV 期望 BGR
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    else:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)


def _write_episode_to_disk(
    ep_path: Path,
    out_ep_dir: Path,
    *,
    cfg: PrepConfig,
):
    (t_mid_ns, t_lw_ns, t_rw_ns, tlq, lq, trq, rq, tlh, lh, trh, rh) = _collect_timeseries_only(ep_path)
    (
        triplets,
        tref,
        states,
        n_bad,
        mid_idx_to_tid,
        lw_idx_to_tid,
        rw_idx_to_tid,
        mid_ns_to_idx,
        lw_ns_to_idx,
        rw_ns_to_idx,
    ) = _build_triplets_and_interp_states(
        t_mid_ns,
        t_lw_ns,
        t_rw_ns,
        tlq,
        lq,
        trq,
        rq,
        tlh,
        lh,
        trh,
        rh,
        cam_sync_tol_s=cfg.cam_sync_tol_s,
        interp_max_gap_s=cfg.interp_max_gap_s,
    )

    n = len(triplets)
    if n == 0:
        print(f"[WARN] {ep_path} 无可写入帧")
        return 0

    frames_dir = out_ep_dir / "frames"
    _ensure_dir(frames_dir)

    meta_f = (out_ep_dir / "steps.jsonl").open("w", encoding="utf-8")

    # 临时缓冲：tid -> 压缩 bytes；凑齐三帧就写盘
    buf_mid: dict[int, bytes] = {}
    buf_lw: dict[int, bytes] = {}
    buf_rw: dict[int, bytes] = {}

    def _flush_if_ready(tid: int):
        if tid in buf_mid and tid in buf_lw and tid in buf_rw:
            idx_in_ep = tid
            ts = tref[tid]
            # 解码
            img_mid = _decode_img_from_bytes(buf_mid.pop(tid), cfg.decode_flags_cam_mid)
            img_left = _decode_img_from_bytes(buf_lw.pop(tid), cfg.decode_flags_wrist)
            img_right = _decode_img_from_bytes(buf_rw.pop(tid), cfg.decode_flags_wrist)

            # 写出
            stem = f"{idx_in_ep:06d}"
            p_mid = frames_dir / f"{stem}_cam_mid.{cfg.image_ext}"
            p_lw = frames_dir / f"{stem}_cam_left_wrist.{cfg.image_ext}"
            p_rw = frames_dir / f"{stem}_cam_right_wrist.{cfg.image_ext}"
            _imwrite_rgb(p_mid, img_mid, cfg.image_ext, cfg.image_quality)
            _imwrite_rgb(p_lw, img_left, cfg.image_ext, cfg.image_quality)
            _imwrite_rgb(p_rw, img_right, cfg.image_ext, cfg.image_quality)

            # RLDS step 元数据（奖励/终止条件如无可按 0/False）
            step = {
                "index_in_episode": int(idx_in_ep),
                "timestamp": float(ts),
                "observation": {
                    "state": states[tid].tolist(),
                    # 改名并保持固定分辨率（你的相机尺寸已固定）
                    "mid": str(p_mid.name),  # 480x848x3 JPEG
                    "left": str(p_lw.name),  # 240x424x3 JPEG
                    "right": str(p_rw.name),  # 240x424x3 JPEG
                },
                "action": states[tid].tolist(),
                "reward": 0.0,
                "discount": 1.0,
                "is_first": bool(idx_in_ep == 0),
                "is_last": bool(idx_in_ep == n - 1),
                "is_terminal": False,
                "language_instruction": PROMPT,  # 如有指令可填实际字符串
            }
            meta_f.write(json.dumps(step, ensure_ascii=False) + "\n")

    # 第二遍：只读图像消息并按三元组写盘
    images_topics = {
        "/cam_right/cam_right_realsense/color/image_raw/compressed",
        "/cam_mid/cam_mid_realsense/color/image_raw/compressed",
        "/cam_left/cam_left_realsense/color/image_raw/compressed",
    }
    with AnyReader([ep_path]) as reader:
        connections = [c for c in reader.connections if c.topic in images_topics]
        for connection, t_ns, raw in reader.messages(connections=connections):
            topic = connection.topic
            msg_type = connection.msgtype
            if msg_type != "sensor_msgs/msg/CompressedImage":
                continue

            if topic.startswith("/cam_right/"):
                idx = rw_ns_to_idx.get(int(t_ns), None)
                if idx is None:
                    continue
                tid = rw_idx_to_tid[idx]
                if tid >= 0:
                    msg = reader.deserialize(raw, msg_type)
                    buf_rw[tid] = bytes(cast(CompressedImage, msg).data)
                    _flush_if_ready(tid)
            elif topic.startswith("/cam_left/"):
                idx = lw_ns_to_idx.get(int(t_ns), None)
                if idx is None:
                    continue
                tid = lw_idx_to_tid[idx]
                if tid >= 0:
                    msg = reader.deserialize(raw, msg_type)
                    buf_lw[tid] = bytes(cast(CompressedImage, msg).data)
                    _flush_if_ready(tid)
            else:
                idx = mid_ns_to_idx.get(int(t_ns), None)
                if idx is None:
                    continue
                tid = mid_idx_to_tid[idx]
                if tid >= 0:
                    msg = reader.deserialize(raw, msg_type)
                    buf_mid[tid] = bytes(cast(CompressedImage, msg).data)
                    _flush_if_ready(tid)

    meta_f.close()
    # 写一个 episode 元信息
    with (out_ep_dir / "episode_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "num_steps": int(n),
                "n_bad_interp": int(n_bad),
                "file_path": str(ep_path),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return n


def main(
    raw_dir: Path,
    out_root: Path,
    prompt: str,
    task: str = "debug",
    episodes: list[int] | None = None,
    image_ext: Literal["jpg", "png"] = "jpg",
):
    global PROMPT
    PROMPT = prompt
    cfg = PrepConfig(image_ext=image_ext)
    rosbag_files = sorted(raw_dir.glob(f"{task}_bag_*"))
    if episodes is None:
        episodes = list(range(len(rosbag_files)))
    out_root = out_root.resolve()
    _ensure_dir(out_root)
    all_index = []
    for ep_idx in tqdm.tqdm(episodes, desc="preprocess episodes"):
        ep_path = rosbag_files[ep_idx]
        ep_dir = out_root / f"ep_{ep_idx:06d}"
        _ensure_dir(ep_dir)
        n = _write_episode_to_disk(ep_path, ep_dir, cfg=cfg)
        if n > 0:
            all_index.append({"episode_id": f"ep_{ep_idx:06d}", "num_steps": int(n)})
    with (out_root / "index.json").open("w", encoding="utf-8") as f:
        json.dump(all_index, f, ensure_ascii=False, indent=2)
    print(f"Done. Episodes written: {len(all_index)} at {out_root}")


# uv run src/dubhe_vla/convert_rosbag_to_pre.py --raw_dir data/apple/ --out_root tf --task pick_apple --prompt 'Pick up the apple'
if __name__ == "__main__":
    tyro.cli(main)
