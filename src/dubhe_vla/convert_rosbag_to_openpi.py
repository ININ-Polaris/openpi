from collections.abc import Sequence
import dataclasses
from pathlib import Path
import shutil
from typing import Literal, cast

import cv2
from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from rosbags.highlevel import AnyReader

# from lerobot.common.robot_devices.robots.utils import Robot
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import torch
import tqdm
import tyro


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = False
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()
NS2S = 1e-9


def _decode_img(msg: CompressedImage) -> np.ndarray:
    """HxWxC (BGR uint8)"""
    np_arr = np.frombuffer(msg.data, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode failed for a CompressedImage frame.")
    return img


def _linear_interp_at(tq: float, t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, bool, float]:
    """
    在标量时间轴 t 上对多维信号 y 做线性插值，返回 y(tq)，以及是否可靠、与最近样本的时间差（秒）。
    t: (N,), 单调递增; y: (N,D) 或 (N,)；tq: 标量秒
    """
    if t.ndim != 1:
        raise ValueError("t must be 1-D")
    y2 = y if y.ndim == 2 else y[:, None]
    # 边界处理（最近邻）
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


def _sync_triplets(
    t_mid: np.ndarray, t_left: np.ndarray, t_right: np.ndarray, tol_s: float = 0.010
) -> list[tuple[int, int, int, float]]:
    """
    把三路相机时间戳做“三元组”匹配，要求三者之间的最大时间差 <= tol_s。
    返回列表 [(i_mid, i_left, i_right, t_ref), ...]，t_ref 取三者时间的中位数。
    策略：三指针贪心推进，尽量保留更多可配对帧。
    """
    i = j = k = 0
    out: list[tuple[int, int, int, float]] = []
    while i < len(t_mid) and j < len(t_left) and k < len(t_right):
        tm, tl, tr = t_mid[i], t_left[j], t_right[k]
        tmin, tmax = min(tm, tl, tr), max(tm, tl, tr)
        skew = tmax - tmin
        if skew <= tol_s:
            tref = np.median([tm, tl, tr]).item()
            out.append((i, j, k, float(tref)))
            # 前进一步：谁的时间最小就推进谁（避免重复使用同一帧）
            if tm == tmin:
                i += 1
            if tl == tmin:
                j += 1
            if tr == tmin:
                k += 1
        # 推进最落后的那一路
        elif tm == tmin:
            i += 1
        elif tl == tmin:
            j += 1
        else:
            k += 1
    return out


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode="image",
    overwrite: bool = False,  # noqa: FBT001, FBT002
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    if overwrite and Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    motors = [
        "left_base",  # 基座
        "left_shoulder",  # 肩膀
        "left_upperarm_roll",  # 大臂
        "left_elbow",  # 肘
        "left_forearm_roll",  # 前臂 roll
        "left_wrist",  # 手腕
        "left_wrist_roll",  # 手腕
        "left_hand",  # 手
        "right_base",  # 基座
        "right_shoulder",  # 肩膀
        "right_upperarm_roll",  # 大臂
        "right_elbow",  # 肘
        "right_forearm_roll",  # 前臂 roll
        "right_wrist",  # 手腕
        "right_wrist_roll",  # 手腕
        "right_hand",  # 手
    ]

    cameras = {  # H W C
        "cam_mid": (480, 848, 3),
        "cam_left_wrist": (240, 424, 3),
        "cam_right_wrist": (240, 424, 3),
    }

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": motors,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": motors,
        },
    }

    for cam, shape in cameras.items():
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": shape,
            "names": [
                "height",
                "width",
                "channels",
            ],
        }

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        root=None,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def load_raw_episode_data(
    ep_path: Path,
    cam_sync_tol_s: float = 0.1,  # 三相机三元组匹配容差（秒）
    interp_max_gap_s: float = 0.030,  # 插值可接受的最大最近距离（秒）
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor]:
    """
    更优对齐版本：
    1) 先完整收集各话题(time, data)
    2) 三相机做“三元组”匹配形成同步帧（时间尽量贴近）
    3) 以三元组中位时间 t_ref，对两路机械臂与手势做线性插值对齐到 t_ref
    返回：
      imgs_per_cam: {"cam_mid": (N,H,W,C), "cam_left_wrist": (N, ...), "cam_right_wrist": (N, ...)}
      state: (N, 16) = [L7, Lh, R7, Rh]，float32
      action: (N, 16)，此处与你原逻辑保持相同（=state）
    """
    images_topics = {
        "/cam_right/cam_right_realsense/color/image_raw/compressed",
        "/cam_mid/cam_mid_realsense/color/image_raw/compressed",
        "/cam_left/cam_left_realsense/color/image_raw/compressed",
    }
    joints_topics = {"/left/joint_states", "/right/joint_states"}
    hands_topics = {"/left/teleop/hand_val", "/right/teleop/hand_val"}
    wanted_topics = images_topics | joints_topics | hands_topics

    # 离线存储各流
    imgs_store: dict[str, list[tuple[float, np.ndarray]]] = {t: [] for t in images_topics}
    lq_store: list[tuple[float, np.ndarray]] = []
    rq_store: list[tuple[float, np.ndarray]] = []
    lh_store: list[tuple[float, float]] = []
    rh_store: list[tuple[float, float]] = []

    with AnyReader([ep_path]) as reader:
        connections = [c for c in reader.connections if c.topic in wanted_topics]
        for connection, timestamp, raw in reader.messages(connections=connections):
            topic = connection.topic
            msg_type = connection.msgtype
            try:
                msg = reader.deserialize(raw, msg_type)
            except AssertionError:
                print(f"反序列化 {ep_path}[{topic}] 时失败")
                raise
            t_sec = timestamp * NS2S

            if topic in images_topics and msg_type == "sensor_msgs/msg/CompressedImage":
                cv_image = _decode_img(cast(CompressedImage, msg))
                imgs_store[topic].append((t_sec, cv_image))

            elif topic in joints_topics and msg_type == "sensor_msgs/msg/JointState":
                data = cast(JointState, msg).position
                if not isinstance(data, np.ndarray) and len(data) != 7:
                    raise RuntimeError("错误的数据类型，不是 Float64[7]")
                arr = np.asarray(data, dtype=np.float64)
                if topic == "/left/joint_states":
                    lq_store.append((t_sec, arr))
                else:
                    rq_store.append((t_sec, arr))

            elif topic in hands_topics and msg_type == "std_msgs/msg/Float64":
                val = float(cast(Float64, msg).data)
                if topic == "/left/teleop/hand_val":
                    lh_store.append((t_sec, val))
                else:
                    rh_store.append((t_sec, val))

    # 转为 numpy 数组并按时间排序（稳妥）
    def _to_ts_and_arr(pairs: Sequence[tuple[float, np.ndarray | float]]) -> tuple[np.ndarray, np.ndarray]:
        if not pairs:
            return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        t = np.asarray([p[0] for p in pairs_sorted], dtype=np.float64)
        v0 = pairs_sorted[0][1]
        if isinstance(v0, np.ndarray):
            v = np.stack([p[1] for p in pairs_sorted], axis=0)  # (N,D)
        else:
            v = np.asarray([p[1] for p in pairs_sorted], dtype=np.float64)[:, None]  # (N,1)
        return t, v

    t_mid, mid_imgs = _to_ts_and_arr(imgs_store["/cam_mid/cam_mid_realsense/color/image_raw/compressed"])
    t_lw, lw_imgs = _to_ts_and_arr(imgs_store["/cam_left/cam_left_realsense/color/image_raw/compressed"])
    t_rw, rw_imgs = _to_ts_and_arr(imgs_store["/cam_right/cam_right_realsense/color/image_raw/compressed"])

    tlq, lq = _to_ts_and_arr(lq_store)  # (Nlq,), (Nlq,7)
    trq, rq = _to_ts_and_arr(rq_store)  # (Nrq,), (Nrq,7)
    tlh, lh = _to_ts_and_arr(lh_store)  # (Nlh,), (Nlh,1)
    trh, rh = _to_ts_and_arr(rh_store)  # (Nrh,), (Nrh,1)

    # 基础健壮性检查
    if min(len(t_mid), len(t_lw), len(t_rw)) == 0:
        raise RuntimeError("相机帧为空，请检查数据")
    if min(len(tlq), len(trq), len(tlh), len(trh)) == 0:
        raise RuntimeError(f"机械臂或手势流为空，请检查数据 {ep_path}")

    # 三相机三元组匹配（尽量多保留可匹配帧）
    triplets = _sync_triplets(t_mid, t_lw, t_rw, tol_s=cam_sync_tol_s)

    if len(triplets) == 0:
        raise RuntimeError(f"在给定容差内（三相机）没有可匹配的三元组，请增大 cam_sync_tol_s({cam_sync_tol_s})")

    # 选择匹配到的相机帧与参考时间
    sel_mid = [mid_imgs[i] for i, _, _, _ in triplets]
    sel_lw = [lw_imgs[j] for _, j, _, _ in triplets]
    sel_rw = [rw_imgs[k] for _, _, k, _ in triplets]
    tref = np.asarray([t for _, _, _, t in triplets], dtype=np.float64)  # (N,)

    # 对齐高频流到 t_ref（线性插值；超出最大间隔则回退最近邻并标记）
    states = []
    n_bad = 0
    for t in tref:
        lq_t, ok1, g1 = _linear_interp_at(t, tlq, lq)  # (7,)
        rq_t, ok2, g2 = _linear_interp_at(t, trq, rq)
        lh_t, ok3, g3 = _linear_interp_at(t, tlh, lh)  # (1,)
        rh_t, ok4, g4 = _linear_interp_at(t, trh, rh)

        ok = (
            (g1 <= interp_max_gap_s)
            and (g2 <= interp_max_gap_s)
            and (g3 <= interp_max_gap_s)
            and (g4 <= interp_max_gap_s)
        )
        if not ok:
            n_bad += 1
        state_vec = np.concatenate([lq_t.ravel(), lh_t.ravel(), rq_t.ravel(), rh_t.ravel()]).astype(np.float32)  # (16,)
        states.append(state_vec)

    # 组装返回结构
    imgs_per_cam = {
        "cam_mid": np.asarray(sel_mid),
        "cam_left_wrist": np.asarray(sel_lw),
        "cam_right_wrist": np.asarray(sel_rw),
    }
    state = torch.from_numpy(np.stack(states, axis=0))  # (N,16) float32
    action = state.clone()  # 与原逻辑一致

    # 诊断信息
    print(f"[三元组匹配] 共匹配到 {len(triplets)} 帧；插值回退/低置信样本：{n_bad}")
    print(f"相机帧数：mid={len(t_mid)}, left={len(t_lw)}, right={len(t_rw)} → 同步后={len(triplets)}")
    print(f"机械臂样本：Lq={len(tlq)}, Rq={len(trq)}, Lh={len(tlh)}, Rh={len(trh)}")
    print(f"对齐容差：cam_sync_tol={cam_sync_tol_s * 1e3:.1f} ms, interp_max_gap={interp_max_gap_s * 1e3:.1f} ms")
    return imgs_per_cam, state, action


def populate_dataset(
    dataset: LeRobotDataset,
    rosbag_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = list(range(len(rosbag_files)))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = rosbag_files[ep_idx]
        imgs_per_cam, state, action = load_raw_episode_data(ep_path)

        num_frames = state.shape[0]

        img_frame_counts = [img_array.shape[0] for img_array in imgs_per_cam.values()]
        target_frame_count = min(state.shape[0], *img_frame_counts)

        # 生成等距索引
        indices = np.linspace(0, state.shape[0] - 1, target_frame_count).astype(int)
        state = state[indices]
        action = action[indices]

        num_frames = target_frame_count
        print(f"[INFO] Synced num_frames: {num_frames}")

        for i in range(num_frames):
            frame = {"observation.state": state[i], "action": action[i]}
            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]
            dataset.add_frame(frame, task=task)
        dataset.save_episode()

    return dataset


def port_dubhe(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "debug",
    overwrite: bool = False,  # noqa: FBT001, FBT002
    *,
    episodes: list[int] | None = None,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if not raw_dir.exists() and raw_repo_id is None:
        raise ValueError("raw_repo_id must be provided if raw_dir does not exist")

    rosbag_files = sorted(raw_dir.glob(f"{task}_bag_*"))
    print(rosbag_files)

    dataset = create_empty_dataset(
        repo_id, robot_type="dubhe", mode=mode, dataset_config=dataset_config, overwrite=overwrite
    )
    print(dataset)

    dataset = populate_dataset(dataset, rosbag_files, task=task, episodes=episodes)
    print(dataset)


# export HF_LEROBOT_HOME=~/workspace/models/lerobot/
if __name__ == "__main__":
    tyro.cli(port_dubhe)
