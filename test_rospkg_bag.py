#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message

# 可选：图像导出
try:
    from cv_bridge import CvBridge
    import cv2

    BRIDGE = CvBridge()
except Exception:
    BRIDGE = None
    cv2 = None


def open_reader(uri: str, storage_id: Optional[str]) -> rosbag2_py.SequentialReader:
    # uri 是 bag 目录（不是单个文件）。storage_id 通常是 'sqlite3' 或 'mcap'（Jazzy 默认）
    storage_options = rosbag2_py.StorageOptions(uri=uri, storage_id=storage_id or "")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def topic_type_map(reader: rosbag2_py.SequentialReader) -> Dict[str, str]:
    return {t.name: t.type for t in reader.get_all_topics_and_types()}


def set_topic_filter(reader: rosbag2_py.SequentialReader, topics: Optional[List[str]]):
    if topics:
        filt = rosbag2_py.StorageFilter(topics=topics)
        reader.set_filter(filt)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def maybe_write_csv_header(csv_path: Path, header: List[str]):
    if not csv_path.exists():
        csv_path.write_text(",".join(header) + "\n", encoding="utf-8")


def append_csv_row(csv_path: Path, row: List[str]):
    with csv_path.open("a", encoding="utf-8") as f:
        f.write(",".join(row) + "\n")


def export_odom_like(msg, t, csv_path: Path):
    # 适配 nav_msgs/Odometry，或包含 pose.pose.position/ orientation 的结构
    try:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        row = [f"{t / 1e9:.9f}", f"{p.x}", f"{p.y}", f"{p.z}", f"{q.x}", f"{q.y}", f"{q.z}", f"{q.w}"]
        append_csv_row(csv_path, row)
    except Exception:
        # 忽略不匹配的消息
        pass


def export_image(topic: str, msg, t, out_dir: Path):
    if BRIDGE is None or cv2 is None:
        return
    # 仅处理 sensor_msgs/Image
    if msg.__class__.__name__ != "Image":
        return
    cv_img = BRIDGE.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    fn = out_dir / f"{topic.strip('/').replace('/', '_')}_{int(t)}.png"
    cv2.imwrite(str(fn))


def main():
    parser = argparse.ArgumentParser(description="Decode ROS 2 bag with rosbag2_py.")
    parser.add_argument("--bag", required=True, help="Path to the bag directory (folder).")
    parser.add_argument(
        "--storage", default=None, help="Storage plugin id: 'sqlite3' or 'mcap'. Auto-detect if omitted."
    )
    parser.add_argument("--topics", nargs="*", help="Optional: topics to read only.")
    parser.add_argument(
        "--export-odom-csv", default=None, help="CSV path to export odom-like messages (e.g., nav_msgs/Odometry)."
    )
    parser.add_argument(
        "--export-images-dir", default=None, help="Directory to export sensor_msgs/Image frames as PNG."
    )
    parser.add_argument("--max", type=int, default=0, help="Max messages to read (0 = no limit).")
    args = parser.parse_args()

    bag_dir = Path(args.bag)
    if not bag_dir.exists():
        raise FileNotFoundError(f"Bag directory not found: {bag_dir}")

    # 尝试自动判断存储类型
    storage_id = args.storage
    if storage_id is None:
        if any(p.suffix == ".mcap" for p in bag_dir.rglob("*.mcap")):
            storage_id = "mcap"
        else:
            storage_id = "sqlite3"  # 传统默认

    reader = open_reader(str(bag_dir), storage_id)
    tmap = topic_type_map(reader)
    if not tmap:
        print("No topics found in bag.")
        return

    print("Topics and types:")
    for k, v in tmap.items():
        print(f"  {k}: {v}")

    # 过滤 topic（可选）
    set_topic_filter(reader, args.topics)

    # 准备导出
    csv_path = Path(args.export_odom_csv) if args.export_odom_csv else None
    if csv_path:
        ensure_dir(csv_path.parent)
        maybe_write_csv_header(csv_path, ["stamp(s)", "x", "y", "z", "qx", "qy", "qz", "qw"])

    img_dir = Path(args.export_images_dir) if args.export_images_dir else None
    if img_dir:
        ensure_dir(img_dir)

    # 缓存 type→class
    typeclass_cache: Dict[str, object] = {}

    def get_class(ros_type: str):
        if ros_type not in typeclass_cache:
            typeclass_cache[ros_type] = get_message(ros_type)
        return typeclass_cache[ros_type]

    count = 0
    while reader.has_next():
        topic, data, t = reader.read_next()  # t: nanoseconds
        ros_type = tmap.get(topic)
        if ros_type is None:
            continue
        msg_cls = get_class(ros_type)
        msg = deserialize_message(data, msg_cls)

        # 示例打印
        # 注意：t 是纳秒整型，这里不转成 rclpy.Time 以减少依赖
        if count < 5:
            print(f"[{count}] {topic} ({ros_type}) @ {t} ns -> {msg.__class__.__name__}")

        # 导出示例：里程计
        if csv_path and ros_type.endswith("nav_msgs/msg/Odometry"):
            export_odom_like(msg, t, csv_path)

        # 导出示例：图像
        if img_dir and ros_type.endswith("sensor_msgs/msg/Image"):
            export_image(topic, msg, t, img_dir)

        count += 1
        if args.max > 0 and count >= args.max:
            break

    print(f"Done. Read {count} messages.")
    if csv_path:
        print(f"Odometry CSV -> {csv_path}")
    if img_dir:
        print(f"Images dir -> {img_dir}")


if __name__ == "__main__":
    main()
