from pathlib import Path
from typing import cast

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores
from rosbags.typesys import get_typestore
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

bagpath = Path("/home/charles/tmp/green_block/episode_0")
typestore = get_typestore(Stores.ROS2_HUMBLE)


def decode_img(msg: CompressedImage) -> np.ndarray:
    np_arr = np.frombuffer(msg.data, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode failed for a CompressedImage frame.")
    return img


if __name__ == "__main__":
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        for c in reader.connections:
            print(c.topic)

        print()

        connections = []

        while not connections.__len__():
            topic = input("输入想要查看的Topic:")

            connections = [c for c in reader.connections if c.topic == topic]

        for connection, _timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)

            print(msg)
            if connection.msgtype == "sensor_msgs/msg/CompressedImage":
                msg_compress: CompressedImage = cast(CompressedImage, msg)
                img = decode_img(msg_compress)
                print("==> ", img.shape)
            elif connection.msgtype == "std_msgs/msg/Float64":
                print("==>", cast(Float64, msg).data)
            elif connection.msgtype == "sensor_msgs/msg/JointState":
                print(cast(JointState, msg).position.__len__())
                print("joints ==>", cast(JointState, msg).position)
            else:
                print("Type is: ", connection.msgtype)

            # print(msg)
