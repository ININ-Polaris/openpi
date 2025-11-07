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
topic = "/right/joint_states"
typestore = get_typestore(Stores.ROS2_HUMBLE)


if __name__ == "__main__":
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        for c in reader.connections:
            print(c.topic)

        print()

        connections = []

        while not connections.__len__():
            connections = [c for c in reader.connections if c.topic == topic]

        for connection, _timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)

            print(cast(JointState, msg).position.tolist())
