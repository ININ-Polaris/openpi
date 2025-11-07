import tensorflow_datasets as tfds

# 读取数据集
ds = tfds.load("pick_block", data_dir="/home/charles/workspace/openpi/runtime/output/dubhe_rlds_episode", split="train")

print(ds)  # 输出 Dataset 对象信息
