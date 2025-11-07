import json

import matplotlib.pyplot as plt


def show(data: dict):
    """
    data: dict, 形如 {"name1": [y1, y2, ...], "name2": [y1, y2, ...], ...}
    x 轴自动为递增序列 [0, 1, 2, ...]
    """
    plt.figure(figsize=(8, 5))

    for name, values in data.items():
        plt.plot(range(len(values)), values, label=name)

    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Curve Plot")
    plt.legend()
    plt.grid(visible=True)
    plt.savefig("output.png")


file_path = "/home/charles/workspace/openpi/runtime/pre_green/ep_000020/steps.jsonl"

keys = ["j0", "j1", "j2", "j3", "j4", "j5", "j6", "hand"]
actions = [[] for _ in range(8)]
with open(file_path, encoding="utf-8") as f:
    for line in f:
        # 每一行都是一个 JSON 对象
        data = json.loads(line.strip())
        for out, inp in zip(actions, data["action"][-8:], strict=True):
            out.append(inp)

show(dict(zip(keys, actions, strict=True)))
