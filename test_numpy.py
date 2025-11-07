import orbax.checkpoint as ocp
from transformers import AutoProcessor

x = {"a": {"b": 1, "c": 2}}
y = {"a": {"b": 3, "c": 4}}

res = ocp.transform_utils.intersect_trees(x, y)
print(res)

path = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
print(path)
