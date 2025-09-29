import orbax.checkpoint as ocp

x = {"a": {"b": 1, "c": 2}}
y = {"a": {"b": 3, "c": 4}}

res = ocp.transform_utils.intersect_trees(x, y)

print(res)
