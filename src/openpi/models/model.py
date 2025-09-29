import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Any, Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch

from openpi.models_pytorch import pi0_pytorch
from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

# Type variable for array types (JAX arrays, PyTorch tensors, or numpy arrays)
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"


# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)


# 数据格式说明

# 数据变换（transforms）将模型的输入构造成一个嵌套字典，之后再被转换为 `Observation` 和 `Actions` 对象。如下所示：

# 在字典形式中，数据结构应如下：
# {
#     # 观测（Observation） 相关数据
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB 图像，像素值可为 [-1, 1] 或 [0, 255]
#         …  # 其他相机视图
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # 若对应视图的图像有效，则为 True
#         …  # 其他视图的 mask
#     },
#     "state": float32[*b, s],  # 低维机器人状态向量
#     "tokenized_prompt": int32[*b, l],  # （可选）语言提示 prompt 的 token 化表示
#     "tokenized_prompt_mask": bool[*b, l],  # （可选）tokenized prompt 的 mask
#     "token_ar_mask": int32[*b, l],  # （可选）用于 FAST 模型的自回归掩码
#     "token_loss_mask": bool[*b, l],  # （可选）用于 FAST 模型的损失掩码

#     # 操作（Actions） 相关数据
#     "actions": float32[*b, ah, ad]
# }

# 其中：
#   *b = 批次维度（batch dimensions）
#   h, w = 图像的高度和宽度
#   s = 状态向量的维度
#   l = 序列长度（token 序列长度）
#   ah = 动作头数（action heads）
#   ad = 每个头的动作维度


@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images, in [-1, 1] float32.
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*b s"]

    # Tokenized prompt.
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # If images are uint8, convert them to [-1, 1] float32.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions = at.Float[ArrayT, "*b ah ad"]


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """对 Observation 进行预处理，包括以下几步：

    1. 图像增强（如果 train=True）：对图像做随机裁剪、旋转、色彩扰动等变换；
    2. 图像尺寸调整（如果观测中的图像尺寸与目标分辨率不一致）：使用填充或缩放将其调整为给定的 image_resolution；
    3. 填充默认的图像掩码（如果 observation.image_masks 中缺少对应 key）：对于缺失的图像 mask，默认设置为全 True（即不进行遮掩）。

    参数：
        rng: 随机数生成器种子或状态，用于图像增强时生成随机变换；
        observation: 原始的 Observation 对象；
        train: 是否以训练模式进行增强（仅在 True 时应用增强）；
        image_keys: 要处理的图像 key 列表（即 observation.images 中要处理的那些键）；
        image_resolution: 目标图像分辨率 (高度, 宽度)；

    返回：
        一个新的 Observation 对象，其 images 和 image_masks 已经过上述预处理。
    """

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image: at.Float[at.Array, "h w c"] = observation.images[key]
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            if rng is None:
                raise ValueError("Rng must be given when train is True.")

            sub_rngs = jax.random.split(rng, image.shape[0])
            image: at.Float[Any, "h w c"] = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """所有模型共用的配置。具体模型应继承此类，并实现 `create` 方法以创建对应的模型。"""

    # 动作空间的维度
    action_dim: int
    # 动作序列的长度
    action_horizon: int
    # 分词后 prompt 的最大长度
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """模型的类型。"""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """创建一个新的模型，并初始化参数。"""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """用给定的参数构建模型（加载参数）。"""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            # 用于正交两颗 PyTree
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)  # type: ignore
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str):
        logger.info(f"train_config: {train_config}")
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """返回模型的输入规格（spec）。这些规格是 jax.ShapeDtypeStruct 类型。"""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """
    所有模型实现的基类。具体的模型应当继承这个类。它们应该调用
    super().__init__() 来初始化共享属性（action_dim, action_horizon, 和 max_token_len）。
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...
    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """从检查点恢复无结构参数 PyTree

    这个函数能处理在 openpi 训练过程中通过 `save_state` 保存的检查点（参见 `training/checkpoints.py`），
    也能处理为 openpi 发布的预训练检查点

    参数:
        params_path: 检查点的路径
        restore_type: 恢复参数时使用的类型。可以设为 `np.ndarray`，将参数加载为 numpy 数组。
        dtype: 用于恢复所有参数的数据类型 (dtype)。如果未提供，则使用检查点中原有的 dtype。
        sharding: 参数的分片 (sharding) 方式。如果未提供，参数将在所有设备上被复制 (replicated)。

    返回:
        恢复后的参数 (params)。
    """
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,  # type: ignore
                restore_args=jax.tree.map(  # type: ignore
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # 如果这些参数是用 openpi 训练过程中的 `save_state` 保存的，
    # 则每一个键路径 (key path) 最后会以 "value" 结尾，这是由 `nnx.State` 添加的。
    # 我们在这里移除 "value" 后缀，并始终返回 NNX 所说的 “纯 dict”。
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
