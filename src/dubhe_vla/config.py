from collections.abc import Sequence
import dataclasses
import pathlib

from typing_extensions import override
import tyro

from dubhe_vla import policy
from openpi.models import pi0_fast
import openpi.models.model as _model
from openpi.training import weight_loaders
import openpi.training.config as _config
import openpi.transforms as _transforms


@dataclasses.dataclass(frozen=True)
class LeRobotDubheDataConfig(_config.DataConfigFactory):
    # 如果为 True，则在传递给模型之前，将关节尺寸转换为相对于当前状态的delta。
    # 抓手尺寸将保持绝对值。
    use_delta_joint_actions: bool = False
    # 如果提供且 prompt 键不存在，将被注入到输入数据中
    default_prompt: str | None = None

    # 如果为真，这将把关节和夹持器的值从标准 Dubhe 空间转换为 pi 内部运行时使用的空间，
    # 用于训练基本模型。
    adapt_to_pi: bool = False

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_mid": "observation.images.cam_mid",
                            "cam_left_wrist": "observation.images.cam_left_wrist",
                            "cam_right_wrist": "observation.images.cam_right_wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )

    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        data_transforms = _transforms.Group(
            inputs=[policy.DubheInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[policy.DubheOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            # TODO
            raise NotImplementedError()
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = _config.ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


def get_dubhe_config(overwrite: bool = False):  # noqa: FBT001, FBT002
    return _config.TrainConfig(
        name="dubhe_low_mem",
        project_name="dubhe",
        exp_name="test",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=16, action_horizon=10, max_token_len=250, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotDubheDataConfig(
            repo_id="test", use_delta_joint_actions=False, adapt_to_pi=False, default_prompt="test"
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/charles/workspace/models/openpi/openpi-assets/checkpoints/pi0_fast_libero/params"
        ),
        num_train_steps=30_000,
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=16, action_horizon=10, max_token_len=250, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="/home/charles/workspace/openpi/assets",
        seed=42,
        overwrite=overwrite,
        num_workers=1,
    )
