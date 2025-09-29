import logging
from typing import cast

import lerobot.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import tqdm
import tyro

from dubhe_vla.config import get_dubhe_config
import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)


class RemoveStrings(_transforms.DataTransformFn):
    """
    移除样本字典中所有 字符串类型 的条目, JAX 不支持 字符串张量
    """

    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def _create_torch_dataset(data_config: _config.DataConfig, action_horizon: int) -> _data_loader.Dataset:
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)

    logger.debug(f"Get dataset meta: {dataset_meta}")

    logger.info(f"Get dateset meta [Tasks]: {dataset_meta.tasks}")

    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },  # type: ignore
    )

    dataset = cast(_data_loader.Dataset, dataset)

    if data_config.prompt_from_task:
        dataset = _data_loader.TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_torch_dataloader(
    data_config: _config.DataConfig,  # 数据配置：数据源、变换、repo_id 等。
    action_horizon: int,  # 动作序列长度（窗口长度），来自模型配置。
    batch_size: int,  # 每批样本数。
    model_config: _model.BaseModelConfig,  # 模型配置（影响数据打包/特征规格等）。
    num_workers: int,  # DataLoader 工作进程数（并行加载）。
    max_frames: int | None = None,  # 可选上限：最多用于统计的帧数（子采样）。
) -> tuple[_data_loader.TorchDataLoader, int]:  # 返回类型注解：一个 数据集 和 num_batches
    # 基于模型规格创建底层 Torch 风格数据集。
    dataset = _create_torch_dataset(data_config, action_horizon)

    logger.debug(f"{data_config.repack_transforms.inputs=}")
    logger.debug(f"{data_config.data_transforms.inputs=}")

    # 用 可组合 的变换包装数据集：
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,  # 1) 重打包/结构化输入
            *data_config.data_transforms.inputs,  # 2) 其他数据增强/预处理
            RemoveStrings(),  # 3) 移除字符串字段
        ],
    )

    # 如果设置了上限且小于全集长度：
    if max_frames is not None and max_frames < len(dataset):
        # 计算用于统计的批次数（向下取整）
        num_batches = max_frames // batch_size
        # 开启打乱，避免只看前段数据导致偏差
        shuffle = True
    # 否则：用整个数据集的整批数
    else:
        num_batches = len(dataset) // batch_size
        # 不打乱（全量遍历时无必要）
        shuffle = False

    # 构建实际迭代器（按批产出字典批次）
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,  # 批大小
        num_workers=num_workers,  # 读盘/预处理进程数
        shuffle=shuffle,  # 是否洗牌
        num_batches=num_batches,  # 只取设定的批次数
    )
    # 返回 DataLoader 和 预计迭代总批数（用于 tqdm 的 total）
    return data_loader, num_batches


def main(max_frames: int | None = None):
    config = get_dubhe_config()

    # 结合资产目录/模型配置，实例化“可用的数据配置”
    data_config = config.data.create(config.assets_dirs, config.model)

    logger.debug(f"Create data config: {data_config}")

    # LeRobot: Torch 风格数据集
    data_loader, num_batches = create_torch_dataloader(
        data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
    )

    torch_loader = data_loader.torch_loader
    ds = cast(_data_loader.TransformedDataset, torch_loader.dataset)
    print("Dataset type:", type(ds))

    sample = ds[0]["image"]["base_0_rgb"]
    # print("Sample 0:", sample.shape)
    print("Sample Keys:", ds[0].keys())

    # 我们只统计这两类数值字段：机器人状态 & 动作
    keys = ["state", "actions"]
    # 为每个 key 构建在线统计器（均值/方差累计）。
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(
        data_loader,
        total=num_batches,
        desc="Computing stats",
    ):
        # 遍历要统计的字段
        for key in keys:
            # 将该字段的一个批次（转为 np.array）送入在线统计器累计。
            stats[key].update(np.asarray(batch[key]))

    # 统计结束后，取每个 RunningStats 的汇总（如 mean/std）
    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    if data_config.repo_id is None:
        raise RuntimeError("Impossible")

    # 生成输出目录：<assets_root>/<repo_id>（repo_id 唯一标识数据源）
    output_path = config.assets_dirs / data_config.repo_id

    # 打印即将写入的路径
    print(f"Writing stats to: {output_path}")
    # 将统计结果持久化到磁盘（后续训练/推理会加载使用）。
    normalize.save(output_path, norm_stats)


# export HF_LEROBOT_HOME=~/workspace/models/lerobot/
if __name__ == "__main__":
    tyro.cli(main)
