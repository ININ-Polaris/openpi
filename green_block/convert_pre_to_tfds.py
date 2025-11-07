# dubhe_rlds_episode/dubhe_rlds_episode.py
from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds

_DESCRIPTION = """
Rosbag → RLDS-style episodic dataset.
Each example is an **episode** containing a variable-length sequence `steps`.
Each step has observation (state + 3 RGB images), action, reward/discount, flags, and language_instruction.
"""

_CITATION = ""


class DubheRldsEpisodeConfig(tfds.core.BuilderConfig):
    def __init__(self, *, split_spec: Dict[str, List[str]] | None = None, **kwargs):
        super().__init__(version=tfds.core.Version("1.0.0"), **kwargs)
        self.split_spec = split_spec or {}


class DubheRldsEpisode(tfds.core.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [DubheRldsEpisodeConfig(name="default", description="Episodic RLDS with steps sequence")]

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Point manual_dir to the preprocessed root that contains:
      - index.json
      - ep_xxxxxx/frames/*.jpg
      - ep_xxxxxx/steps.jsonl
      - ep_xxxxxx/episode_meta.json (optional but recommended, with 'file_path')
    """

    def _info(self) -> tfds.core.DatasetInfo:
        # === steps.feature ===
        step_feature = tfds.features.FeaturesDict(
            {
                "action": tfds.features.Tensor(
                    shape=(8,), dtype=tf.float32, doc="Robot action for joints in one arms + grippers."
                ),
                "is_terminal": tfds.features.Scalar(
                    dtype=tf.bool, doc="True on last step of the episode if terminal; True for demos."
                ),
                "is_last": tfds.features.Scalar(dtype=tf.bool, doc="True on last step of the episode."),
                "language_instruction": tfds.features.Text(doc="Language Instruction."),
                "observation": tfds.features.FeaturesDict(
                    {
                        "state": tfds.features.Tensor(
                            shape=(8,), dtype=tf.float32, doc="Robot joint pos (one arms + grippers)."
                        ),
                        # 固定分辨率 JPEG
                        "mid": tfds.features.Image(
                            shape=(720, 1280, 3), encoding_format="jpeg", doc="RGB camera observation."
                        ),
                        "right": tfds.features.Image(
                            shape=(240, 424, 3), encoding_format="jpeg", doc="RGB camera observation."
                        ),
                    }
                ),
                "is_first": tfds.features.Scalar(dtype=tf.bool, doc="True on first step of the episode."),
                "discount": tfds.features.Scalar(dtype=tf.float32, doc="Discount if provided, default to 1."),
                "reward": tfds.features.Scalar(dtype=tf.float32, doc="Reward if provided, 1 on final step for demos."),
            }
        )
        steps_feature = tfds.features.Dataset(step_feature)

        features = tfds.features.FeaturesDict(
            {
                # 可变长序列（-1）
                "steps": steps_feature,
                "episode_metadata": tfds.features.FeaturesDict(
                    {
                        "file_path": tfds.features.Text(doc="Path to the original data file."),
                    }
                ),
            }
        )

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=features,
            citation=_CITATION,
            homepage="",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        root = Path(dl_manager.manual_dir)
        index_path = root / "index.json"
        with index_path.open("r", encoding="utf-8") as f:
            idx = json.load(f)
        all_eps = [e["episode_id"] for e in idx]

        spec = getattr(self.builder_config, "split_spec", {}) or {}
        if spec:
            return {name: self._generate_examples(root, eps) for name, eps in spec.items()}

        # 默认 90/10
        n = len(all_eps)
        k = max(1, int(n * 0.9))
        return {
            "train": self._generate_examples(root, all_eps[:k]),
            "validation": self._generate_examples(root, all_eps[k:]),
        }

    def _generate_examples(self, root: Path, episode_ids: List[str]) -> Iterator[Tuple[str, Dict[str, Any]]]:
        for ep_id in episode_ids:
            ep_dir = root / ep_id
            steps_path = ep_dir / "steps.jsonl"
            frames_dir = ep_dir / "frames"
            meta_path = ep_dir / "episode_meta.json"

            if not steps_path.exists():
                continue

            # episode_metadata
            file_path = str(meta_path)
            if meta_path.exists():
                try:
                    with meta_path.open("r", encoding="utf-8") as f:
                        meta = json.load(f)
                    file_path = str(meta.get("file_path", file_path))
                except Exception:
                    pass

            # 重要：steps 用迭代器（逐行读，内存友好）
            def steps_iter():
                with steps_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line)
                        obs = row["observation"]

                        # 将相对名变成绝对路径字符串，Image 特征会按路径读取 JPEG
                        yield {
                            "action": [float(x) for x in row["action"]][-8:],
                            "is_terminal": bool(row.get("is_terminal", False)),
                            "is_last": bool(row.get("is_last", False)),
                            "language_instruction": str(row.get("language_instruction", "")),
                            "observation": {
                                "state": [float(x) for x in obs["state"]][-8:],
                                "mid": str(frames_dir / obs["mid"]),
                                "right": str(frames_dir / obs["right"]),
                            },
                            "is_first": bool(row.get("is_first", False)),
                            "discount": float(row.get("discount", 1.0)),
                            "reward": float(row.get("reward", 0.0)),
                        }

            example = {
                "steps": steps_iter(),  # 交给 TFDS 按序消费（不会一次性载入）
                "episode_metadata": {"file_path": file_path},
            }
            yield ep_id, example


def main(prep_dir: Path, data_dir: Path, name: str):
    index_path = Path(prep_dir) / "index.json"
    with index_path.open("r", encoding="utf-8") as f:
        idx = json.load(f)
    all_eps = [e["episode_id"] for e in idx]  # 不需要知道数量，直接全取

    cfg = DubheRldsEpisodeConfig(name=name, description="Built from Python", split_spec={"train": all_eps})
    builder = DubheRldsEpisode(data_dir=data_dir, config=cfg)

    # 使用 DownloadConfig 指定 manual_dir=预处理输出目录
    dl_cfg = tfds.download.DownloadConfig(manual_dir=prep_dir)

    # 真正构建（逐步读写，内存友好）
    builder.download_and_prepare(download_config=dl_cfg)


# uv run green_block/convert_pre_to_tfds.py --prep_dir runtime/pre_green --data_dir runtime/output --name pick_block
if __name__ == "__main__":
    import tyro

    tyro.cli(main)
