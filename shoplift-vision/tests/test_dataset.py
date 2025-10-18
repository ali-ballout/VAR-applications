import cv2
import numpy as np
import pytest

from shoplift_vision.data.dataset import DatasetConfig, VideoClassificationDataset
from shoplift_vision.utils.io import load_video_frames


@pytest.fixture()
def synthetic_dataset(tmp_path):
    data_root = tmp_path / "data"
    for split in ["train", "val"]:
        for label in ["shoplifting", "no_shoplifting"]:
            class_dir = data_root / split / label
            class_dir.mkdir(parents=True, exist_ok=True)
            video_path = class_dir / "sample.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                5.0,
                (64, 64),
            )
            for _ in range(5):
                frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                writer.write(frame)
            writer.release()
    return data_root


def test_dataset_loading(synthetic_dataset):
    config = DatasetConfig(
        root=synthetic_dataset,
        split="train",
        labels=["shoplifting", "no_shoplifting"],
        num_frames=4,
    )
    dataset = VideoClassificationDataset(config)
    sample = dataset[0]
    assert sample["pixel_values"].shape[0] == 4
    assert sample["labels"] in {0, 1}


def test_load_video_frames(synthetic_dataset):
    video_path = next((synthetic_dataset / "train" / "shoplifting").glob("*.mp4"))
    video = load_video_frames(str(video_path), num_frames=4)
    assert video.frames.shape[0] == 4
