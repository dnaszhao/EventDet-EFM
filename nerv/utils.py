import glob
import os
import pickle
from dataclasses import dataclass
from typing import Iterable, List

import cv2


def glob_all(path: str, only_dir: bool = False) -> List[str]:
    entries = glob.glob(os.path.join(path, "*"))
    if only_dir:
        entries = [p for p in entries if os.path.isdir(p)]
    return sorted(entries)


def sort_file_by_time(paths: Iterable[str]) -> List[str]:
    return sorted(paths, key=lambda p: os.path.getmtime(p))


def load_obj(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_obj(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


@dataclass
class AverageMeter:
    val: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    count: int = 0

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


def save_video(frames, path: str, fps: int = 30):
    if len(frames) == 0:
        raise ValueError("frames must not be empty")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


class VideoReader:
    def __init__(self, path: str, to_rgb: bool = True):
        self.path = path
        self.to_rgb = to_rgb

    def read_video(self):
        cap = cv2.VideoCapture(self.path)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if self.to_rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame
        finally:
            cap.release()

