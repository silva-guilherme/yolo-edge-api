"""
preprocessing/utils/letterbox.py
Implementacao manual do letterbox.
"""
from typing import Tuple
import cv2
import numpy as np


def letterbox(
    frame: np.ndarray,
    target_size: int = 640,
    pad_color: int = 114,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    h, w = frame.shape[:2]

    scale = min(target_size / h, target_size / w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2

    frame_lb = np.full((target_size, target_size, 3), pad_color, dtype=np.uint8)
    frame_lb[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    return frame_lb, scale, (pad_w, pad_h)


def adjust_bboxes(
    boxes_xyxy: np.ndarray,
    scale: float,
    pad_w: int,
    pad_h: int,
) -> np.ndarray:
    boxes = boxes_xyxy.copy().astype(float)
    boxes[:, [0, 2]] -= pad_w
    boxes[:, [1, 3]] -= pad_h
    boxes /= scale
    return boxes
