from pathlib import Path

import numpy as np
from PIL import Image


def cvt_img_float_to_uint8(img: np.ndarray) -> np.ndarray:
    img = (img * 255).astype(np.uint8)
    return img


def get_current_robot_frame(env) -> np.ndarray:
    state = env.get_state()
    img = cvt_img_float_to_uint8(img=state["rgb"])
    return img


def get_current_robot_camera_pose(env):
    robot = env.robots[0]
    camera_pos = robot.eyes.get_position()
    camera_orn = robot.eyes.get_orientation()
    return camera_pos, camera_orn


def save_image_to_disk(img: np.ndarray, path: Path, mode: str = 'RGB'):
    assert path.parent.exists(), f"Invalid save destination: {path}"
    if mode == 'RGB':
        assert img.shape[-1] == 3, f"Expected 3 channels for RGB image, found: {img.shape[-1]}"
        assert path.suffix.lower() == ".jpg", f"Invalid RGB image format: {path.suffix}"
    elif mode == 'RGBA':
        assert img.shape[-1] == 4, f"Expected 4 channels for RGBA image, found: {img.shape[-1]}"
        assert path.suffix.lower() == ".png", f"Invalid RGBA image format: {path.suffix}"
    else:
        raise Exception(f"Unknown image mode provided: {mode}")
    Image.fromarray(img, mode).save(path)
