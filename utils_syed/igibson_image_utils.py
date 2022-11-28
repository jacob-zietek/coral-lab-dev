from pathlib import Path
from typing import List

import numpy as np

from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from image_utils import cvt_img_float_to_uint8
from json_utils import NeRFFrameMetadata


def get_rgb_image_for_camera_params(renderer: MeshRenderer, camera_position: List[float], camera_target: List[float],
                                    camera_up: List[float], hidden_instances: List = []) -> np.ndarray:
    renderer.set_camera(camera_position, camera_target, camera_up, cache=False)
    frames = renderer.render(modes=("rgb",), hidden=hidden_instances)
    assert len(frames) == 1
    img = frames[0]
    assert len(img.shape) == 3 and img.shape[-1] == 4
    print(img.min(), img.max(), img.shape, img.dtype)
    rgb = img[:, :, :3]
    rgb = cvt_img_float_to_uint8(rgb)
    return rgb


def get_rgb_images_w_metadata_for_list_of_camera_params(renderer: MeshRenderer, camera_params_list: List,
                                                        hidden_instances: List = []) -> List:
    """
    camera_params_list should be a list of lists/tuples, each of the format: [camera_position, camera_target, camera_up]
    """
    frames_with_metadata = []
    for camera_params in camera_params_list:
        camera_position, camera_target, camera_up = camera_params
        rgb_img_arr = get_rgb_image_for_camera_params(renderer=renderer, camera_position=camera_position,
                                                      camera_target=camera_target, camera_up=camera_up,
                                                      hidden_instances=hidden_instances)
        c2w_transform = np.linalg.inv(renderer.V)
        frame_metadata = NeRFFrameMetadata.from_params(file_path=Path(""), transform_matrix=c2w_transform)
        frames_with_metadata.append([rgb_img_arr, frame_metadata])
    return frames_with_metadata
