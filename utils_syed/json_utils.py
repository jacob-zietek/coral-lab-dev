import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from pydantic import BaseModel


class JsonBaseModel(BaseModel):
    def to_json(self) -> Dict:
        return self.dict()

    def to_json_file(self, path: Path):
        assert path.parent.exists(), f"Invalid save directory: {path.parent}"
        assert path.suffix == ".json", f"Invalid json file suffix: {path.suffix}"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=4)

    @classmethod
    def from_json_file(cls, path: Path) -> 'BaseModel':
        with open(path, 'r', encoding='utf-8') as f:
            json_dict = json.load(f)
        return cls(**json_dict)


class NeRFFrameMetadata(JsonBaseModel):
    file_path: str
    transform_matrix: List[List[float]]
    rotation: Optional[float]

    @classmethod
    def from_params(cls, file_path: Path, transform_matrix: np.ndarray, rotation: Optional[float] = None) \
            -> 'NeRFFrameMetadata':
        assert transform_matrix.shape == (4, 4), f"transform matrix has incorrect shape: {transform_matrix.shape}"
        return cls(file_path=str(file_path), transform_matrix=transform_matrix.tolist(), rotation=rotation)


class NeRFFrameCollectionMetadata(JsonBaseModel):
    camera_angle_x: float
    frames: List[NeRFFrameMetadata]
    offset: Optional[List[float]] = None

    @classmethod
    def from_coll(cls, coll: List['NeRFFrameCollectionMetadata']) -> 'NeRFFrameCollectionMetadata':
        camera_angle_x = coll[0].camera_angle_x
        frames_list = coll[0].frames
        for frame_coll_metadata in coll[1:]:
            assert frame_coll_metadata.camera_angle_x == camera_angle_x
            frames_list += frame_coll_metadata.frames
        return cls(camera_angle_x=camera_angle_x, frames=frames_list)

    def compute_overall_offset_in_camera_poses(self):
        # translations_mean = np.zeros(3).flatten()
        bounds = np.array([[np.inf, -np.inf],
                           [np.inf, -np.inf],
                           [np.inf, -np.inf]], dtype=np.float32)
        assert bounds.shape == (3, 2)
        for frame_metadata in self.frames:
            frame_transform = np.array(frame_metadata.transform_matrix)
            assert frame_transform.shape == (4, 4) and frame_transform[-1, :].flatten().tolist() == [0, 0, 0, 1]
            for xyz_idx in range(3):
                bounds[xyz_idx, 0] = np.min((bounds[xyz_idx, 0], frame_transform[xyz_idx, -1]))
                bounds[xyz_idx, 1] = np.max((bounds[xyz_idx, 1], frame_transform[xyz_idx, -1]))
                # instead of min/max values, we can use 5th/95th-percentile values
            # translations_mean += frame_transform[:3, -1].flatten() / len(self.frames)
        # self.offset = translations_mean.tolist()
        self.offset = bounds.mean(axis=1).flatten().tolist()[::-1]
    
        print(bounds - np.array(self.offset[::-1]).reshape(3,1))
        print(bounds) 
        print(self.offset)
        print(self.offset[::-1])
        # translations_mean = np.array([0, -1.5, 1.1]).flatten()  # predefined offset
        # for idx, frame_metadata in enumerate(self.frames):
        #     frame_transform = np.array(frame_metadata.transform_matrix)
        #     frame_transform[:3, -1] -= translations_mean[:]
        #     self.frames[idx].transform_matrix = frame_transform.tolist()

    # @classmethod
    # def create_split(cls, split_point: float, shuffle: bool = True) -> Tuple['NeRFFrameCollectionMetadata',
    #                                                                          'NeRFFrameCollectionMetadata']:
    #     assert 0 < split_point < 1, f"Expected split_point to be a float between 0 and 1, got: {split_point}"
    #     assert cls.camera_angle_x > 0
    #     assert cls.frames != []
    #     if shuffle:
    #         random.shuffle(cls.frames)
    #     abs_split_point = np.ceil(split_point * len(cls.frames))
    #     split1 = cls(camera_angle_x=cls.camera_angle_x, frames=cls.frames[:abs_split_point])
    #     split2 = cls(camera_angle_x=cls.camera_angle_x, frames=cls.frames[abs_split_point:])
    #     return split1, split2


if __name__ == "__main__":
    # Create a split from a NeRFFrameCollectionMetadata

    #data_path = sys.argv[1]  # data_path should contain a "transforms_metadata.json" file in it.
    data_path = "/home/jzietek/Documents/coral-lab-dev/scripts/pose_collection_turtle_bot/data/test2"
    base_dir = Path(data_path)
    metadata = NeRFFrameCollectionMetadata.from_json_file(base_dir / "transforms.json")
    metadata.compute_overall_offset_in_camera_poses()
    exit()
    frames_list = metadata.frames
    split_point = 0.8  # 80 train / 20 test split

    random.shuffle(frames_list)
    abs_split_point = int(np.ceil(split_point * len(frames_list)))
    split1 = NeRFFrameCollectionMetadata(camera_angle_x=metadata.camera_angle_x, frames=frames_list[:abs_split_point],
                                         offset=metadata.offset)
    split2 = NeRFFrameCollectionMetadata(camera_angle_x=metadata.camera_angle_x, frames=frames_list[abs_split_point:],
                                         offset=metadata.offset)

    # split1, split2 = metadata.create_split(split_point=0.8)
    split1.to_json_file(base_dir / "transforms_train.json")
    split2.to_json_file(base_dir / "transforms_test.json")
