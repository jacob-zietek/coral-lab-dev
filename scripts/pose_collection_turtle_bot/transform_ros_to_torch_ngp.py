"""
This script transforms the ODOM and Image data collected
by ROS in the pose_collection2.py script to instant-npg/
torch-ngp compatible scene with the images and 
transform.json to train a NeRF.

Input folder should be in the format of 

data/
    images/
        image1
        image2
        image3
        ...
    odom/
        odom1
        odom2
        odom3
        ...

Each image has its cooresponding odom data.

python3 transform_ros_to_torch_ngp.py --input data/11_28_2022_16_06_00 --numPictures 1000 --output data/test --run_laplacian True
"""

import cv2
import argparse
from itertools import islice
import glob
from scipy.spatial.transform import Rotation as R
import os
import pathlib
import shutil
import yaml
import json


def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := list(islice(it, n))):
        yield batch


def laplacian_var(img):
    # calculates the laplacian variance to find
    # the blur of an image

    # convert to greyscale
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # return laplacian variance
    return cv2.Laplacian(grey, cv2.CV_64F).var()


def extract_least_blurriest_frames_laplacian(picture_paths: list[str], target_num_pictures: int) -> list[str]:
    """
    Given a list of pictures, this function will return about the target_num_pictures least
    blurriest ones. 
    """
    return_list = []

    total_num_pictures = len(picture_paths)

    partitionlen = total_num_pictures // target_num_pictures

    for partition in batched(picture_paths, partitionlen):
        least_blurriest = sorted(partition, key=lambda x: laplacian_var(cv2.imread(x)), reverse=True)[0]
        return_list.append(least_blurriest)

    return return_list


def create_scene(output_path: str, picture_paths: list[str], odom_paths: list[str]):
    """
    Creates a instant-ngp/torch-ngp compatible scene.
    """

    transforms = {}


    output_images_path = output_path + "/images"

    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    for picture in picture_paths: shutil.copy(picture, output_images_path)

    frames = []

    for image, odom in zip(picture_paths, odom_paths):

        frame = {}

        frame["file_path"] = "images/" + pathlib.PurePath(image).name

        img = cv2.imread(image)
        sharpness = laplacian_var(img)
        frame["sharpness"] = sharpness

        odom_data = None

        with open(odom, 'r') as file:
            odom_data = yaml.safe_load(file)

        rot_quaternion = odom_data["orientation"]
        position_vector = odom_data["position"]

        rot_matrix = R.from_quat([rot_quaternion['x'], rot_quaternion['y'], rot_quaternion['z'],\
                                  rot_quaternion['w']]).as_matrix()

        transformation_matrix = []

        # Create a transformation matrix from the rot matrix and position
        # vector. Rot mtx is 3x3, pos vec is 3x1
        # r r r p
        # r r r p
        # r r r p
        # 0 0 0 1
        
        # Append each row from the rotation matrix
        transformation_matrix.append(list(rot_matrix[0]) + [position_vector['x']])
        transformation_matrix.append(list(rot_matrix[1]) + [position_vector['y']])
        transformation_matrix.append(list(rot_matrix[2]) + [position_vector['z']])
        transformation_matrix.append([0, 0, 0, 1])

        frame["transform_matrix"] = transformation_matrix

        frames.append(frame)

    transforms["frames"] = frames

    print(transforms)

    with open(output_path + "/transforms.json", "w") as outfile:
        json.dump(transforms, outfile)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', dest='input', type=str, help='Input folder to read from. This should be the root of the directory, with root/images and root/odom.')
    parser.add_argument('--output', dest='output', type=str, help='Output folder.')

    parser.add_argument("--run_laplacian", dest='laplacian', type=bool, help='Whether or not to extract the least blurriest frames using laplacian variance. Final transform.json will only contain the least blurriest frames.')
    parser.add_argument('--numPictures', dest='numPictures', type=int, help='Rough number of pictures to target. Used if run_laplacian is True.')

    args = parser.parse_args()

    picture_paths = sorted(glob.glob(args.input+"/images/*"))
    odom_paths = []

    if args.laplacian:
        picture_paths = extract_least_blurriest_frames_laplacian(picture_paths, args.numPictures)
    
    # Replace .png with .yml and folder to odom
    odom_paths = [x.replace("images", "odom")[:-4] + ".yml" for \
                  x in picture_paths] 

    create_scene(args.output, picture_paths, odom_paths)
