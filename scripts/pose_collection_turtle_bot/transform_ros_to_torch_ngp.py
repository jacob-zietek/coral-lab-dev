"""
This script transforms the ODOM and Image data collected
by ROS in the pose_collection.py script to instant-npg/
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

python3 transform_ros_to_torch_ngp.py --input data/11_28_2022_16_06_00 --numPictures 1000 --output data/test --run_laplacian True --camera_translation 0.14
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
import numpy as np


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
        least_blurriest = sorted(partition, key=lambda x: laplacian_var(
            cv2.imread(x)), reverse=True)[0]
        return_list.append(least_blurriest)

    return return_list


def create_scene(output_path: str, picture_paths: list[str], odom_paths: list[str], camera_translation: float):
    """
    Creates a instant-ngp/torch-ngp compatible scene.
    """

    AABB_SCALE = 16
    angle_x = 0.925
    angle_y = 0.7155
    w = 640
    h = 480
    k1 = 0.16399
    k2 = -0.27184
    p1 = 0.00105
    p2 = -0.00166
    cx = w / 2
    cy = h / 2
    fl_x = 2
    fl_y = 2.67

    transforms = {
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        # "fl_x": fl_x,
        # "fl_y": fl_y,
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        # "aabb_scale": AABB_SCALE
    }

    output_images_path = output_path + "/images"

    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    for picture in picture_paths:
        shutil.copy(picture, output_images_path)

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

        # Both odom and instant_ngp are +Z up
        odom_euler = R.from_quat([rot_quaternion['x'], rot_quaternion['y'], rot_quaternion['z'],
                                  rot_quaternion['w']]).as_euler('xyz', degrees=True)

        # Default camera position in instant_ngp is pointing toward -Z facing +X
        # Rotate camera 90 deg on the X axis to make the camera point toward +X,
        # then adjust the Z rotation given by the odom data -90 deg
        odom_euler[1] = 0
        odom_euler[0] = 90
        odom_euler[2] -= 90

        rot_matrix = R.from_euler('xyz', odom_euler, degrees=True).as_matrix()

        if camera_translation:

            theta = odom_euler[2] + 90
            mag = camera_translation

            dy = mag * np.sin(np.deg2rad(theta))
            dx = mag * np.cos(np.deg2rad(theta))

            position_vector['x'] += dx
            position_vector['y'] += dy

        transformation_matrix = []

        # Append each row from the rotation matrix
        transformation_matrix.append(
            list(rot_matrix[0]) + [position_vector['x']])
        transformation_matrix.append(
            list(rot_matrix[1]) + [position_vector['y']])
        transformation_matrix.append(
            list(rot_matrix[2]) + [position_vector['z']])
        transformation_matrix.append([0, 0, 0, 1])

        frame["transform_matrix"] = transformation_matrix

        frames.append(frame)

    transforms["frames"] = frames

    # print(transforms)

    with open(output_path + "/transforms.json", "w") as outfile:
        json.dump(transforms, outfile, sort_keys=True, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', dest='input', type=str,
                        help='Input folder to read from. This should be the root of the directory, with root/images and root/odom.')
    parser.add_argument('--output', dest='output',
                        type=str, help='Output folder.')

    parser.add_argument("--run_laplacian", dest='laplacian', type=bool,
                        help='Whether or not to extract the least blurriest frames using laplacian variance. Final transform.json will only contain the least blurriest frames.')
    parser.add_argument('--numPictures', dest='numPictures', type=int,
                        help='Rough number of pictures to target. Used if run_laplacian is True.')
    parser.add_argument('--camera_translation', dest="camera_translation", type=float,
                        help="The distance of the camera from the center of the turtlebot. This assumes the camera is facing away from the center.")

    args = parser.parse_args()

    picture_paths = sorted(glob.glob(args.input+"/images/*"))
    odom_paths = []

    if args.laplacian:
        print("Extracing least blurriest frames...")
        picture_paths = extract_least_blurriest_frames_laplacian(
            picture_paths, args.numPictures)
        print("Done!")

    # Replace .png or .jpg with .yml and folder to odom
    odom_paths = [x.replace("images", "odom")[:-4] + ".yml" for
                  x in picture_paths]

    print("Creating instant ngp scene...")
    create_scene(args.output, picture_paths,
                 odom_paths, args.camera_translation)
    print("Done!")
