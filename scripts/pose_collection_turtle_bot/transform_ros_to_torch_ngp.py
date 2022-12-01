"""
This script transforms the ODOM and Image data collected
by ROS in the pose_collection2.py script to instant-npg/
torch-ngp compatible transform.json format to train
a NeRF.

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
"""

import cv2
import argparse
from itertools import islice
import glob


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

    #print(picture_paths, target_num_pictures)

    return_list = []

    total_num_pictures = len(picture_paths)

    partitionlen = total_num_pictures // target_num_pictures

    for partition in batched(picture_paths, partitionlen):

        least_blurriest = sorted(partition, key=lambda x: laplacian_var(cv2.imread(x)), reverse=True)[0]

        return_list.append(least_blurriest)

    return least_blurriest


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', dest='input', type=str, help='Input folder to read from. This should be the root of the directory, with root/images and root/odom.')
    parser.add_argument('--output', dest='output', type=str, help='Output folder.')

    parser.add_argument("--run_laplacian", dest='laplacian', type=bool, help='Whether or not to extract the least blurriest frames using laplacian variance. Final transform.json will only contain the least blurriest frames.')
    parser.add_argument('--numPictures', dest='numPictures', type=int, help='Rough number of pictures to target. Used if run_laplacian is True.')

    args = parser.parse_args()

    picture_paths = sorted(glob.glob(args.input+"/images/*"))

    extract_least_blurriest_frames_laplacian(picture_paths, args.numPictures)



