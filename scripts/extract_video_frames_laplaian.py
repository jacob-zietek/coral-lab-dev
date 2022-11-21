"""
This script extracts keyframes out of a video for COLMAP processing
and NeRF generation. It chooses the least blurred image out of an interval of images
via its laplacian variance.
"""

import cv2
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--input', dest='input', type=str, help='Input file to read from')
parser.add_argument('--output', dest='output', type=str, help='Output folder')
parser.add_argument('--numPictures', dest='numPictures', type=int, help='Rough number of pictures to target')

args = parser.parse_args()

def laplacian_var(img):
    # calculates the laplacian variance to find
    # the blur of an image

    # convert to greyscale
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # return laplacian variance
    return cv2.Laplacian(grey, cv2.CV_64F).var()

input_f = args.input
output_f = args.output
numPictures = args.numPictures

videocap = cv2.VideoCapture(input_f)

videolen = int(videocap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

# Partition video into numPictures pieces, record the least blurred
# photo of each partition
partitionlen = videolen // numPictures

totalframes = videolen/partitionlen

print(f'Beginning to extract the best {totalframes} frames...')

success, image = videocap.read()
image_buffer = [image]
inner = 1

imgcount = 0
while success:

    # Extract partitionlen # of frames, add to buffer
    while success and inner < partitionlen:
        image_buffer.append(image)
        success, image = videocap.read()
        inner += 1

    # Write least blurred frame
    image_buffer.sort(key=laplacian_var)
    cv2.imwrite(f'{output_f}/{imgcount:08}.png', image_buffer[0])
    imgcount += 1
    
    inner = 0
    image_buffer = []

print(f'Success! Extracted {totalframes} frames into {output_f}.')
