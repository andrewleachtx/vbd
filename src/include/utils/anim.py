import os
import sys
from moviepy import *

"""
    Takes in a directory full of images, ideally named frame_0000, frame_0001, ..., frame_abcd
    and converts to a 
"""

### SCRIPT/ARGS HANDLING ###
if len(sys.argv) < 2:
    print("Usage: python anim.py <image dirname> [resource dir=docs/demo/] [framerate=60fps] [output_dir=resource_dir]")

    sys.exit(1)

IMAGE_DIR = sys.argv[1]
if IMAGE_DIR[-1] == "/":
    IMAGE_DIR = IMAGE_DIR[:-1]

RESOURCE_DIR = "./docs/demo"
if len(sys.argv) == 3:
    RESOURCE_DIR = sys.argv[2]

if RESOURCE_DIR[-1] == "/":
    RESOURCE_DIR = RESOURCE_DIR[:-1]

FRAMERATE = 60
if len(sys.argv) == 4:
    FRAMERATE = int(sys.argv[3])

FULL_DIR = f"{RESOURCE_DIR}/{IMAGE_DIR}"

### IMAGE LOADING & CLIP GENERATION ###
images = []
for filename in os.listdir(FULL_DIR):
    if filename[-4:] == ".png":
        images.append(f"{FULL_DIR}/{filename}")
print(f"Found {len(images)} images, creating clip with {FRAMERATE} fps")
clip = ImageSequenceClip(images, fps=FRAMERATE)

# video name 
filename = f"{FULL_DIR}/a.mp4"

# if there's already an output.mp4 don't overwrite
if os.path.exists(filename):
    print(f"{filename} already exists, please delete it or rename it")
    sys.exit(1)

clip.write_videofile(filename, fps=FRAMERATE)