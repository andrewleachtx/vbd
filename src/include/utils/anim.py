import os
from moviepy import *

# quick script for paraview visualization
RESOURCE_DIR = "output/animation"

# look for frame.####
image_files = [os.path.join(RESOURCE_DIR, f) for f in sorted(os.listdir(RESOURCE_DIR)) if f.startswith('frame.') and f.endswith('.png')]
print(f"Found {image_files} images, moving to a clip")
clip = ImageSequenceClip(image_files, fps=60)
clip.write_videofile('output.mp4', fps=60)