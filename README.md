# Flank Face Photometric Stereo Reconstruction

This repository contains Python scripts for reconstructing the flank-face
surface of a tool from a sequence of images. The workflow unwraps images of a
conical surface, aligns and stitches image sectors, calibrates the brightness
response, and estimates a height map from the resulting photometric stereo
measurements.

## Workflow

The scripts are standalone and currently use file paths and acquisition
parameters defined in the source code. The intended processing order is:

1. **Unwrap the cone images** with `image_warping.py`. It maps each captured
   image to an unwrapped coordinate system and writes the warped frames.
2. **Align, filter, and stitch the sectors** with `side_alignment.py`. It
   applies a Fourier-domain stripe filter and combines image strips into
   sector images.
3. **Calibrate the brightness response** with `post_calibrate.py`. It fits a
   polynomial brightness model to image profiles.
4. **Reconstruct the surface** with `photo_stereo_ls.py`. It estimates the
   surface gradient from the sector images and integrates that gradient into a
   height map.

Check and update the input/output paths and calibration parameters in each
script before running it. The scripts expect image data under a `tool1`
directory, with subdirectories such as `raw5`, `warp1`, `warp5`, and
`stitch1`; these paths and naming conventions are currently hard-coded.

## Data instructions

Detailed instructions for preparing, organizing, and processing the required
data are currently being organized and will be added to this README.