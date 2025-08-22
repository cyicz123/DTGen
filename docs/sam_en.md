# Documentation: `sam.py`

This script is an interactive tool for segmenting tableware from images using the Segment Anything Model (SAM).

## Functionality

The core function of this tool is to allow users to precisely segment tableware regions by drawing a bounding box on an image. The segmented tableware is then saved as a PNG image with a transparent background.

Key features:

*   **Interactive Segmentation**: A GUI window allows users to easily select the target area with a mouse.
*   **SAM Model Powered**: Utilizes the powerful SAM model for high-precision image segmentation.
*   **Two Operating Modes**:
    1.  **Command-Line Mode**: For quickly processing one or more specified image files.
    2.  **Interactive Batch Mode**: Designed specifically for processing datasets with a preset directory structure, facilitating batch operations.
*   **Smart Filtering**: A minimum area ratio threshold can be set to automatically filter out small segmentation results that may be noise.
*   **Mask Optimization**: Fills holes in the generated segmentation mask to ensure the integrity of the segmented region.

## Prerequisites

Before running this script, you need to download the SAM pre-trained model weights.

*   Download the model weights file `sam_vit_h_4b8939.pth`.
*   Create a `models` folder in the project root directory.
*   Place the downloaded model file into the `models` folder.

The final path should be: `models/sam_vit_h_4b8939.pth`.

## Usage

### Mode 1: Command-Line Mode

This mode is used for processing single or multiple images from any location.

**Basic Usage:**

```bash
# Process a single image, results will be saved in the default `output/` directory
python scripts/sam.py /path/to/your/image.jpg

# Process multiple images
python scripts/sam.py image1.jpg image2.png image3.jpeg
```

**Specify Output Directory:**

Use the `-o` or `--output` argument.

```bash
python scripts/sam.py image.jpg -o my_segmented_images/
```

**Set Area Filter Threshold:**

Use the `-m` or `--min-area-ratio` argument. This value represents the minimum ratio of the segmented area to the total image area. Segmentation results smaller than this ratio will be discarded.

```bash
# Only keep segmentation results that cover more than 5% of the image area
python scripts/sam.py image.jpg -m 0.05
```

### Mode 2: Interactive Batch Mode

This mode is specifically designed for processing datasets with a particular directory structure. It will prompt you to choose between processing the `datasets/train/cleaned` or `datasets/train/dirty` directories, and will save the results to `datasets/plate/cleaned` and `datasets/plate/dirty` respectively.

**Start Interactive Mode:**

```bash
python scripts/sam.py -i
# or
python scripts/sam.py --interactive
```

After starting, the program will prompt you to select which directory to process (clean tableware, dirty tableware, or both).

## GUI Operation Guide

In either mode, a window will pop up for each image being processed:

*   **Select Area**: Click and drag the left mouse button to draw a rectangle around the tableware.
*   **Confirm Selection**: After drawing, release the mouse button, then press the `Spacebar` to confirm the selection and start segmentation.
*   **Reset Selection**: If you are not satisfied with the selection, press the `r` key to clear it and draw again.
*   **Skip Image**: Press the `q` key to skip the current image without processing it.

## Command-Line Arguments

*   `images`: **[Command-Line Mode]** File paths of one or more input images (positional argument).
*   `-o`, `--output`: **[Command-Line Mode]** The path for the output directory. Defaults to `output`.
*   `-m`, `--min-area-ratio`: The minimum area ratio threshold for the segmentation mask. Defaults to `0.01`.
*   `-i`, `--interactive`: Enables interactive batch mode for processing preset dataset directories.
