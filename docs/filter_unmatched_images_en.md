# Documentation: `filter_unmatched_images.py`

This script is used to automatically filter an image dataset. It generates prompts from filenames, calculates image-text CLIP similarity, and uses an adaptive threshold to move images that do not match their descriptions.

## Functionality

After generating a large number of images, this script aims to automate the filtering process, removing images of poor quality or those that do not match the expected content.

Its workflow is divided into two phases:

1.  **Phase 1: Calculate Similarity**
    *   The script recursively finds all image files in the input path.
    *   It extracts a "dirtiness level" (e.g., `clean`, `slightly_dirty`, `moderately_dirty`, `heavily_dirty`) from the filename of each image.
    *   Based on the extracted level, it automatically generates a simple text prompt (e.g., `"a photo of a clean of tableware"`).
    *   It uses a CLIP model to calculate the similarity score between the image and the generated prompt, storing the scores categorized by the dirtiness level.

2.  **Phase 2: Adaptive Filtering and Moving**
    *   For each dirtiness level category, the script calculates the **mean** and **standard deviation (std dev)** of the similarity scores for all images in that category.
    *   Based on these statistics, it dynamically calculates an **adaptive threshold** using the formula: `threshold = mean - k * std_dev`, where `k` is an adjustable factor.
    *   Finally, the script iterates through the images of that category again, moving any image with a similarity score below this dynamic threshold to a specified output directory.

This adaptive approach is more intelligent than a fixed threshold because it can identify outliers based on the data distribution within each category.

## Usage

Run this script from the command line, providing the input path containing the images.

```bash
python scripts/filter_unmatched_images.py -i <input_path> -o <output_directory> -k <std_dev_factor>
```

### Examples

*   **Process an entire dataset directory**:
    Assuming your image dataset is in `my_dataset/`, you want to move mismatched images to `unmatched_images/`, using a relatively lenient factor of `k=1.2`.

    ```bash
    python scripts/filter_unmatched_images.py -i my_dataset/ -o unmatched_images/ -k 1.2
    ```

*   **Process a single image file**:
    Although designed for directories, it can also process a single file.

    ```bash
    python scripts/filter_unmatched_images.py -i my_dataset/clean/clean_001.png
    ```

*   **Use stricter or looser filtering**:
    Increasing the value of `k` will lower the threshold, thus retaining more images (looser filtering). Decreasing `k` will raise the threshold, moving more images (stricter filtering).

    ```bash
    # Use a standard deviation factor of k=2.0 for looser filtering
    python scripts/filter_unmatched_images.py -i my_dataset/ -k 2.0
    ```

## Command-Line Arguments

*   `-i`, `--input`: **Required**. The input path, which can be an image file or a directory containing image files. The script will search this directory recursively.
*   `-o`, `--output`: The output directory for storing low-similarity images. Defaults to `./unmatched`.
*   `-k`, `--std-dev-factor`: The standard deviation factor used to calculate the adaptive threshold. `threshold = mean - k * std`. A larger k value means a lower (looser) threshold. Defaults to `1.5`.
*   `--model-name`: The name of the CLIP model to use from the Hugging Face Hub. Defaults to `openai/clip-vit-base-patch32`.
