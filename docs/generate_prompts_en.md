# Documentation: `generate_prompts.py`

This script is used to generate a large number of prompts for text-to-image models based on YAML configuration files, specifically for creating tableware image datasets.

## Functionality

The script automatically generates a large and diverse set of prompts by combining descriptions from different dimensions. Key features include:

1.  **Configuration-Based**: Loads description fragments from multiple YAML files, including:
    *   Tableware descriptions (`tableware_description.yaml`)
    *   Dirtiness descriptions (`dirtiness_description_origin.yaml`)
    *   Background descriptions (`background.yaml`)
2.  **Combinatorial Generation**: Randomly combines these description fragments to construct final prompts based on user-specified **tableware type** (`plate` or `bowl`), **dirtiness level** (`clean`, `slightly_dirty`, `moderately_dirty`, `heavily_dirty`), and **quantity**.
3.  **Structured Output**: Saves each generated prompt as a separate `.txt` file, organized in a directory structure of `output/<tableware_type>/<dirtiness_level>/`.
4.  **Reproducibility**: Supports setting a random seed to ensure that the same dataset is generated on each run.

This script greatly simplifies the process of creating large-scale, structured image-text datasets.

## Configuration File Format

The script relies on YAML configuration files to provide the "raw material" for prompt generation.

*   **`tableware_description.yaml`**: Defines descriptions for different types of tableware.
    ```yaml
    plate:
      - "a white ceramic plate"
      - "a blue patterned porcelain plate # Comments in Chinese will be ignored"
    bowl:
      - "a deep stoneware bowl"
    ```
*   **`dirtiness_description_origin.yaml`**: Defines specific dirtiness descriptions for different tableware and dirtiness levels.
    ```yaml
    plate:
      slightly_dirty:
        - "with a few crumbs"
      moderately_dirty:
        - "with some leftover sauce stains"
    ```
*   **`background.yaml`**: Defines the background descriptions for the image.
    ```yaml
    backgrounds:
      - "on a wooden table"
      - "on a marble countertop"
    ```

## Usage

Run the script from the command line and provide the necessary arguments to control the generation process.

```bash
python scripts/generate_prompts.py --tableware-type <type> --dirtiness-level <level> -n <num_prompts> [options]
```

### Examples

*   **Generate 1000 "slightly_dirty" prompts for plates**:

    ```bash
    python scripts/generate_prompts.py --tableware-type plate --dirtiness-level slightly_dirty -n 1000
    ```
    This will generate 1000 `.txt` files in the `output/plate/slightly_dirty/` directory.

*   **Generate 500 "clean" prompts for bowls, specifying output directory and config files**:

    ```bash
    python scripts/generate_prompts.py \
        --tableware-type bowl \
        --dirtiness-level clean \
        -n 500 \
        -t my_prompts/tableware.yaml \
        -b my_prompts/backgrounds.yaml \
        -o my_dataset/
    ```

## Command-Line Arguments

*   `--tableware-type`: **Required**. Specifies the tableware type. Choices: `plate`, `bowl`.
*   `--dirtiness-level`: **Required**. Specifies the dirtiness level. Choices: `clean`, `slightly_dirty`, `moderately_dirty`, `heavily_dirty`.
*   `-n`, `--num-prompts`: **Required**. The number of prompts to generate.
*   `-t`, `--tableware`: Path to the tableware description YAML file. Defaults to `prompts/tableware_description.yaml`.
*   `-d`, `--dirtiness`: Path to the dirtiness description YAML file. Defaults to `prompts/dirtiness_description_origin.yaml`.
*   `-b`, `--background`: Path to the background description YAML file. Defaults to `prompts/background.yaml`.
*   `-o`, `--output`: The root output directory. Defaults to `output`.
*   `--digits`: The number of digits for the output file's serial number. Defaults to `5` (e.g., `clean_00001.txt`).
*   `--seed`: Random seed for reproducible generation. Defaults to `42`.
