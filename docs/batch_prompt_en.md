# Documentation: `batch_prompt.py`

This script is used to batch-generate images based on prompts from text files.

## Functionality

The script connects to the backend service of ComfyUI and uses a given workflow to generate images. It traverses a specified path (which can be a single file or a directory), reads prompts from `.txt` files, and then generates one or more images for each prompt.

Key features:

*   **Service Connection**: Connects to the service running at `127.0.0.1:8188` via WebSocket.
*   **Prompt Reading**: Reads prompts from specified `.txt` files or a directory. The script looks for files ending in `_ddddd.txt` (where d is a digit).
*   **Workflow Template**: Uses a JSON file as a workflow template (defaults to `workflows/flux_api.json`).
*   **Prompt Injection**: Injects the prompt read from the file into a specified node in the workflow.
*   **Generate and Save Images**: Calls the service to generate images and saves them in the same directory as the corresponding `.txt` file.
*   **Skip Existing Files**: Before generating an image, it checks if an output file with the same name already exists and skips it if it does.

## Usage

You can run this script via the command line.

```bash
python scripts/batch_prompt.py <path_to_prompts> [options]
```

### Examples

*   Process a single file:
    ```bash
    python scripts/batch_prompt.py /path/to/your/prompt_00001.txt
    ```

*   Process all matching files in a directory:
    ```bash
    python scripts/batch_prompt.py /path/to/your/prompts_directory/
    ```

*   Use a custom workflow and node ID:
    ```bash
    python scripts/batch_prompt.py /path/to/prompts/ --workflow /path/to/custom_workflow.json --positive-id 10
    ```

## Command-Line Arguments

*   `path`: Required argument. The path to a `.txt` file or a directory containing `.txt` files.
*   `--workflow`: Optional argument. The file path for the workflow API JSON file. Defaults to `workflows/flux_api.json`.
*   `--positive-id` / `-pid`: Optional argument. The ID of the positive prompt node in the workflow. Defaults to `6`.
