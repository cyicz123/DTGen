# 文档：`batch_prompt.py`

此脚本用于根据文本文件中的提示词批量生成图像。

## 功能

该脚本会连接到ComfyUI的后端服务，并使用给定的工作流（workflow）来生成图像。它会遍历指定的路径（可以是单个文件或目录），读取 `.txt` 文件中的提示词，然后为每个提示词生成一张或多张图像。

主要功能点：

*   **连接服务**：通过 WebSocket 连接到在 `127.0.0.1:8188` 上运行的服务。
*   **读取提示词**：从指定的 `.txt` 文件或目录中读取提示词。脚本会查找文件名以 `_ddddd.txt`（d代表数字）结尾的文件。
*   **工作流模板**：使用一个 JSON 文件作为工作流模板（默认为 `workflows/flux_api.json`）。
*   **注入提示词**：将从文件中读取的提示词注入到工作流的指定节点中。
*   **生成并保存图像**：调用服务生成图像，并将其保存在与对应 `.txt` 文件相同的目录下。
*   **跳过已存在文件**：在生成图像之前，会检查是否已存在同名的输出文件，如果存在则跳过。

## 使用方法

你可以通过命令行来运行此脚本。

```bash
python scripts/batch_prompt.py <path_to_prompts> [options]
```

### 示例

*   处理单个文件：
    ```bash
    python scripts/batch_prompt.py /path/to/your/prompt_00001.txt
    ```

*   处理目录下的所有匹配文件：
    ```bash
    python scripts/batch_prompt.py /path/to/your/prompts_directory/
    ```

*   使用自定义的工作流和节点ID：
    ```bash
    python scripts/batch_prompt.py /path/to/prompts/ --workflow /path/to/custom_workflow.json --positive-id 10
    ```

## 命令行参数

*   `path`：必需参数。指定一个 `.txt` 文件或一个包含 `.txt` 文件的目录的路径。
*   `--workflow`：可选参数。指定工作流API的JSON文件路径。默认为 `workflows/flux_api.json`。
*   `--positive-id` / `-pid`：可选参数。指定工作流中正向提示词节点的ID。默认为 `6`。
