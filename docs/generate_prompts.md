# 文档: `generate_prompts.py`

此脚本用于根据 YAML 配置文件大规模地生成用于文生图模型的提示词（prompts），专门用于创建餐具图像数据集。

## 功能概述

脚本通过组合不同维度的描述来自动生成大量的、多样化的提示词。主要功能包括：

1.  **基于配置**：从多个 YAML 文件中加载描述片段，包括：
    *   餐具描述 (`tableware_description.yaml`)
    *   脏污描述 (`dirtiness_description_origin.yaml`)
    *   背景描述 (`background.yaml`)
2.  **组合生成**：根据用户指定的**餐具类型** (`plate` 或 `bowl`)、**脏污等级** (`clean`, `slightly_dirty`, `moderately_dirty`, `heavily_dirty`) 和**数量**，随机组合这些描述片段来构建最终的提示词。
3.  **结构化输出**：将生成的每个提示词保存为单独的 `.txt` 文件，并按照 `output/<tableware_type>/<dirtiness_level>/` 的目录结构进行组织。
4.  **可复现性**：支持设置随机种子，以确保每次运行生成相同的数据集。

这个脚本极大地简化了创建大规模、结构化图文数据集的过程。

## 配置文件格式

脚本依赖于 YAML 配置文件来提供生成提示词的“素材”。

*   **`tableware_description.yaml`**: 定义了不同类型餐具的描述。
    ```yaml
    plate:
      - "a white ceramic plate"
      - "a blue patterned porcelain plate # 中文注释会被忽略"
    bowl:
      - "a deep stoneware bowl"
    ```
*   **`dirtiness_description_origin.yaml`**: 定义了不同餐具和脏污等级下的具体脏污描述。
    ```yaml
    plate:
      slightly_dirty:
        - "with a few crumbs"
      moderately_dirty:
        - "with some leftover sauce stains"
    ```
*   **`background.yaml`**: 定义了图像的背景描述。
    ```yaml
    backgrounds:
      - "on a wooden table"
      - "on a marble countertop"
    ```

## 使用方法

通过命令行运行脚本，并提供所需的参数来控制生成过程。

```bash
python scripts/generate_prompts.py --tableware-type <type> --dirtiness-level <level> -n <num_prompts> [options]
```

### 示例

*   **为盘子生成1000条“轻微脏污”的提示词**：

    ```bash
    python scripts/generate_prompts.py --tableware-type plate --dirtiness-level slightly_dirty -n 1000
    ```
    这将在 `output/plate/slightly_dirty/` 目录下生成1000个 `.txt` 文件。

*   **为碗生成500条“干净”的提示词，并指定输出目录和配置文件**：

    ```bash
    python scripts/generate_prompts.py \
        --tableware-type bowl \
        --dirtiness-level clean \
        -n 500 \
        -t my_prompts/tableware.yaml \
        -b my_prompts/backgrounds.yaml \
        -o my_dataset/
    ```

## 命令行参数

*   `--tableware-type`: **必需参数**。指定餐具类型。可选值为: `plate`, `bowl`。
*   `--dirtiness-level`: **必需参数**。指定脏污等级。可选值为: `clean`, `slightly_dirty`, `moderately_dirty`, `heavily_dirty`。
*   `-n`, `--num-prompts`: **必需参数**。指定要生成的提示词数量。
*   `-t`, `--tableware`: 餐具描述 YAML 文件的路径。默认为 `prompts/tableware_description.yaml`。
*   `-d`, `--dirtiness`: 脏污描述 YAML 文件的路径。默认为 `prompts/dirtiness_description_origin.yaml`。
*   `-b`, `--background`: 背景描述 YAML 文件的路径。默认为 `prompts/background.yaml`。
*   `-o`, `--output`: 输出的根目录。默认为 `output`。
*   `--digits`: 输出文件名的序号位数。默认为 `5` (例如, `clean_00001.txt`)。
*   `--seed`: 随机种子，用于可复现的生成。默认为 `42`。
