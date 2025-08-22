# 文档: `filter_unmatched_images.py`

此脚本用于自动筛选图像数据集。它通过从文件名生成提示词，计算图文 CLIP 相似度，并使用自适应阈值来移动与描述不匹配的图像。

## 功能概述

在生成大量图像后，此脚本旨在自动化筛选过程，剔除那些质量不佳或与预期内容不符的图像。

它的工作流程分为两个阶段：

1.  **第一阶段：计算相似度**
    *   脚本会递归地查找输入路径中的所有图像文件。
    *   从每个图像的文件名中提取其“脏污等级”（例如 `clean`, `slightly_dirty`, `moderately_dirty`, `heavily_dirty`）。
    *   根据提取的等级，自动生成一个简单的文本提示词（例如，`"a photo of a clean of tableware"`）。
    *   使用 CLIP 模型计算图像与生成的提示词之间的相似度分数，并将分数按脏污等级分类存储。

2.  **第二阶段：自适应筛选与移动**
    *   对于每一个脏污等级类别，脚本会计算该类别下所有图像相似度分数的**平均值（mean）**和**标准差（std dev）**。
    *   基于这些统计数据，它会动态计算一个**自适应阈值**，公式为：`threshold = mean - k * std_dev`。这里的 `k` 是一个可调节的因子。
    *   最后，脚本会再次遍历该类别的图像，将任何相似度分数低于这个动态阈值的图像移动到指定的输出目录。

这种自适应方法比固定阈值更智能，因为它能根据每个类别内数据自身的分布情况来判断哪些是异常值。

## 使用方法

通过命令行运行此脚本，并提供包含图像的输入路径。

```bash
python scripts/filter_unmatched_images.py -i <input_path> -o <output_directory> -k <std_dev_factor>
```

### 示例

*   **处理整个数据集目录**：
    假设你的图像数据集存放在 `my_dataset/` 目录下，你想将不匹配的图片移动到 `unmatched_images/` 目录，并使用一个较为宽松的因子 `k=1.2`。

    ```bash
    python scripts/filter_unmatched_images.py -i my_dataset/ -o unmatched_images/ -k 1.2
    ```

*   **处理单个图片文件**：
    虽然主要是为目录设计的，但也可以处理单个文件。

    ```bash
    python scripts/filter_unmatched_images.py -i my_dataset/clean/clean_001.png
    ```

*   **使用更严格的筛选**：
    增大 `k` 的值会使阈值降低，从而保留更多图片（筛选更宽松）。减小 `k` 的值会使阈值升高，从而移动更多图片（筛选更严格）。

    ```bash
    # 使用 k=2.0 的标准差因子进行更宽松的筛选
    python scripts/filter_unmatched_images.py -i my_dataset/ -k 2.0
    ```

## 命令行参数

*   `-i`, `--input`: **必需参数**。指定输入的路径，可以是一个图像文件，也可以是一个包含图像文件的目录。脚本会递归搜索该目录。
*   `-o`, `--output`: 指定用于存放低相似度图像的输出目录。默认为 `./unmatched`。
*   `-k`, `--std-dev-factor`: 用于计算自适应阈值的标准差因子。`threshold = mean - k * std`。一个较大的k值意味着一个更低的（更宽松的）阈值。默认为 `1.5`。
*   `--model-name`: 指定要使用的 CLIP 模型名称（从 Hugging Face Hub）。默认为 `openai/clip-vit-base-patch32`。
