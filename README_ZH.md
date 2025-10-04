# DTGen: 基于生成式扩散模型的少样本餐具污渍识别数据增强方案

## 📖 项目简介

**DTGen** 是一个基于生成式扩散模型的少样本数据增强框架，专为细粒度餐具污渍识别而设计。在现实世界的智能清洁和食品安全监控应用中，获取大量且多样化的标注数据成本高昂，本项目旨在解决这一"数据稀缺"的痛点。

借助最新的生成式人工智能技术，DTGen 能够利用极少数（例如仅40张）真实样本，合成数千张高质量、多样化的虚拟餐具污渍图像。这些合成数据可以显著提升分类器在细粒度识别任务上的性能，为开发更智能、更高效的自动化餐具清洁系统提供了可行的技术路径。

### 核心特性

  * **高效领域自适应**：通过参数高效微调（PEFT）技术 **LoRA**，使通用扩散模型快速学习餐具的材质、形状和污渍的特定视觉特征。
  * **结构化提示生成**：设计了一套层次化的提示词模板，通过组合不同的餐具类型、样式、污渍描述和环境，系统性地生成多样化的图像。
  * **跨模态质量过滤**：利用 **CLIP** 模型的图文匹配能力，自动筛选掉与文本描述不符或质量较低的生成图像，确保合成数据集的语义准确性和可靠性。
  * **端到端工作流**：提供从提示词生成、图像合成、质量过滤到数据整理的全套脚本，方便用户快速上手。

该项目是论文 **《DTGen: Generative Diffusion-Based Few-Shot Data Augmentation for Fine-Grained Dirty Tableware Recognition》** 的官方代码实现。

## 📂 项目结构

```
.
├── datasets
│   └── sd3.5-synthetic       # 存放最终生成并筛选后的高质量合成数据集
├── docs
│   ├── ...                   # 中英文文档，详细说明各脚本的使用方法
│   └── sam.md
├── models
│   └── drt_tableware_lora.safetensors # 预训练的餐具污渍LoRA模型文件
├── output
│   ├── bowl                  # 生成的碗的图像
│   └── plate                 # 生成的盘子的图像
├── prompts
│   ├── background.yaml       # 背景和光照风格的描述
│   ├── dirtiness_description.yaml # 污渍描述
│   └── tableware_description.yaml # 餐具类型和样式的描述
├── scripts
│   ├── batch_prompt.py       # 调用API生成图像数据集脚本
│   ├── filter_unmatched_images.py # 过滤不匹配图像脚本
│   ├── generate_prompts.py   # 批量生成提示词脚本
│   └── sam.py                # 分割餐具用于标注的辅助脚本
├── unmatched                 # 存放被CLIP模型过滤掉的低质量或不匹配图像
└── workflows
    ├── flux_api.json         # ComfyUI的Flux模型API工作流
    ├── sd3.5_no_lora.json    # ComfyUI的Stable Diffusion 3.5 (无LoRA) API工作流
    └── sdxl-api.json         # ComfyUI的SDXL模型API工作流
```

## 🚀 快速开始

### 1\. 环境配置

本项目依赖 Python 环境和一些第三方库。建议使用 `uv` 来管理依赖。

```bash
# 安装依赖
uv sync
```

### 2\. 生成多样化的提示词

运行 `generate_prompts.py` 脚本，为不同类别的餐具和污渍等级生成结构化的提示词。你需要为每个组合分别运行此脚本。

例如，以下命令将为**盘子**生成450条“轻微脏污”的提示词：
```bash
python scripts/generate_prompts.py --tableware-type plate --dirtiness-level slightly_dirty -n 450
```

为了方便地为所有类别生成提示词，你可以使用如下的循环脚本：

```bash
#!/bin/bash
# 为所有类别生成提示词

NUM_PROMPTS=450 # 每个类别生成的数量

for tableware in "plate" "bowl"; do
  for dirtiness in "clean" "slightly_dirty" "moderately_dirty" "heavily_dirty"; do
    echo "Generating prompts for ${tableware} - ${dirtiness}..."
    python scripts/generate_prompts.py \
      --tableware-type "${tableware}" \
      --dirtiness-level "${dirtiness}" \
      -n "${NUM_PROMPTS}"
  done
done

echo "All prompts generated."
```

执行后，所有生成的提示词将根据类别保存在 `output/` 目录下的对应子文件夹中（例如 `output/plate/slightly_dirty/`）。

### 3\. 合成图像数据集

接下来，使用 `batch_prompt.py` 脚本调用生成模型的API（例如ComfyUI部署的Stable Diffusion 3.5）来合成图像。请确保你的生成服务正在运行，并已加载 `models/drt_tableware_lora.safetensors` LoRA模型。

```bash
# 确保ComfyUI等后端服务已启动并加载了正确的工作流 (workflows/)
python scripts/batch_prompt.py output/
```

生成的图像将根据提示词中的类别（如 `bowl`, `plate`）保存在 `output/` 目录下的对应文件夹中。

### 4\. 筛选高质量图像

为了保证数据集的质量，我们使用 `filter_unmatched_images.py` 脚本来剔除生成效果不佳的样本。该脚本利用CLIP模型计算图像与提示词的相似度，并根据一个自适应阈值进行过滤。

该脚本会遍历 `output/` 目录，并将识别出的低质量或不匹配的图像移动到 `unmatched/` 目录，而高质量的图像会保留在原位。

```bash
# 运行筛选脚本，它会自动处理 output/ 目录下的所有图像
python scripts/filter_unmatched_images.py --input output/ --output unmatched/
```

筛选完成后，`output/` 目录中剩下的就是高质量的图像。最后，我们将这些图像移动到最终的数据集目录 `datasets/sd3.5-synthetic` 以备训练：

```bash
# 确保目标目录存在
mkdir -p datasets/sd3.5-synthetic

# 将筛选后留下的高质量图像移动到最终数据集目录
# (此操作会移动 output/ 下的 plate, bowl 等文件夹)
mv output/* datasets/sd3.5-synthetic/
```

## ⚙️ 工作流与模型

  * **LoRA模型**：`models/drt_tableware_lora.safetensors` 是本项目的核心，它封装了关于餐具和特定污渍的领域知识。使用时，请在扩散模型（如Stable Diffusion 3.5）中加载此LoRA权重。
  * **API工作流**：`workflows/` 目录下的JSON文件是为ComfyUI设计的API模板。你可以根据自己的后端部署情况进行修改。

## 📊 实验结果

我们的研究表明，使用DTGen生成的合成数据训练的分类器，性能远超仅使用少量真实样本或传统数据增强方法的模型。

  * **二分类（干净 vs. 脏）**：准确率达到 **93%**，相比少样本基线提升了 **28%**。
  * **三分类（干净、轻度脏、重度脏）**：准确率达到 **86%**，显著优于传统增强方法的 **71%**。

更多详细信息，请参阅我们的论文。

## 引用

如果您在研究中使用了本项目的代码或思路，请引用我们的论文：

```
@misc{hao2025dtgengenerativediffusionbasedfewshot,
      title={DTGen: Generative Diffusion-Based Few-Shot Data Augmentation for Fine-Grained Dirty Tableware Recognition}, 
      author={Lifei Hao and Yue Cheng and Baoqi Huang and Bing Jia and Xuandong Zhao},
      year={2025},
      eprint={2509.11661},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.11661}, 
}
```

## 致谢

本项目的部分研究得到了国家自然科学基金等多个项目的支持。同时感谢 `Cleaned vs Dirty V2` 数据集的提供者。