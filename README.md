# DTGen: Few-Shot Data Augmentation for Fine-Grained Dirty Tableware Recognition Based on Generative Diffusion Models

## 📖 Project Overview

**DTGen** is a few-shot data augmentation framework based on generative diffusion models, specifically designed for fine-grained dirty tableware recognition. In real-world intelligent cleaning and food safety monitoring applications, acquiring large amounts of diverse labeled data is expensive. This project aims to address this "data scarcity" challenge.

Leveraging cutting-edge generative AI technology, DTGen can utilize extremely few (e.g., only 40) real samples to synthesize thousands of high-quality, diverse virtual dirty tableware images. These synthetic data can significantly improve classifier performance on fine-grained recognition tasks, providing a viable technical pathway for developing smarter and more efficient automated tableware cleaning systems.

### Core Features

  * **Efficient Domain Adaptation**: Through parameter-efficient fine-tuning (PEFT) technique **LoRA**, enables general diffusion models to quickly learn specific visual features of tableware materials, shapes, and stains.
  * **Structured Prompt Generation**: Designed a hierarchical prompt template system that systematically generates diverse images by combining different tableware types, styles, dirt descriptions, and environments.
  * **Cross-Modal Quality Filtering**: Utilizes **CLIP** model's image-text matching capability to automatically filter out generated images that don't match text descriptions or are of low quality, ensuring semantic accuracy and reliability of the synthetic dataset.
  * **End-to-End Workflow**: Provides a complete set of scripts from prompt generation, image synthesis, quality filtering to data organization, making it easy for users to get started quickly.

This project is the official code implementation of the paper **"DTGen: Generative Diffusion-Based Few-Shot Data Augmentation for Fine-Grained Dirty Tableware Recognition"**.

## 📂 Project Structure

```
.
├── datasets
│   └── sd3.5-synthetic       # Final generated and filtered high-quality synthetic dataset
├── docs
│   ├── ...                   # Chinese and English documentation with detailed script usage instructions
│   └── sam.md
├── models
│   └── drt_tableware_lora.safetensors # Pre-trained dirty tableware LoRA model file
├── output
│   ├── bowl                  # Generated bowl images
│   └── plate                 # Generated plate images
├── prompts
│   ├── background.yaml       # Background and lighting style descriptions
│   ├── dirtiness_description.yaml # Dirt descriptions
│   └── tableware_description.yaml # Tableware types and style descriptions
├── scripts
│   ├── batch_prompt.py       # Script for calling API to generate image dataset
│   ├── filter_unmatched_images.py # Script for filtering unmatched images
│   ├── generate_prompts.py   # Script for batch generating prompts
│   └── sam.py                # Auxiliary script for segmenting tableware for annotation
├── unmatched                 # Low-quality or unmatched images filtered out by CLIP model
└── workflows
    ├── flux_api.json         # ComfyUI Flux model API workflow
    ├── sd3.5_no_lora.json    # ComfyUI Stable Diffusion 3.5 (no LoRA) API workflow
    └── sdxl-api.json         # ComfyUI SDXL model API workflow
```

## 🚀 Quick Start

### 1. Environment Setup

This project depends on Python environment and some third-party libraries. We recommend using `uv` to manage dependencies.

```bash
# Install dependencies
uv sync
```

### 2. Generate Diverse Prompts

Run the `generate_prompts.py` script to generate structured prompts for different categories of tableware and dirt levels. You need to run this script separately for each combination.

For example, the following command will generate 450 "slightly dirty" prompts for **plates**:
```bash
python scripts/generate_prompts.py --tableware-type plate --dirtiness-level slightly_dirty -n 450
```

To conveniently generate prompts for all categories, you can use the following loop script:

```bash
#!/bin/bash
# Generate prompts for all categories

NUM_PROMPTS=450 # Number of prompts to generate for each category

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

After execution, all generated prompts will be saved in corresponding subfolders under the `output/` directory according to category (e.g., `output/plate/slightly_dirty/`).

### 3. Synthesize Image Dataset

Next, use the `batch_prompt.py` script to call the generative model API (e.g., ComfyUI deployed Stable Diffusion 3.5) to synthesize images. Please ensure your generation service is running and has loaded the `models/drt_tableware_lora.safetensors` LoRA model.

```bash
# Ensure ComfyUI and other backend services are started and loaded with correct workflows (workflows/)
python scripts/batch_prompt.py output/
```

Generated images will be saved in corresponding folders under the `output/` directory according to the categories in the prompts (such as `bowl`, `plate`).

### 4. Filter High-Quality Images

To ensure dataset quality, we use the `filter_unmatched_images.py` script to remove poorly generated samples. This script utilizes the CLIP model to calculate similarity between images and prompts, filtering based on an adaptive threshold.

The script will traverse the `output/` directory and move identified low-quality or unmatched images to the `unmatched/` directory, while high-quality images remain in place.

```bash
# Run the filtering script, it will automatically process all images in the output/ directory
python scripts/filter_unmatched_images.py --input output/ --output unmatched/
```

After filtering, the `output/` directory will contain only high-quality images. Finally, we move these images to the final dataset directory `datasets/sd3.5-synthetic` for training:

```bash
# Ensure target directory exists
mkdir -p datasets/sd3.5-synthetic

# Move the filtered high-quality images to the final dataset directory
# (This operation will move folders like plate, bowl, etc. under output/)
mv output/* datasets/sd3.5-synthetic/
```

## ⚙️ Workflows and Models

  * **LoRA Model**: `models/drt_tableware_lora.safetensors` is the core of this project, encapsulating domain knowledge about tableware and specific stains. When using, please load this LoRA weight in diffusion models (such as Stable Diffusion 3.5).
  * **API Workflows**: JSON files in the `workflows/` directory are API templates designed for ComfyUI. You can modify them according to your backend deployment situation.

## 📊 Experimental Results

Our research shows that classifiers trained with synthetic data generated by DTGen significantly outperform models using only few real samples or traditional data augmentation methods.

  * **Binary Classification (Clean vs. Dirty)**: Achieved **93%** accuracy, a **28%** improvement over the few-shot baseline.
  * **Three-Class Classification (Clean, Lightly Dirty, Heavily Dirty)**: Achieved **86%** accuracy, significantly outperforming traditional augmentation methods' **71%**.

For more detailed information, please refer to our paper.

## Citation

If you use the code or ideas from this project in your research, please cite our paper:

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

## Acknowledgments

Part of this project's research was supported by multiple projects including the National Natural Science Foundation. We also thank the providers of the `Cleaned vs Dirty V2` dataset.
