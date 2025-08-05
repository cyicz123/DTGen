# CLIP图片文本相似度计算工具

这个工具用于计算合成图片与对应描述的匹配程度，使用OpenAI的CLIP模型（Vision Transformer patch14版本）进行图片-文本相似度计算。

## 功能特性

- 🔍 自动查找output文件夹下的所有`*_prompts.txt`文件
- 📊 使用CLIP ViT-Large-Patch14模型计算图片与文本的相似度
- 📝 解析prompts文件，提取图片名称和描述信息
- 💾 将相似度结果保存到文件，按相似度排序
- 🗑️ 支持自动移动低相似度图片到待删除文件夹
- 📈 提供详细的统计信息（平均值、最值、标准差等）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
python clip_similarity_calculator.py
```

这将使用默认参数：
- 从`output`文件夹查找prompts文件
- 在`datasets/synthetic`文件夹查找对应图片
- 将结果保存到`similarity_results.txt`

### 完整参数说明

```bash
python clip_similarity_calculator.py \
    --output-dir output \
    --synthetic-dir datasets/synthetic \
    --result-file similarity_results.txt \
    --del 0.5 \
    --to-delete-dir to_delete \
    --model openai/clip-vit-large-patch14
```

#### 参数详解

- `--output-dir`: prompts文件所在的目录（默认：`output`）
- `--synthetic-dir`: 合成图片所在的目录（默认：`datasets/synthetic`）
- `--result-file`: 结果输出文件路径（默认：`similarity_results.txt`）
- `--del`: 相似度阈值，低于此值的图片将被移动到待删除文件夹（可选）
- `--to-delete-dir`: 待删除图片存放目录（默认：`to_delete`）
- `--model`: CLIP模型名称（默认：`openai/clip-vit-large-patch14`）

### 使用示例

#### 1. 仅计算相似度，不移动文件

```bash
python clip_similarity_calculator.py --result-file my_results.txt
```

#### 2. 计算相似度并移动低质量图片

```bash
python clip_similarity_calculator.py --del 0.3 --to-delete-dir low_quality_images
```

#### 3. 使用不同的CLIP模型

```bash
python clip_similarity_calculator.py --model openai/clip-vit-base-patch32
```

## 文件格式说明

### Prompts文件格式

工具会自动解析以下格式的prompts文件：

```
name: bowl_clean_00045

positive:empty, a small, square-shaped snack bowl, photorealistic, top-down view, on a wall with peeling layers of posters, gritty urban light, sharp focus

negative:
---
name: bowl_clean_00410

positive:empty, a deep purple, almost black, ceramic bowl, photorealistic, top-down view, on a fabric with a black and white Herringbone weave, crisp studio light, sharp focus

negative:
---
```

### 输出结果格式

结果文件包含以下信息：
- 图片名称
- 文本描述（截取前50字符）
- 相似度分数（0-1之间）
- 图片完整路径
- 对应的prompts文件路径

## 工作流程

1. **扫描文件**: 递归扫描output文件夹，查找所有`*_prompts.txt`文件
2. **解析内容**: 解析每个prompts文件，提取`name`和`positive`字段
3. **加载模型**: 加载CLIP ViT-Large-Patch14模型
4. **计算相似度**: 对每个图片-文本对计算相似度分数
5. **保存结果**: 将结果按相似度降序保存到文件
6. **处理低质量图片**: 如果指定了`--del`参数，将低相似度图片移动到指定文件夹
7. **输出统计**: 显示平均相似度、最值、标准差等统计信息

## 性能优化

- 自动检测并使用GPU加速（如果可用）
- 使用批处理提高效率
- 显示进度条跟踪处理状态

## 注意事项

1. **图片格式**: 默认假设图片为PNG格式，文件名与prompts中的name字段对应
2. **内存使用**: 处理大量图片时可能需要较大内存，建议分批处理
3. **模型下载**: 首次运行时会自动下载CLIP模型，需要网络连接
4. **相似度阈值**: 建议根据实际需求调整`--del`参数的阈值

## 输出示例

```
正在加载CLIP模型: openai/clip-vit-large-patch14
模型已加载到设备: cuda
找到 8 个prompts文件
处理文件: output/bowl/clean/bowl_clean_prompts.txt
  找到 500 个条目
计算相似度: 100%|██████████| 500/500 [02:15<00:00,  3.69it/s]
...
共处理了 4000 个图片-文本对
结果已保存到: similarity_results.txt

处理相似度低于 0.3 的图片...
已移动低相似度图片: datasets/synthetic/bowl_clean_00123.png -> to_delete/bowl_clean_00123.png (相似度: 0.2543)
...
共移动了 156 个低相似度图片到 to_delete

统计信息:
  平均相似度: 0.7234
  最高相似度: 0.9876
  最低相似度: 0.1234
  标准差: 0.1456
  低于阈值 0.3 的图片数量: 156
```

## 故障排除

1. **CUDA内存不足**: 减少批处理大小或使用CPU模式
2. **图片文件不存在**: 检查图片路径和文件名是否正确
3. **模型下载失败**: 检查网络连接或使用代理
4. **权限错误**: 确保有足够权限读写相关目录

## 技术细节

- 使用Hugging Face Transformers库
- 支持CUDA和CPU计算
- 基于余弦相似度计算图片-文本匹配度
- 使用softmax归一化相似度分数
