#!/usr/bin/env python3
"""
餐具脏污图像生成提示词生成器

使用 Qwen3 模型根据餐具类型和脏污等级生成用于 Stable Diffusion 的完整提示词。
基于固定的模板格式生成逼真的餐具脏污图像描述。

使用方法：
    python scripts/dishware_dirt_generator.py --tableware "ceramic bowl" --dirtiness "Moderate"
    python scripts/dishware_dirt_generator.py --tableware "ceramic bowl" --dirtiness "Moderate" --temperature 1.2 --top-k 200
    python scripts/dishware_dirt_generator.py --default-batch --temperature 0.9
    python scripts/dishware_dirt_generator.py --batch-file tableware_list.txt --seed 42

功能：
    - 根据餐具类型和脏污等级生成完整的 Stable Diffusion 提示词
    - 支持单个生成或批量处理
    - 返回符合固定模板格式的完整提示词
    - 支持自定义模型和设备配置
"""

import argparse
import json
import sys
import random
import time
from pathlib import Path
from typing import Dict, List, Union, Tuple
import torch
from transformers import pipeline
import re

# 提示词模板
SYSTEM_PROMPT = """## Role and Goal
You are an expert AI prompt engineer and a product designer with a keen aesthetic sense. You specialize in creating vivid, specific, and photorealistic visual descriptions. You have a deep understanding of global cuisines, cooking methods, and the specific types of residue that different foods leave on various tableware.

Your goal is to take a user's high-level request, combine it with your expert knowledge and creativity, and populate an advanced template to generate a single, complete, high-quality prompt ready for Stable Diffusion.

## Context and Rules
1.  **Strictly Adhere to the New Template:** You must use the following fixed template structure to generate the final prompt.
    `Photorealistic, top-down view photo of a [color or pattern description] [tableware type] with [amount of food residue] of [dirtiness prompt], isolated on a solid chromakey green background, studio lighting, sharp focus`

2.  **Randomly Generate Color or Pattern:** This is your new creative task. You must generate a description for the `[color or pattern description]` placeholder.
    * **Plausibility:** The generated color or pattern must be realistic and appropriate for the given `[tableware type]`. For example, a "cast iron skillet" is suitable for "black, well-worn," but not "pink with polka dots." A "modern ceramic mug" can have various colors and patterns.
    * **Maximum Variety:** You should ALWAYS create unique and diverse descriptions. Avoid repetitive choices. Randomly choose to describe a solid color, a material finish, a pattern, or a combination. Draw inspiration from concepts like:
        * **Solid Color/Material:** `matte black`, `glossy white`, `earthy brown stoneware`, `clear glass`, `polished stainless steel`, `natural wood grain`, `dark grey metallic`, `cream-colored`, `deep blue ceramic`, `rustic terracotta`, `brushed copper`, `vintage pewter`
        * **Pattern:** `blue and white floral patterned`, `with simple geometric lines`, `hand-painted with a fish design`, `with a speckled glaze`, `with a gold rim`, `striped`, `dotted`, `with oriental motifs`, `with rustic country patterns`, `with modern abstract designs`
    * **Conciseness:** This description should be a concise adjective phrase.
    * **Creativity:** Be creative and avoid common combinations. Think of unique materials, finishes, and patterns that would be realistic for the tableware type.

3.  **Interpret "Dirtiness Level":** You must use the `[Dirtiness Level]` input (Slight, Moderate, or Heavy) to select an appropriate quantifier for the `[amount of food residue]` placeholder.
    * **Slight:** Use quantifiers for small amounts, like `a few`, `a trace of`, `a single splash of`, `a couple of`, `a faint smear of`.
    * **Moderate:** Use quantifiers for medium amounts, like `some`, `a moderate amount of`, `scattered residue of`, `a noticeable amount of`.
    * **Heavy:** Use quantifiers for large amounts, like `a thick layer of`, `a lot of`, `heavily caked with`, `piled up`.

4.  **Logical Association and Visualization:** The `[dirtiness prompt]` you create must be highly relevant to the `[tableware type]` and must be specific and visual. For example, use "congealed, greasy beef stew sauce" instead of just "sauce."
    * **Diverse Food Types:** Vary the type of food residue significantly. Consider different cuisines, cooking methods, and food textures: Asian dishes (soy sauce, curry, noodles), Western dishes (pasta sauce, gravy, cheese), baked goods (flour, chocolate, caramel), beverages (coffee stains, wine residue, tea marks), etc.
    * **Specific Descriptors:** Use precise visual descriptors like "crusty", "sticky", "dried", "caked-on", "splattered", "smeared", "burnt", "congealed", "greasy", "powdery", etc.

## Task Instructions
I will provide you with two inputs: a **[Tableware Type]** and a **[Dirtiness Level]** (Slight/Moderate/Heavy).
Your task is to:
1.  Analyze the two inputs.
2.  **Randomly generate a plausible and creative color or pattern description** for the given `[Tableware Type]`.
3.  Select an appropriate English quantifier that matches the `[Dirtiness Level]`.
4.  Create a specific, logical English `[dirtiness prompt]` that fits both the tableware and its generated appearance.
5.  Combine all these elements perfectly into the specified **new template**.
6.  Your final output **must be only the single, complete line of the English prompt**, with no additional explanations or text.

## Examples

### Example 1
**Input:**
* `Tableware Type: ceramic bowl`
* `Dirtiness Level: Slight`

**Your Correct Output (a possible random result):**
`Photorealistic, top-down view photo of a hand-painted blue and white patterned ceramic bowl with a few scattered grains of rice and a splash of soy sauce, isolated on a solid chromakey green background, studio lighting, sharp focus`

### Example 2
**Input:**
* `Tableware Type: cast iron skillet`
* `Dirtiness Level: Heavy`

**Your Correct Output (a possible random result):**
`Photorealistic, top-down view photo of a well-used, matte black cast iron skillet with a thick layer of burnt-on, greasy bacon remnants, isolated on a solid chromakey green background, studio lighting, sharp focus`

---
Now, please generate a prompt based on the following input."""

class DishwarePromptGenerator:
    """餐具脏污图像生成提示词生成器"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-8B", device: str = "auto", torch_dtype: str = "auto", 
                 temperature: float = 0.9, top_k: int = 100, top_p: float = 0.95, 
                 repetition_penalty: float = 1.1):
        """
        初始化生成器
        
        Args:
            model_name: 模型名称
            device: 设备类型
            torch_dtype: 数据类型
            temperature: 温度参数，越高越随机 (0.1-2.0)
            top_k: top-k 采样参数
            top_p: top-p 采样参数
            repetition_penalty: 重复惩罚参数
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.generator = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            print(f"正在加载模型: {self.model_name}")
            self.generator = pipeline(
                "text-generation",
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device,
                trust_remote_code=True
            )
            print("模型加载成功！")
        except Exception as e:
            print(f"模型加载失败: {e}")
            sys.exit(1)
    
    def generate_prompt(self, tableware_type: str, dirtiness_level: str, seed: int = None) -> str:
        """
        根据餐具类型和脏污等级生成完整的图像生成提示词
        
        Args:
            tableware_type: 餐具类型
            dirtiness_level: 脏污等级 (Slight/Moderate/Heavy)
            seed: 随机种子，用于控制生成的随机性
            
        Returns:
            完整的 Stable Diffusion 提示词
        """
        # 设置随机种子
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
        else:
            # 如果没有指定种子，使用时间戳增加随机性
            current_time = int(time.time() * 1000000) % 1000000
            torch.manual_seed(current_time)
            random.seed(current_time)
        
        # 构建完整的提示词，添加多样性提示和 /no_think 指令
        diversity_hints = [
            "Please be highly creative and vary the color/pattern and dirt type significantly from previous generations.",
            "Generate a unique and diverse description. Avoid common combinations and be creative with colors, patterns, and food residue types.",
            "Create a distinctive and original description. Use unusual but realistic color/pattern combinations and diverse food residue types.",
            "Be innovative in your description. Choose uncommon but plausible colors/patterns and vary the food residue significantly.",
            "Generate a creative and unique prompt. Explore different color schemes, patterns, and food types for maximum variety."
        ]
        
        diversity_hint = random.choice(diversity_hints)
        user_prompt = f"* **Tableware Type:** `{tableware_type}`\n* **Dirtiness Level:** `{dirtiness_level}`\n\n{diversity_hint}\n\n/no_think"
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # 生成回复 - 使用实例化时的参数
            result = self.generator(
                messages,
                max_new_tokens=256,
                do_sample=True,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                eos_token_id=[151645, 151643]
            )
            
            # 提取生成的文本
            generated_text = result[0]["generated_text"][-1]["content"]
            
            # 如果包含思考内容，先移除
            if "<think>" in generated_text and "</think>" in generated_text:
                # 移除思考内容
                generated_text = re.sub(r'<think>.*?</think>\s*', '', generated_text, flags=re.DOTALL)
            
            # 提取以 "Photorealistic" 开头的完整提示词
            prompt_match = re.search(r'Photorealistic.*?sharp focus', generated_text, re.DOTALL)
            if prompt_match:
                prompt = prompt_match.group(0).strip()
                # 清理多余的换行符和空格
                prompt = re.sub(r'\s+', ' ', prompt)
                return prompt
            else:
                print(f"无法提取有效的提示词: {generated_text}")
                return self._create_fallback_prompt(tableware_type, dirtiness_level)
                
        except Exception as e:
            print(f"生成过程中出错: {e}")
            return self._create_fallback_prompt(tableware_type, dirtiness_level)
    
    def _create_fallback_prompt(self, tableware_type: str, dirtiness_level: str) -> str:
        """创建后备提示词"""
        # 简单的后备逻辑
        amount_map = {
            "Slight": "a few traces of",
            "Moderate": "some scattered",
            "Heavy": "a thick layer of"
        }
        
        amount = amount_map.get(dirtiness_level, "some")
        
        return f"Photorealistic, top-down view photo of a white {tableware_type} with {amount} food residue, isolated on a solid chromakey green background, studio lighting, sharp focus"
    
    def batch_generate(self, tableware_data: List[Tuple[str, str]], use_random_seed: bool = True) -> Dict[str, str]:
        """
        批量生成提示词
        
        Args:
            tableware_data: 包含 (餐具类型, 脏污等级) 元组的列表
            use_random_seed: 是否为每个生成使用不同的随机种子
            
        Returns:
            包含所有生成结果的字典，键为 "餐具类型_脏污等级"
        """
        results = {}
        
        for i, (tableware_type, dirtiness_level) in enumerate(tableware_data, 1):
            key = f"{tableware_type}_{dirtiness_level}"
            print(f"正在处理 ({i}/{len(tableware_data)}): {tableware_type} - {dirtiness_level}")
            
            # 为每个生成使用不同的随机种子以增加多样性
            seed = i * 1000 if use_random_seed else None
            prompt = self.generate_prompt(tableware_type, dirtiness_level, seed=seed)
            results[key] = {
                "tableware_type": tableware_type,
                "dirtiness_level": dirtiness_level,
                "prompt": prompt,
                "seed": seed
            }
            
            print(f"完成: {key}")
            print(f"提示词: {prompt}")
            print("-" * 80)
        
        return results

def setup_args():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(
        description="餐具脏污图像生成提示词生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="模型名称 (默认: Qwen/Qwen3-8B)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="设备类型 (默认: auto)"
    )
    
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        help="数据类型 (默认: auto)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="温度参数，越高越随机 (默认: 0.9，范围: 0.1-2.0)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="top-k 采样参数 (默认: 100)"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="top-p 采样参数 (默认: 0.95)"
    )
    
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="重复惩罚参数 (默认: 1.1)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        help="随机种子，用于单次生成的可重现性"
    )
    
    parser.add_argument(
        "--tableware",
        type=str,
        help="单个餐具类型 (如: 'ceramic bowl')"
    )
    
    parser.add_argument(
        "--dirtiness",
        type=str,
        choices=["Slight", "Moderate", "Heavy"],
        help="脏污等级 (Slight/Moderate/Heavy)"
    )
    
    parser.add_argument(
        "--batch-file",
        type=str,
        help="批量处理文件，每行格式: 餐具类型,脏污等级"
    )
    
    parser.add_argument(
        "--default-batch",
        action="store_true",
        help="使用默认的批量处理列表"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="dishware_prompts.json",
        help="输出文件名 (默认: dishware_prompts.json)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出"
    )
    
    return parser.parse_args()

def load_batch_data(batch_file: str) -> List[Tuple[str, str]]:
    """从文件加载批量数据"""
    data = []
    with open(batch_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(',')
                if len(parts) == 2:
                    tableware_type = parts[0].strip()
                    dirtiness_level = parts[1].strip()
                    data.append((tableware_type, dirtiness_level))
    return data

def get_default_batch_data() -> List[Tuple[str, str]]:
    """获取默认的批量数据"""
    tableware_types = ["ceramic bowl", "wine glass", "cast iron skillet", "dinner plate", "coffee mug", "soup spoon"]
    dirtiness_levels = ["Slight", "Moderate", "Heavy"]
    
    data = []
    for tableware in tableware_types:
        for dirtiness in dirtiness_levels:
            data.append((tableware, dirtiness))
    
    return data

def main():
    """主函数"""
    args = setup_args()
    
    # 创建生成器
    generator = DishwarePromptGenerator(
        model_name=args.model,
        device=args.device,
        torch_dtype=args.torch_dtype,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty
    )
    
    # 确定处理模式
    if args.tableware and args.dirtiness:
        # 单个生成
        print(f"生成单个提示词: {args.tableware} - {args.dirtiness}")
        print(f"生成参数: temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, repetition_penalty={args.repetition_penalty}")
        if args.seed:
            print(f"使用随机种子: {args.seed}")
        
        prompt = generator.generate_prompt(args.tableware, args.dirtiness, seed=args.seed)
        
        result = {
            "single_generation": {
                "tableware_type": args.tableware,
                "dirtiness_level": args.dirtiness,
                "prompt": prompt,
                "seed": args.seed,
                "generation_params": {
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                    "repetition_penalty": args.repetition_penalty
                }
            }
        }
        
        print(f"\n生成的提示词:")
        print(prompt)
        
    elif args.batch_file:
        # 从文件批量生成
        batch_data = load_batch_data(args.batch_file)
        print(f"从文件加载 {len(batch_data)} 个任务...")
        result = generator.batch_generate(batch_data)
        
    elif args.default_batch:
        # 默认批量生成
        batch_data = get_default_batch_data()
        print(f"使用默认批量数据，共 {len(batch_data)} 个任务...")
        result = generator.batch_generate(batch_data)
        
    else:
        print("错误: 请指定 --tableware 和 --dirtiness 进行单个生成，或使用 --batch-file 或 --default-batch 进行批量生成")
        sys.exit(1)
    
    # 保存结果
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_path}")

if __name__ == "__main__":
    main() 