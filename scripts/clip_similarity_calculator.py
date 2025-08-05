#!/usr/bin/env python3
"""
CLIP相似度计算工具
用于计算合成图片与对应描述的匹配程度
"""

import os
import re
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from tqdm import tqdm


class CLIPSimilarityCalculator:
    """CLIP相似度计算器"""
    
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        """
        初始化CLIP模型
        
        Args:
            model_name: CLIP模型名称，使用patch14版本
        """
        print(f"正在加载CLIP模型: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        print(f"模型已加载到设备: {self.device}")
    
    def parse_prompt_file(self, file_path: str) -> List[Dict[str, str]]:
        """
        解析prompts文件，提取name和positive字段
        
        Args:
            file_path: prompts文件路径
            
        Returns:
            包含name和positive字段的字典列表
        """
        results = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 按照---分隔符分割
        entries = content.split('---')
        
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue
                
            # 提取name字段
            name_match = re.search(r'name:\s*(.+)', entry)
            # 提取positive字段
            positive_match = re.search(r'positive:\s*(.+?)(?=\n|$)', entry, re.DOTALL)
            
            if name_match and positive_match:
                name = name_match.group(1).strip()
                positive = positive_match.group(1).strip()
                
                results.append({
                    'name': name,
                    'positive': positive
                })
        
        return results
    
    def find_prompt_files(self, output_dir: str) -> List[str]:
        """
        在output文件夹下递归查找*_prompts.txt文件
        
        Args:
            output_dir: output目录路径
            
        Returns:
            prompts文件路径列表
        """
        prompt_files = []
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith('_prompts.txt'):
                    prompt_files.append(os.path.join(root, file))
        return prompt_files
    
    def calculate_similarity(self, image_path: str, text: str) -> float:
        """
        计算图片与文本的相似度
        
        Args:
            image_path: 图片路径
            text: 文本描述
            
        Returns:
            相似度分数 (0-1之间)
        """
        try:
            # 加载图片
            image = Image.open(image_path).convert('RGB')
            
            # 处理输入
            inputs = self.processor(
                text=[text], 
                images=image, 
                return_tensors="pt", 
                padding=True
            )
            
            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 计算相似度
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                # 使用softmax获取概率
                probs = logits_per_image.softmax(dim=1)
                similarity = probs[0][0].cpu().item()
            
            return similarity
            
        except Exception as e:
            print(f"计算相似度时出错 ({image_path}): {e}")
            return 0.0
    
    def process_all_files(self, output_dir: str, synthetic_dir: str) -> List[Dict]:
        """
        处理所有prompts文件并计算相似度
        
        Args:
            output_dir: output目录路径
            synthetic_dir: synthetic图片目录路径
            
        Returns:
            包含所有结果的字典列表
        """
        results = []
        
        # 查找所有prompts文件
        prompt_files = self.find_prompt_files(output_dir)
        print(f"找到 {len(prompt_files)} 个prompts文件")
        
        for prompt_file in prompt_files:
            print(f"处理文件: {prompt_file}")
            
            # 解析prompts文件
            entries = self.parse_prompt_file(prompt_file)
            print(f"  找到 {len(entries)} 个条目")
            
            # 为每个条目计算相似度
            for entry in tqdm(entries, desc=f"计算相似度"):
                image_name = entry['name']
                text_description = entry['positive']
                
                # 构建图片路径 (假设图片是.png格式)
                image_path = os.path.join(synthetic_dir, f"{image_name}.png")
                
                if os.path.exists(image_path):
                    similarity = self.calculate_similarity(image_path, text_description)
                    
                    results.append({
                        'image_name': image_name,
                        'image_path': image_path,
                        'text_description': text_description,
                        'similarity': similarity,
                        'prompt_file': prompt_file
                    })
                else:
                    print(f"  警告: 图片不存在 {image_path}")
        
        return results
    
    def save_results(self, results: List[Dict], output_file: str):
        """
        保存结果到文件
        
        Args:
            results: 结果列表
            output_file: 输出文件路径
        """
        # 按相似度降序排序
        results_sorted = sorted(results, key=lambda x: x['similarity'], reverse=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("图片名称\t文本描述\t相似度\t图片路径\tPrompts文件\n")
            f.write("=" * 100 + "\n")
            
            for result in results_sorted:
                f.write(f"{result['image_name']}\t")
                f.write(f"{result['text_description'][:50]}...\t")
                f.write(f"{result['similarity']:.4f}\t")
                f.write(f"{result['image_path']}\t")
                f.write(f"{result['prompt_file']}\n")
        
        print(f"结果已保存到: {output_file}")
    
    def handle_low_similarity_images(self, results: List[Dict], threshold: float, 
                                   to_delete_dir: str) -> List[Dict]:
        """
        处理低相似度图片，移动到待删除文件夹
        
        Args:
            results: 结果列表
            threshold: 相似度阈值
            to_delete_dir: 待删除文件夹路径
            
        Returns:
            被移动的图片信息列表
        """
        os.makedirs(to_delete_dir, exist_ok=True)
        moved_files = []
        
        for result in results:
            if result['similarity'] < threshold:
                src_path = result['image_path']
                dst_path = os.path.join(to_delete_dir, os.path.basename(src_path))
                
                try:
                    shutil.move(src_path, dst_path)
                    result['moved_to'] = dst_path
                    moved_files.append(result)
                    print(f"已移动低相似度图片: {src_path} -> {dst_path} (相似度: {result['similarity']:.4f})")
                except Exception as e:
                    print(f"移动文件失败 {src_path}: {e}")
        
        print(f"共移动了 {len(moved_files)} 个低相似度图片到 {to_delete_dir}")
        return moved_files


def main():
    parser = argparse.ArgumentParser(description='CLIP图片文本相似度计算工具')
    parser.add_argument('--output-dir', default='output', 
                       help='prompts文件所在的output目录 (默认: output)')
    parser.add_argument('--synthetic-dir', default='datasets/synthetic',
                       help='合成图片所在目录 (默认: datasets/synthetic)')
    parser.add_argument('--result-file', default='similarity_results.txt',
                       help='结果输出文件 (默认: similarity_results.txt)')
    parser.add_argument('--del', type=float, dest='delete_threshold',
                       help='相似度阈值，低于此值的图片将被移动到待删除文件夹')
    parser.add_argument('--to-delete-dir', default='to_delete',
                       help='待删除图片存放目录 (默认: to_delete)')
    parser.add_argument('--model', default='openai/clip-vit-large-patch14',
                       help='CLIP模型名称 (默认: openai/clip-vit-large-patch14)')
    
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.exists(args.output_dir):
        print(f"错误: output目录不存在: {args.output_dir}")
        return
    
    if not os.path.exists(args.synthetic_dir):
        print(f"错误: synthetic目录不存在: {args.synthetic_dir}")
        return
    
    # 初始化计算器
    calculator = CLIPSimilarityCalculator(args.model)
    
    # 处理所有文件
    print("开始处理所有prompts文件...")
    results = calculator.process_all_files(args.output_dir, args.synthetic_dir)
    
    if not results:
        print("没有找到任何匹配的图片和文本对")
        return
    
    print(f"共处理了 {len(results)} 个图片-文本对")
    
    # 保存结果
    calculator.save_results(results, args.result_file)
    
    # 处理低相似度图片
    if args.delete_threshold is not None:
        print(f"\n处理相似度低于 {args.delete_threshold} 的图片...")
        moved_files = calculator.handle_low_similarity_images(
            results, args.delete_threshold, args.to_delete_dir
        )
        
        # 在结果文件中记录被移动的文件
        if moved_files:
            with open(args.result_file, 'a', encoding='utf-8') as f:
                f.write(f"\n\n移动到待删除文件夹的图片 (相似度 < {args.delete_threshold}):\n")
                f.write("=" * 50 + "\n")
                for moved in moved_files:
                    f.write(f"{moved['image_name']}\t{moved['similarity']:.4f}\t{moved['moved_to']}\n")
    
    # 输出统计信息
    similarities = [r['similarity'] for r in results]
    print(f"\n统计信息:")
    print(f"  平均相似度: {np.mean(similarities):.4f}")
    print(f"  最高相似度: {np.max(similarities):.4f}")
    print(f"  最低相似度: {np.min(similarities):.4f}")
    print(f"  标准差: {np.std(similarities):.4f}")
    
    if args.delete_threshold is not None:
        low_sim_count = len([s for s in similarities if s < args.delete_threshold])
        print(f"  低于阈值 {args.delete_threshold} 的图片数量: {low_sim_count}")


if __name__ == '__main__':
    main() 