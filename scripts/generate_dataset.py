#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
餐具文生图数据集生成脚本
根据配置文件生成大约36000条prompts
"""

import yaml
import random
import os
import argparse
from pathlib import Path

def load_yaml_config(file_path):
    """加载YAML配置文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def generate_prompts(tableware_file, dirtiness_file):
    """生成餐具文生图数据集"""
    
    # 加载配置文件
    tableware_config = load_yaml_config(tableware_file)
    dirtiness_config = load_yaml_config(dirtiness_file)
    
    # 定义脏污程度与数值的映射
    dirtiness_mapping = {
        'slightly_dirty': 0.25,
        'moderately_dirty': 0.6,
        'heavily_dirty': 1.0
    }
    
    # 准备输出列表
    prompts = []
    
    # 处理plate和bowl
    for tableware_type in ['plate', 'bowl']:
        # 随机选择30条餐具描述
        tableware_descriptions = tableware_config[tableware_type]
        selected_descriptions = random.sample(tableware_descriptions, min(30, len(tableware_descriptions)))
        
        # 对每个选中的餐具描述，与所有脏污程度组合
        for description in selected_descriptions:
            # 清理描述，去掉中文注释
            clean_description = description.split('#')[0].strip()
            
            # 与每种脏污程度组合
            for dirtiness_level, quantitative_value in dirtiness_mapping.items():
                dirtiness_descriptions = dirtiness_config[tableware_type][dirtiness_level]
                
                # 与该脏污程度下的所有描述组合
                for dirtiness_desc in dirtiness_descriptions:
                    # 清理脏污描述，去掉中文注释
                    clean_dirtiness_desc = dirtiness_desc.split('#')[0].strip()
                    
                    # 生成prompt
                    prompt = f"drt_tableware, {clean_description}, {clean_dirtiness_desc}, photorealistic, top-down view, isolated on chromakey green, studio lighting, sharp focus. {quantitative_value}"
                    prompts.append(prompt)
    
    return prompts

def save_prompts_to_file(prompts, output_file='prompts.txt'):
    """将生成的prompts保存到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for prompt in prompts:
            f.write(prompt + '\n')

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='生成餐具文生图数据集')
    
    parser.add_argument('-t', '--tableware', 
                        default='prompts/tableware_description.yaml',
                        help='餐具描述配置文件路径 (默认: prompts/tableware_description.yaml)')
    
    parser.add_argument('-d', '--dirtiness', 
                        default='prompts/dirtiness_description.yaml',
                        help='脏污程度描述配置文件路径 (默认: prompts/dirtiness_description.yaml)')
    
    parser.add_argument('-o', '--output', 
                        default='prompts/prompts.txt',
                        help='输出文件路径 (默认: prompts/prompts.txt)')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    print("开始生成餐具文生图数据集...")
    print(f"餐具描述文件: {args.tableware}")
    print(f"脏污描述文件: {args.dirtiness}")
    print(f"输出文件: {args.output}")
    
    # 检查配置文件是否存在
    if not os.path.exists(args.tableware):
        print(f"错误：找不到餐具描述文件 {args.tableware}")
        return
    
    if not os.path.exists(args.dirtiness):
        print(f"错误：找不到脏污描述文件 {args.dirtiness}")
        return
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 设置随机种子以确保结果可复现
    random.seed(args.seed)
    
    # 生成prompts
    prompts = generate_prompts(args.tableware, args.dirtiness)
    
    # 保存到文件
    save_prompts_to_file(prompts, args.output)
    
    print(f"数据集生成完成！")
    print(f"总共生成了 {len(prompts)} 条prompts")
    print(f"已保存到 {args.output} 文件")
    
    # 显示一些统计信息
    plate_count = sum(1 for p in prompts if ', a ' in p and any(plate_word in p for plate_word in ['plate', 'platter', 'dish']))
    bowl_count = len(prompts) - plate_count
    
    print(f"\n统计信息:")
    print(f"- Plate相关prompts: {plate_count}")
    print(f"- Bowl相关prompts: {bowl_count}")
    print(f"- 每种餐具类型选择了30条描述")
    print(f"- 每条描述与3种脏污程度组合")
    print(f"- 预计总数: 30 × 2 × 3 × (各脏污程度描述数量)")
    
    # 显示前几个示例
    print(f"\n前5个示例:")
    for i, prompt in enumerate(prompts[:5]):
        print(f"{i+1}. {prompt}")

if __name__ == "__main__":
    main() 