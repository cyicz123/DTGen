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

def generate_prompts(tableware_file, dirtiness_file, background_file, tableware_type, dirtiness_level, num_prompts):
    """生成餐具文生图数据集"""
    
    # 加载配置文件
    tableware_config = load_yaml_config(tableware_file)
    dirtiness_config = load_yaml_config(dirtiness_file)
    background_config = load_yaml_config(background_file)
    
    # 定义脏污程度与数值的映射
    dirtiness_mapping = {
        'slightly_dirty': 0.25,
        'moderately_dirty': 0.6,
        'heavily_dirty': 1.0
    }
    
    # 准备输出列表
    prompts = []
    
    # 获取对应的描述列表
    tableware_descriptions = tableware_config[tableware_type]
    dirtiness_descriptions = dirtiness_config[tableware_type][dirtiness_level]
    background_descriptions = background_config['backgrounds']  # 假设背景配置在backgrounds键下
    quantitative_value = dirtiness_mapping[dirtiness_level]
    
    # 随机生成指定数量的prompts
    for i in range(num_prompts):
        # 随机选择描述
        tableware_desc = random.choice(tableware_descriptions)
        dirtiness_desc = random.choice(dirtiness_descriptions)
        background_desc = random.choice(background_descriptions)
        
        # 清理描述，去掉中文注释
        clean_description = tableware_desc.split('#')[0].strip()
        clean_dirtiness_desc = dirtiness_desc.split('#')[0].strip()
        clean_background_desc = background_desc.split('#')[0].strip()
        
        # 生成prompt
        prompt = f"drt_tableware, {clean_description}, {clean_dirtiness_desc}, photorealistic, top-down view, {clean_background_desc}"
        prompts.append((prompt, quantitative_value))
    
    return prompts

def save_prompts_to_files(prompts, tableware_type, dirtiness_level, output_dir, digits=5):
    """将生成的prompts保存到单独的文件中"""
    # 创建文件夹结构：output_dir/tableware_type/dirtiness_level/
    folder_path = os.path.join(output_dir, tableware_type, dirtiness_level)
    os.makedirs(folder_path, exist_ok=True)
    
    # 为每个prompt创建单独的txt文件
    for i, (prompt, quantitative_value) in enumerate(prompts):
        # 生成文件名：dirtiness_level_序号.txt
        filename = f"{dirtiness_level}_{str(i+1).zfill(digits)}.txt"
        file_path = os.path.join(folder_path, filename)
        
        # 保存prompt和量化值
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"{prompt}\n")
    
    return folder_path

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='生成餐具文生图数据集')
    
    parser.add_argument('-t', '--tableware', 
                        default='prompts/tableware_description.yaml',
                        help='餐具描述配置文件路径 (默认: prompts/tableware_description.yaml)')
    
    parser.add_argument('-d', '--dirtiness', 
                        default='prompts/dirtiness_description.yaml',
                        help='脏污程度描述配置文件路径 (默认: prompts/dirtiness_description.yaml)')
    
    parser.add_argument('-b', '--background', 
                        default='prompts/background.yaml',
                        help='背景描述配置文件路径 (默认: prompts/background.yaml)')
    
    parser.add_argument('--tableware-type', 
                        choices=['plate', 'bowl'],
                        required=True,
                        help='餐具类型 (plate 或 bowl)')
    
    parser.add_argument('--dirtiness-level', 
                        choices=['slightly_dirty', 'moderately_dirty', 'heavily_dirty'],
                        required=True,
                        help='脏污程度 (slightly_dirty, moderately_dirty, heavily_dirty)')
    
    parser.add_argument('-n', '--num-prompts', 
                        type=int,
                        required=True,
                        help='生成的prompt数量')
    
    parser.add_argument('-o', '--output', 
                        default='output',
                        help='输出根目录路径 (默认: output)')
    
    parser.add_argument('--digits', type=int, default=5,
                        help='序号位数 (默认: 5)')
    
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
    print(f"背景描述文件: {args.background}")
    print(f"餐具类型: {args.tableware_type}")
    print(f"脏污程度: {args.dirtiness_level}")
    print(f"生成数量: {args.num_prompts}")
    print(f"输出目录: {args.output}")
    print(f"序号位数: {args.digits}")
    
    # 检查配置文件是否存在
    if not os.path.exists(args.tableware):
        print(f"错误：找不到餐具描述文件 {args.tableware}")
        return
    
    if not os.path.exists(args.dirtiness):
        print(f"错误：找不到脏污描述文件 {args.dirtiness}")
        return
    
    if not os.path.exists(args.background):
        print(f"错误：找不到背景描述文件 {args.background}")
        return
    
    # 确保输出目录存在
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f"创建输出目录: {args.output}")
    
    # 设置随机种子以确保结果可复现
    random.seed(args.seed)
    
    # 生成prompts
    prompts = generate_prompts(
        args.tableware, 
        args.dirtiness, 
        args.background,
        args.tableware_type,
        args.dirtiness_level,
        args.num_prompts
    )
    
    # 保存到文件
    folder_path = save_prompts_to_files(
        prompts, 
        args.tableware_type, 
        args.dirtiness_level, 
        args.output, 
        args.digits
    )
    
    print(f"\n数据集生成完成！")
    print(f"总共生成了 {len(prompts)} 条prompts")
    print(f"已保存到 {folder_path} 文件夹")
    
    # 显示统计信息
    print(f"\n统计信息:")
    print(f"- 餐具类型: {args.tableware_type}")
    print(f"- 脏污程度: {args.dirtiness_level}")
    print(f"- 生成数量: {len(prompts)}")
    print(f"- 文件命名格式: {args.dirtiness_level}_{{序号:0{args.digits}d}}.txt")
    
    # 显示前几个示例
    print(f"\n前3个示例:")
    for i, (prompt, quantitative_value) in enumerate(prompts[:3]):
        filename = f"{args.dirtiness_level}_{str(i+1).zfill(args.digits)}.txt"
        print(f"{filename}: {prompt}. {quantitative_value}")

if __name__ == "__main__":
    main() 