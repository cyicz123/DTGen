#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from pathlib import Path

def find_txt_folders(root_path):
    """递归查找包含txt文件的文件夹"""
    txt_folders = []
    
    for root, dirs, files in os.walk(root_path):
        # 检查当前文件夹是否包含txt文件
        txt_files = [f for f in files if f.endswith('.txt')]
        if txt_files:
            txt_folders.append((root, txt_files))
    
    return txt_folders

def process_txt_folder(folder_path, txt_files):
    """处理单个包含txt文件的文件夹"""
    entries = []
    
    # 获取上级目录名称（例如：clean）
    parent_dir_name = os.path.basename(folder_path)
    # 获取上上级目录名称（例如：bowl）
    grandparent_dir_name = os.path.basename(os.path.dirname(folder_path))
    
    for txt_file in txt_files:
        file_path = os.path.join(folder_path, txt_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # 获取不带扩展名的文件名
            file_name = os.path.splitext(txt_file)[0]
            
            # 创建包含上上级和上级目录名称的name字段
            full_name = f"{grandparent_dir_name}_{file_name}"
            
            # 创建条目
            entry = f"name: {full_name}\n\npositive:{content}\n\nnegative:"
            entries.append(entry)
            
        except Exception as e:
            print(f"警告：读取文件 {file_path} 时出错: {e}")
            continue
    
    return entries

def generate_output_file(folder_path, entries):
    """为文件夹生成输出文件"""
    if not entries:
        return
    
    # 获取上级目录名称（例如：clean）
    parent_dir_name = os.path.basename(folder_path)
    # 获取上上级目录名称（例如：bowl）
    grandparent_dir_name = os.path.basename(os.path.dirname(folder_path))
    
    # 使用上上级和上级目录名作为输出文件名
    output_filename = f"{grandparent_dir_name}_{parent_dir_name}_prompts.txt"
    output_path = os.path.join(folder_path, output_filename)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # 用三个破折号分割每个条目
            f.write('\n---\n'.join(entries))
        
        print(f"已生成输出文件: {output_path}")
        
    except Exception as e:
        print(f"错误：创建输出文件 {output_path} 时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='递归查找包含txt文件的文件夹并生成格式化输出')
    parser.add_argument('folder_path', help='要搜索的根文件夹路径')
    
    args = parser.parse_args()
    
    root_path = args.folder_path
    
    # 检查输入路径是否存在
    if not os.path.exists(root_path):
        print(f"错误：路径 '{root_path}' 不存在")
        sys.exit(1)
    
    if not os.path.isdir(root_path):
        print(f"错误：'{root_path}' 不是一个文件夹")
        sys.exit(1)
    
    print(f"开始搜索路径: {root_path}")
    
    # 查找包含txt文件的文件夹
    txt_folders = find_txt_folders(root_path)
    
    if not txt_folders:
        print("未找到包含txt文件的文件夹")
        return
    
    print(f"找到 {len(txt_folders)} 个包含txt文件的文件夹")
    
    # 处理每个文件夹
    for folder_path, txt_files in txt_folders:
        print(f"处理文件夹: {folder_path}")
        print(f"  包含 {len(txt_files)} 个txt文件")
        
        # 处理文件夹中的txt文件
        entries = process_txt_folder(folder_path, txt_files)
        
        # 生成输出文件
        generate_output_file(folder_path, entries)
    
    print("处理完成！")

if __name__ == "__main__":
    main() 