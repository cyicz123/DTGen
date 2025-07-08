#!/usr/bin/env python3
"""
文生图脚本

使用 Stable Diffusion 3.5 进行文本到图像的生成。
支持完整的命令行参数配置。

使用方法：
    python scripts/text_to_image.py "一只可爱的猫咪坐在花园里"
    python scripts/text_to_image.py "现代都市夜景" --width 1344 --height 768 --steps 50
    python scripts/text_to_image.py "山水画" --model stabilityai/stable-diffusion-3.5-large-turbo --guidance-scale 1.0 --steps 4
    python scripts/text_to_image.py "科幻场景" --negative-prompt "模糊,低质量" --seed 42 --batch-size 2

参数说明：
    - prompt: 文本提示词（位置参数）
    - --model: 模型名称，默认为 stabilityai/stable-diffusion-3.5-large
    - --negative-prompt: 负面提示词
    - --width: 图像宽度，默认1024
    - --height: 图像高度，默认1024
    - --guidance-scale: 引导比例，默认4.5
    - --steps: 推理步数，影响图像质量和生成时间 (默认: 40)
    - --seed: 随机种子，默认随机
    - --batch-size: 批次大小，一次生成多少张图像 (默认: 1)
    - --output-dir: 输出目录，默认output
    - --filename: 输出文件名，默认自动生成
    - --device: 设备，默认cuda
    - --dtype: 数据类型，默认bfloat16
    - --enable-cpu-offload: 启用CPU卸载以节省显存
    - --enable-attention-slicing: 启用注意力切片以节省显存
    - --enable-vae-slicing: 启用VAE切片以节省显存
    - --max-sequence-length: 最大序列长度，默认512
    - --verbose: 详细输出
"""

import argparse
import sys
import time
import torch
from datetime import datetime
from pathlib import Path
from PIL import Image
import hashlib
import json

def setup_args():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(
        description="使用 Stable Diffusion 3.5 进行文生图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  %(prog)s "一只可爱的猫咪坐在花园里"
  %(prog)s "现代都市夜景" --width 1344 --height 768 --steps 50
  %(prog)s "山水画" --model stabilityai/stable-diffusion-3.5-large-turbo --guidance-scale 1.0 --steps 4
  %(prog)s "科幻场景" --negative-prompt "模糊,低质量" --seed 42 --batch-size 2
        """
    )
    
    # 位置参数：提示词
    parser.add_argument(
        "prompt",
        type=str,
        help="文本提示词（描述您想要生成的图像）"
    )
    
    # 模型相关参数
    parser.add_argument(
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-3.5-large",
        help="模型名称 (默认: stabilityai/stable-diffusion-3.5-large)"
    )
    
    # 生成参数
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="负面提示词（描述不想要的内容）"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="图像宽度 (默认: 1024)"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="图像高度 (默认: 1024)"
    )
    
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.5,
        help="引导比例，控制对提示词的遵循程度 (默认: 4.5)"
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=40,
        help="推理步数，影响图像质量和生成时间 (默认: 40)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子，用于生成可重现的结果 (默认: 随机)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="批次大小，一次生成多少张图像 (默认: 1)"
    )
    
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=512,
        help="最大序列长度 (默认: 512)"
    )
    
    # 输出参数
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="输出目录 (默认: output)"
    )
    
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="输出文件名（不包含扩展名），默认自动生成"
    )
    
    # 设备和优化参数
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="计算设备 (默认: cuda)"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="数据类型 (默认: bfloat16)"
    )
    
    parser.add_argument(
        "--enable-cpu-offload",
        action="store_true",
        help="启用CPU卸载以节省显存"
    )
    
    parser.add_argument(
        "--enable-attention-slicing",
        action="store_true",
        help="启用注意力切片以节省显存"
    )
    
    parser.add_argument(
        "--enable-vae-slicing",
        action="store_true",
        help="启用VAE切片以节省显存"
    )
    
    # 其他参数
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出"
    )
    
    parser.add_argument(
        "--save-metadata",
        action="store_true",
        help="保存生成参数到JSON文件"
    )
    
    return parser.parse_args()

def get_torch_dtype(dtype_str):
    """转换数据类型字符串为torch dtype"""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    return dtype_map.get(dtype_str, torch.bfloat16)

def generate_filename(prompt, args):
    """生成输出文件名"""
    if args.filename:
        return args.filename
    
    # 使用提示词的前30个字符作为文件名
    clean_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
    clean_prompt = clean_prompt.replace(' ', '_')
    
    # 添加时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 添加参数哈希（用于区分不同参数的生成）
    param_str = f"{args.width}x{args.height}_{args.guidance_scale}_{args.steps}"
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
    
    filename = f"{clean_prompt}_{timestamp}_{param_hash}"
    return filename

def save_metadata(filepath, args, generation_info):
    """保存生成参数到JSON文件"""
    metadata = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "model": args.model,
        "width": args.width,
        "height": args.height,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.steps,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_sequence_length": args.max_sequence_length,
        "device": args.device,
        "dtype": args.dtype,
        "generation_time": generation_info.get("generation_time", 0),
        "timestamp": datetime.now().isoformat()
    }
    
    metadata_path = filepath.with_suffix('.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    return metadata_path

def main():
    """主函数"""
    args = setup_args()
    
    # 验证参数
    if args.device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA不可用，切换到CPU")
        args.device = "cpu"
    
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("警告: MPS不可用，切换到CPU")
        args.device = "cpu"
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.verbose:
        print(f"输出目录: {output_dir.absolute()}")
        print(f"使用设备: {args.device}")
        print(f"数据类型: {args.dtype}")
        print(f"模型: {args.model}")
        print(f"提示词: {args.prompt}")
        if args.negative_prompt:
            print(f"负面提示词: {args.negative_prompt}")
        print(f"图像尺寸: {args.width}x{args.height}")
        print(f"引导比例: {args.guidance_scale}")
        print(f"推理步数: {args.steps}")
        print(f"批次大小: {args.batch_size}")
        print("-" * 50)
    
    try:
        # 导入diffusers
        from diffusers import StableDiffusion3Pipeline
        
        print("正在加载模型...")
        start_time = time.time()
        
        # 加载模型
        torch_dtype = get_torch_dtype(args.dtype)
        pipe = StableDiffusion3Pipeline.from_pretrained(
            args.model,
            torch_dtype=torch_dtype
        )
        
        # 设置设备
        if args.device != "cpu":
            pipe = pipe.to(args.device)
        
        # 应用优化
        if args.enable_cpu_offload:
            pipe.enable_model_cpu_offload()
            if args.verbose:
                print("已启用CPU卸载")
        
        if args.enable_attention_slicing:
            pipe.enable_attention_slicing()
            if args.verbose:
                print("已启用注意力切片")
        
        if args.enable_vae_slicing:
            pipe.enable_vae_slicing()
            if args.verbose:
                print("已启用VAE切片")
        
        load_time = time.time() - start_time
        print(f"模型加载完成，耗时: {load_time:.2f}秒")
        
        # 设置随机种子
        if args.seed is not None:
            torch.manual_seed(args.seed)
            if args.verbose:
                print(f"设置随机种子: {args.seed}")
        
        # 生成图像
        print("正在生成图像...")
        generation_start = time.time()
        
        # 准备生成参数
        generation_kwargs = {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt if args.negative_prompt else None,
            "width": args.width,
            "height": args.height,
            "guidance_scale": args.guidance_scale,
            "num_inference_steps": args.steps,
            "num_images_per_prompt": args.batch_size,
            "max_sequence_length": args.max_sequence_length,
        }
        
        # 过滤None值
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
        
        # 生成图像
        result = pipe(**generation_kwargs)
        images = result.images
        
        generation_time = time.time() - generation_start
        print(f"图像生成完成，耗时: {generation_time:.2f}秒")
        
        # 保存图像
        saved_files = []
        generation_info = {"generation_time": generation_time}
        
        for i, image in enumerate(images):
            # 生成文件名
            if args.batch_size == 1:
                filename = generate_filename(args.prompt, args)
            else:
                base_filename = generate_filename(args.prompt, args)
                filename = f"{base_filename}_{i+1:02d}"
            
            # 保存图像
            image_path = output_dir / f"{filename}.png"
            image.save(image_path)
            saved_files.append(image_path)
            
            print(f"已保存: {image_path}")
            
            # 保存元数据
            if args.save_metadata:
                metadata_path = save_metadata(image_path, args, generation_info)
                if args.verbose:
                    print(f"已保存元数据: {metadata_path}")
        
        print(f"\n成功生成 {len(images)} 张图像")
        print(f"总耗时: {time.time() - start_time:.2f}秒")
        
        # 显示保存的文件
        if args.verbose:
            print("\n保存的文件:")
            for file_path in saved_files:
                print(f"  {file_path}")
        
    except ImportError as e:
        print(f"错误: 缺少依赖库 - {e}")
        print("请安装必要的依赖:")
        print("pip install diffusers torch torchvision transformers accelerate")
        sys.exit(1)
    
    except Exception as e:
        print(f"错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 