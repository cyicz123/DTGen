#!/usr/bin/env python3
"""
SAM餐具分割工具

这个工具使用SAM (Segment Anything Model) 来分割餐具图片。
支持两种模式：
1. 命令行模式：处理指定的单个或多个图片
2. 交互模式：处理预设的训练数据集

使用方法：

1. 命令行模式 - 处理单个图片：
   python scripts/sam.py image.jpg

2. 命令行模式 - 处理多个图片：
   python scripts/sam.py image1.jpg image2.jpg image3.jpg

3. 指定输出目录：
   python scripts/sam.py image.jpg -o output_folder
   python scripts/sam.py image.jpg --output output_folder

4. 设置面积过滤阈值：
   python scripts/sam.py image.jpg -m 0.05
   python scripts/sam.py image.jpg --min-area-ratio 0.05

5. 交互模式（原有功能）：
   python scripts/sam.py -i
   python scripts/sam.py --interactive

6. 组合使用：
   python scripts/sam.py img1.jpg img2.jpg -o results -m 0.03

交互模式详细说明：
- 交互模式会处理预设的训练数据集目录
- 支持选择处理 cleaned 文件夹、dirty 文件夹或两者
- 输入选项：
  * 1: 仅处理 datasets/train/cleaned 目录
  * 2: 仅处理 datasets/train/dirty 目录  
  * 3: 处理两个目录
- 输出目录：
  * cleaned 图片 → datasets/plate/cleaned/
  * dirty 图片 → datasets/plate/dirty/
- 交互操作：
  * 用鼠标拖拽框选餐具区域
  * 按空格键确认选择
  * 按 'r' 键重置选择
  * 按 'q' 键跳过当前图片
- 如果输出文件已存在，会询问是否跳过

功能特点：
- 支持掩码空洞填充，确保餐具分割完整
- 可设置面积比例阈值过滤小的分割区域
- 输出透明背景的PNG图片
- 支持批量处理

注意事项：
- 需要预先下载SAM模型文件到 models/sam_vit_h_4b8939.pth
- 处理时需要手动框选餐具区域
- 输出文件名格式：原文件名.png（命令行模式）
- 交互模式保持原有的文件名格式
"""

import os
import numpy as np
from PIL import Image
import cv2
from segment_anything import SamPredictor, sam_model_registry
import argparse
import sys

# 全局变量用于鼠标事件
drawing = False
start_point = None
end_point = None
bbox = None

def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数，用于框选区域"""
    global drawing, start_point, end_point, bbox
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        # 计算边界框 (x1, y1, x2, y2)
        x1, y1 = start_point
        x2, y2 = end_point
        bbox = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

def get_user_bbox(image_path):
    """显示图片并让用户框选餐具区域"""
    global bbox, drawing, start_point, end_point
    
    # 重置全局变量
    bbox = None
    drawing = False
    start_point = None
    end_point = None
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image {image_path}")
        return None
    
    # 获取图片尺寸并调整显示大小
    height, width = image.shape[:2]
    max_size = 800
    if width > max_size or height > max_size:
        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        display_image = cv2.resize(image, (new_width, new_height))
        scale_factor = scale
    else:
        display_image = image.copy()
        scale_factor = 1.0
    
    # 创建窗口
    window_name = "Select Plate Area - Draw rectangle around the plate"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print(f"Processing: {os.path.basename(image_path)}")
    print("Instructions:")
    print("- Click and drag to draw a rectangle around the plate")
    print("- Press SPACE to confirm selection")
    print("- Press 'r' to reset selection")
    print("- Press 'q' to quit")
    
    while True:
        # 创建显示图片的副本
        img_display = display_image.copy()
        
        # 如果正在绘制，显示临时矩形
        if drawing and start_point and end_point:
            cv2.rectangle(img_display, start_point, end_point, (0, 255, 0), 2)
        
        # 如果已经有确定的边界框，显示它
        if bbox and not drawing:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 添加文本提示
            cv2.putText(img_display, "Press SPACE to confirm, 'r' to reset", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(window_name, img_display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # 空格键确认
            if bbox:
                break
            else:
                print("Please draw a rectangle first!")
        elif key == ord('r'):  # 'r'键重置
            bbox = None
            drawing = False
            start_point = None
            end_point = None
            print("Selection reset. Please draw a new rectangle.")
        elif key == ord('q'):  # 'q'键退出
            cv2.destroyAllWindows()
            return None
    
    cv2.destroyAllWindows()
    
    if bbox:
        # 将显示坐标转换回原始图片坐标
        x1, y1, x2, y2 = bbox
        original_bbox = (
            int(x1 / scale_factor),
            int(y1 / scale_factor),
            int(x2 / scale_factor),
            int(y2 / scale_factor)
        )
        return original_bbox
    
    return None

def create_output_dirs():
    """创建输出目录"""
    os.makedirs("datasets/plate/cleaned", exist_ok=True)
    os.makedirs("datasets/plate/dirty", exist_ok=True)

def load_sam_model():
    """加载SAM模型"""
    sam = sam_model_registry["vit_h"](checkpoint="models/sam_vit_h_4b8939.pth")
    predictor = SamPredictor(sam)
    return predictor

def segment_plate_with_bbox(predictor, image_path, bbox, min_area_ratio=0.01):
    """使用边界框提示进行SAM分割"""
    # 读取图片
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 设置图片到预测器
    predictor.set_image(image_rgb)
    
    # 转换边界框格式为SAM需要的格式 [x1, y1, x2, y2]
    box_prompt = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
    
    # 进行预测
    masks, scores, logits = predictor.predict(
        box=box_prompt,
        multimask_output=False  # 使用单一mask，因为边界框提示通常比较准确
    )
    
    # 取第一个mask
    mask = masks[0]
    
    # 过滤小面积掩码
    filtered_mask = filter_mask_by_area(mask, image_rgb.shape, min_area_ratio)
    if filtered_mask is None:
        return None, image_rgb
    
    # 填充掩码空洞
    filled_mask = fill_mask_holes(filtered_mask)
    
    return filled_mask, image_rgb

def extract_plate_from_mask(image, mask):
    """从mask中提取餐具区域"""
    # 创建3通道的mask
    mask_3channel = np.stack([mask] * 3, axis=-1)
    
    # 应用mask到图片
    segmented_image = image * mask_3channel
    
    # 找到mask的边界框
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return None
    
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    # 裁剪到边界框
    cropped_image = segmented_image[y_min:y_max+1, x_min:x_max+1]
    cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]
    
    # 创建透明背景的图片
    result = np.zeros((cropped_image.shape[0], cropped_image.shape[1], 4), dtype=np.uint8)
    result[:, :, :3] = cropped_image
    result[:, :, 3] = cropped_mask * 255  # alpha通道
    
    return result

def fill_mask_holes(mask):
    """填充掩码中的空洞，使其成为实心掩码"""
    # 转换为uint8格式
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # 使用形态学操作填充空洞
    kernel = np.ones((3, 3), np.uint8)
    
    # 闭运算：先膨胀后腐蚀，用于填充小的空洞
    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 使用轮廓填充来处理更大的空洞
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建填充后的掩码
    filled_mask = np.zeros_like(mask_uint8)
    
    # 填充所有外部轮廓
    for contour in contours:
        cv2.fillPoly(filled_mask, [contour], 255)
    
    # 转换回布尔类型
    return (filled_mask > 0).astype(bool)

def filter_mask_by_area(mask, image_shape, min_area_ratio):
    """根据面积比例过滤掩码"""
    total_pixels = image_shape[0] * image_shape[1]
    mask_area = np.sum(mask)
    area_ratio = mask_area / total_pixels
    
    if area_ratio < min_area_ratio:
        print(f"掩码面积比例 {area_ratio:.4f} 小于阈值 {min_area_ratio:.4f}，已过滤")
        return None
    
    print(f"掩码面积比例: {area_ratio:.4f}")
    return mask

def process_images(predictor, input_dir, output_dir, min_area_ratio=0.01):
    """处理指定目录下的所有图片"""
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()  # 排序以便按顺序处理
    
    print(f"\nFound {len(image_files)} images in {input_dir}")
    
    for i, image_file in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] Processing: {image_file}")
        
        image_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file.replace('.jpg', '.png'))
        
        # 检查输出文件是否已存在
        if os.path.exists(output_path):
            print(f"Output file already exists: {output_path}")
            user_input = input("Skip this image? (y/n): ").lower()
            if user_input == 'y':
                continue
        
        try:
            # 获取用户框选的边界框
            bbox = get_user_bbox(image_path)
            
            if bbox is None:
                print(f"Skipping {image_file} - no selection made")
                continue
            
            print(f"Selected bbox: {bbox}")
            
            # 使用边界框进行分割
            mask, image = segment_plate_with_bbox(predictor, image_path, bbox, min_area_ratio)
            
            if mask is None:
                print(f"Skipping {image_file} - mask filtered out")
                continue
            
            # 提取餐具区域
            plate_image = extract_plate_from_mask(image, mask)
            
            if plate_image is not None:
                # 保存结果
                Image.fromarray(plate_image, 'RGBA').save(output_path)
                print(f"Successfully saved: {output_path}")
            else:
                print(f"Warning: Could not extract plate from {image_file}")
                
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue

def process_single_images(predictor, image_paths, output_dir, min_area_ratio=0.01):
    """处理命令行指定的单个或多个图片"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nProcessing {len(image_paths)} images...")
    
    for i, image_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] Processing: {image_path}")
        
        # 检查输入文件是否存在
        if not os.path.exists(image_path):
            print(f"Error: Input file not found: {image_path}")
            continue
        
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.png")
        
        # 检查输出文件是否已存在
        if os.path.exists(output_path):
            print(f"Output file already exists: {output_path}")
            user_input = input("Overwrite? (y/n): ").lower()
            if user_input != 'y':
                continue
        
        try:
            # 获取用户框选的边界框
            bbox = get_user_bbox(image_path)
            
            if bbox is None:
                print(f"Skipping {image_path} - no selection made")
                continue
            
            print(f"Selected bbox: {bbox}")
            
            # 使用边界框进行分割
            mask, image = segment_plate_with_bbox(predictor, image_path, bbox, min_area_ratio)
            
            if mask is None:
                print(f"Skipping {image_path} - mask filtered out")
                continue
            
            # 提取餐具区域
            plate_image = extract_plate_from_mask(image, mask)
            
            if plate_image is not None:
                # 保存结果
                Image.fromarray(plate_image, 'RGBA').save(output_path)
                print(f"Successfully saved: {output_path}")
            else:
                print(f"Warning: Could not extract plate from {image_path}")
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue

def main():
    """主函数"""
    args = parse_arguments()
    
    # 加载SAM模型
    print("Loading SAM model...")
    predictor = load_sam_model()
    print("SAM model loaded successfully!")
    
    if args.interactive:
        # 交互模式：处理预设的训练数据集
        print("\n=== Interactive Mode: Plate Segmentation Tool ===")
        print("This tool uses SAM to segment plates from images.")
        print("You will be asked to draw rectangles around plates for each image.")
        
        # 创建输出目录
        create_output_dirs()
        
        # 询问用户要处理哪个文件夹
        print("\nWhich folder would you like to process?")
        print("1. cleaned folder only")
        print("2. dirty folder only") 
        print("3. both folders")
        
        choice = input("Enter your choice (1/2/3): ").strip()
        
        if choice == '1':
            print("\nProcessing cleaned folder...")
            process_images(predictor, "datasets/train/cleaned", "datasets/plate/cleaned", args.min_area_ratio)
        elif choice == '2':
            print("\nProcessing dirty folder...")
            process_images(predictor, "datasets/train/dirty", "datasets/plate/dirty", args.min_area_ratio)
        elif choice == '3':
            print("\nProcessing cleaned folder...")
            process_images(predictor, "datasets/train/cleaned", "datasets/plate/cleaned", args.min_area_ratio)
            print("\nProcessing dirty folder...")
            process_images(predictor, "datasets/train/dirty", "datasets/plate/dirty", args.min_area_ratio)
        else:
            print("Invalid choice. Exiting.")
            return
    else:
        # 命令行模式：处理指定的图片
        print(f"\n=== Command Line Mode ===")
        print(f"Processing {len(args.images)} images...")
        print(f"Output directory: {args.output}")
        print(f"Min area ratio: {args.min_area_ratio}")
        
        process_single_images(predictor, args.images, args.output, args.min_area_ratio)
    
    print("\n=== Plate segmentation completed! ===")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用SAM模型分割餐具图片')
    parser.add_argument('images', nargs='*', help='输入图片路径，支持单个或多个图片')
    parser.add_argument('-o', '--output', type=str, default='output', 
                       help='输出目录 (默认: output)')
    parser.add_argument('-m', '--min-area-ratio', type=float, default=0.01,
                       help='最小面积比例阈值，过滤小于此比例的分割掩码 (默认: 0.01)')
    parser.add_argument('-i', '--interactive', action='store_true',
                       help='使用交互模式，处理预设的训练数据集')
    
    args = parser.parse_args()
    
    # 验证参数
    if not args.interactive and not args.images:
        print("错误：请指定输入图片或使用 --interactive 模式")
        parser.print_help()
        sys.exit(1)
    
    return args

if __name__ == "__main__":
    main()