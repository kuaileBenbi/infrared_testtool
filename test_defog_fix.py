#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
透雾算法修复测试脚本 - 测试灰度图像处理
"""

import cv2
import numpy as np
from preprocessor import ImagePreprocessor

def test_defog_with_grayscale():
    """测试透雾算法处理灰度图像"""
    # 创建预处理器实例
    preprocessor = ImagePreprocessor()
    
    # 创建灰度测试图像
    height, width = 240, 320
    gray_image = np.zeros((height, width), dtype=np.uint8)
    
    # 创建渐变背景
    for i in range(height):
        for j in range(width):
            gray_image[i, j] = int(255 * (i + j) / (height + width))
    
    # 添加噪声模拟雾气效果
    noise = np.random.normal(0, 20, gray_image.shape).astype(np.int16)
    gray_image = np.clip(gray_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    print(f"输入图像形状: {gray_image.shape}")
    print(f"输入图像类型: {gray_image.dtype}")
    
    try:
        # 应用透雾算法
        result = preprocessor.apply_defog(gray_image)
        
        print(f"输出图像形状: {result.shape}")
        print(f"输出图像类型: {result.dtype}")
        print("灰度图像透雾处理测试成功！")
        
        # 保存结果用于验证
        cv2.imwrite("test_gray_input.jpg", gray_image)
        cv2.imwrite("test_gray_defog_result.jpg", result)
        print("已保存测试图像")
        
    except Exception as e:
        print(f"灰度图像透雾处理测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_defog_with_color():
    """测试透雾算法处理彩色图像"""
    # 创建预处理器实例
    preprocessor = ImagePreprocessor()
    
    # 创建彩色测试图像
    height, width = 240, 320
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 创建渐变背景
    for i in range(height):
        for j in range(width):
            color_image[i, j] = [
                int(255 * j / width),  # 红色通道
                int(255 * i / height),  # 绿色通道
                int(255 * (i + j) / (height + width))  # 蓝色通道
            ]
    
    # 添加噪声模拟雾气效果
    noise = np.random.normal(0, 20, color_image.shape).astype(np.int16)
    color_image = np.clip(color_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    print(f"输入彩色图像形状: {color_image.shape}")
    print(f"输入彩色图像类型: {color_image.dtype}")
    
    try:
        # 应用透雾算法
        result = preprocessor.apply_defog(color_image)
        
        print(f"输出彩色图像形状: {result.shape}")
        print(f"输出彩色图像类型: {result.dtype}")
        print("彩色图像透雾处理测试成功！")
        
        # 保存结果用于验证
        cv2.imwrite("test_color_input.jpg", color_image)
        cv2.imwrite("test_color_defog_result.jpg", result)
        print("已保存测试图像")
        
    except Exception as e:
        print(f"彩色图像透雾处理测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=== 透雾算法修复测试 ===")
    print("\n1. 测试灰度图像处理:")
    test_defog_with_grayscale()
    
    print("\n2. 测试彩色图像处理:")
    test_defog_with_color()
    
    print("\n测试完成！")
