#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Bilibili视频下载功能
"""

import os
import sys
import argparse

from backend.bilidown import download_video


def main():
    """测试主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试Bilibili视频下载功能')
    parser.add_argument('--url', default="https://www.bilibili.com/video/BV1e2FUzLE7M/", help='测试视频URL')
    parser.add_argument('--output', default="./test_output", help='输出目录')
    parser.add_argument('--quality', type=int, default=80, help='视频质量 (默认: 80, 对应1080P)')
    parser.add_argument('--audio-quality', default='medium', choices=['high', 'medium', 'low'], help='音频质量 (默认: high)')
    parser.add_argument('--sessdata', help='SESSDATA cookie (可选，用于登录状态)')
    
    args = parser.parse_args()
    
    print(f"测试下载视频: {args.url}")
    print(f"输出目录: {args.output}")
    print(f"视频质量: {args.quality}")
    print(f"音频质量: {args.audio_quality}")
    
    try:
        # 创建输出目录
        os.makedirs(args.output, exist_ok=True)
        
        # 下载视频
        download_video(args.url, args.output, args.quality, args.audio_quality, args.sessdata)
        
        print("测试成功！")
        
    except Exception as e:
        print(f"测试失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
