#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bilibili视频下载工具
支持下载视频、音频和弹幕
"""

import os
import re
import json
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, parse_qs

import requests
from tqdm import tqdm

class BiliClient:
    def __init__(self, sessdata=None):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.bilibili.com/',
        }
        if sessdata:
            self.headers['Cookie'] = f'SESSDATA={sessdata}'
    
    def simple_get(self, url, params=None):
        """发送GET请求"""
        try:
            response = requests.get(url, headers=self.headers, params=params, stream=True)
            response.raise_for_status()
            return response
        except Exception as e:
            raise Exception(f"请求失败: {e}")
    
    def get_video_info(self, bvid):
        """获取视频信息"""
        url = "https://api.bilibili.com/x/web-interface/wbi/view"
        params = {"bvid": bvid}
        response = self.simple_get(url, params)
        data = response.json()
        if data['code'] != 0:
            raise Exception(f"获取视频信息失败: {data['message']}")
        return data['data']
    
    def get_play_info(self, bvid, cid):
        """获取播放信息"""
        url = "https://api.bilibili.com/x/player/playurl"
        params = {
            "bvid": bvid,
            "cid": cid,
            "fnval": "4048",
            "fnver": "0",
            "fourk": "1",
        }
        response = self.simple_get(url, params)
        data = response.json()
        if data['code'] != 0:
            raise Exception(f"获取播放信息失败: {data['message']}")
        return data['data']
    
    def get_danmaku(self, cid):
        """获取弹幕"""
        url = f"https://api.bilibili.com/x/v1/dm/list.so?oid={cid}"
        response = self.simple_get(url)
        return response.content

def extract_bvid(url):
    """从URL中提取BV号"""
    # 匹配BV号的正则表达式
    bv_pattern = r'BV[0-9A-Za-z]+'
    match = re.search(bv_pattern, url)
    if match:
        return match.group(0)
    raise Exception("无法从URL中提取BV号")

def get_video_url(medias, quality='auto', default_quality='80'):
    """获取视频URL"""
    # 按质量排序（id越大质量越高）
    sorted_medias = sorted(medias, key=lambda x: x['id'], reverse=True)
    
    # 处理 auto 选项
    if quality == "auto":
        # 优先选择codecid为12、7、13的视频
        for code in [12, 7, 13]:
            for item in sorted_medias:
                if item.get('codecid') == code:
                    return item['baseUrl']
        # 默认返回最高质量的视频
        return sorted_medias[0]['baseUrl']
    
    try:
        # 尝试将质量参数转换为整数
        quality_int = int(quality)
        
        # 优先选择指定质量的视频
        for item in sorted_medias:
            if item['id'] == quality_int:
                return item['baseUrl']
        
        # 如果没有找到指定质量，尝试使用默认质量
        try:
            default_quality_int = int(default_quality)
            # 返回质量不超过默认质量的最高质量视频
            for item in sorted_medias:
                if item['id'] <= default_quality_int:
                    return item['baseUrl']
        except (ValueError, TypeError):
            # 如果默认质量参数无法转换为整数，返回最高质量的视频
            pass
    except (ValueError, TypeError):
        # 如果质量参数无法转换为整数，返回最高质量的视频
        pass
    
    # 如果仍然没有找到，返回第一个视频
    if sorted_medias:
        return sorted_medias[0]['baseUrl']
    
    raise Exception("未找到对应视频分辨率格式")

def get_audio_url(dash, audio_quality='high', default_audio_quality='high'):
    """获取音频URL"""
    if audio_quality == 'high' and 'flac' in dash and dash['flac']:
        return dash['flac']['audio']['baseUrl']
    # 根据质量选择音频
    if not dash.get('audio'):
        return ""
    
    # 按音频质量排序（id越大质量越高）
    sorted_audios = sorted(dash['audio'], key=lambda x: x['id'], reverse=True)
    
    if audio_quality == 'high':
        # 选择最高质量的音频
        return sorted_audios[0]['baseUrl']
    elif audio_quality == 'medium':
        # 选择中等质量的音频
        return sorted_audios[min(1, len(sorted_audios) - 1)]['baseUrl']
    elif audio_quality == 'low':
        # 选择最低质量的音频
        return sorted_audios[-1]['baseUrl']
    
    # 如果没有找到指定质量，使用默认音频质量
    if default_audio_quality == 'high':
        return sorted_audios[0]['baseUrl']
    elif default_audio_quality == 'medium':
        return sorted_audios[min(1, len(sorted_audios) - 1)]['baseUrl']
    elif default_audio_quality == 'low':
        return sorted_audios[-1]['baseUrl']
    
    # 默认返回最高质量的音频
    return sorted_audios[0]['baseUrl']

def download_file(client, url, filepath, desc):
    """下载文件"""
    response = client.simple_get(url)
    total_size = int(response.headers.get('content-length', 0))
    
    pbar = tqdm(desc=desc, total=total_size, unit='B', unit_scale=True, unit_divisor=1024)
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    pbar.close()

def convert_xml_to_ass(xml_content):
    """将XML弹幕转换为ASS格式"""
    import xml.etree.ElementTree as ET
    
    # 解析XML
    root = ET.fromstring(xml_content)
    
    # ASS头部
    ass_header = """
[Script Info]
; Script generated by Bilidown
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,SimHei,32,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""".strip()
    
    # 转换弹幕
    events = []
    for d in root.findall('d'):
        # 获取弹幕属性
        p = d.get('p').split(',')
        if len(p) < 5:
            continue
        
        # 解析时间（秒）
        start_time = float(p[0])
        # 弹幕持续时间（秒）
        duration = float(p[1])
        end_time = start_time + duration
        
        # 转换时间格式为ASS格式（时:分:秒.厘秒）
        def format_time(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            cs = int((seconds % 1) * 100)
            return f"{h:02d}:{m:02d}:{s:02d}.{cs:02d}"
        
        # 弹幕文本
        text = d.text if d.text else ""
        
        # 构建ASS事件
        event = f"Dialogue: 0,{format_time(start_time)},{format_time(end_time)},Default,,0,0,0,,{text}"
        events.append(event)
    
    # 组合ASS内容
    ass_content = ass_header + '\n' + '\n'.join(events)
    return ass_content

def merge_media(output_path, video_path, audio_path):
    """合并音视频"""
    try:
        # 检查ffmpeg是否安装
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except subprocess.CalledProcessError:
        raise Exception("ffmpeg未安装，请先安装ffmpeg")
    
    cmd = [
        'ffmpeg', '-i', video_path, '-i', audio_path,
        '-c:v', 'copy', '-c:a', 'copy',
        '-strict', '-2', output_path
    ]
    
    print("正在合并音视频...")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    if result.returncode != 0:
        raise Exception(f"合并失败: {result.stderr}")
    print("合并完成")

def download_video(url, output_dir='.', quality='auto', default_video_quality='80', audio_quality='high', default_audio_quality='high', session_token=None):
    """下载视频"""
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化客户端
        client = BiliClient(session_token)
        
        # 提取BV号
        bvid = extract_bvid(url)
        print(f"提取到BV号: {bvid}")
        
        # 获取视频信息
        print("获取视频信息...")
        video_info = client.get_video_info(bvid)
        title = video_info['title']
        owner = video_info['owner']['name']
        pages = video_info['pages']
        
        print(f"视频标题: {title}")
        print(f"UP主: {owner}")
        print(f"分P数量: {len(pages)}")
        print(f"视频质量: {quality}")
        print(f"默认视频质量: {default_video_quality}")
        print(f"音频质量: {audio_quality}")
        print(f"默认音频质量: {default_audio_quality}")
        
        # 遍历所有分P
        for page in pages:
            cid = page['cid']
            page_title = page['part']
            print(f"\n下载分P: {page_title}")
            
            # 获取播放信息
            print("获取播放信息...")
            play_info = client.get_play_info(bvid, cid)
            
            if 'dash' not in play_info:
                raise Exception("未找到dash格式的播放信息")
            
            dash = play_info['dash']
            
            # 获取视频和音频URL
            video_url = get_video_url(dash['video'], quality, default_video_quality)
            audio_url = get_audio_url(dash, audio_quality, default_audio_quality)
            
            # 构建文件名
            safe_title = re.sub(r'[\\/:*?"<>|]', '_', title)
            video_path = os.path.join(output_dir, f"{safe_title}.video")
            audio_path = os.path.join(output_dir, f"{safe_title}.audio")
            output_path = os.path.join(output_dir, f"{safe_title}.mp4")
            danmaku_path = os.path.join(output_dir, f"{safe_title}.xml")
            
            # 下载视频和音频
            print("下载视频...")
            download_file(client, video_url, video_path, "视频")
            
            print("下载音频...")
            download_file(client, audio_url, audio_path, "音频")
            
            # 下载弹幕并转换为ASS格式
            print("下载弹幕...")
            danmaku_content = client.get_danmaku(cid)
            # 转换XML弹幕为ASS格式
            ass_content = convert_xml_to_ass(danmaku_content)
            # 修改文件路径为ASS格式
            ass_path = os.path.join(output_dir, f"{safe_title}.ass")
            with open(ass_path, 'w', encoding='utf-8') as f:
                f.write(ass_content)
            print(f"弹幕已保存到: {ass_path}")
            
            # 合并音视频
            merge_media(output_path, video_path, audio_path)
            
            # 删除临时文件
            os.remove(video_path)
            os.remove(audio_path)
            
            print(f"视频已保存到: {output_path}")
        
        print("\n所有分P下载完成！")
        
    except Exception as e:
        print(f"错误: {e}")
        raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Bilibili视频下载工具')
    parser.add_argument('url', help='视频URL')
    parser.add_argument('-o', '--output', default='.', help='输出目录')
    parser.add_argument('-q', '--quality', type=int, default=80, help='视频质量 (默认: 80, 对应1080P)')
    parser.add_argument('-a', '--audio-quality', default='high', choices=['high', 'medium', 'low'], help='音频质量 (默认: high)')
    
    args = parser.parse_args()
    
    download_video(args.url, args.output, args.quality, args.audio_quality)

if __name__ == '__main__':
    main()
