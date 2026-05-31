#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
关键帧电影生成工具 (Keyframe-to-Video Packer)
将关键帧图片 + 字幕文本重新打包为固定帧率的视频。
与 video-subtitle-extractor 形成完整闭环管线。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 可选依赖：Pillow 用于硬字幕烧录
# ---------------------------------------------------------------------------
try:
    from PIL import Image, ImageDraw, ImageFont

    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False


# ===================================================================
# 中文字体查找
# ===================================================================

_WIN_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",       # 微软雅黑
    "C:/Windows/Fonts/msyhbd.ttc",     # 微软雅黑 Bold
    "C:/Windows/Fonts/simhei.ttf",     # 黑体
    "C:/Windows/Fonts/simsun.ttc",     # 宋体
    "C:/Windows/Fonts/simkai.ttf",     # 楷体
    "C:/Windows/Fonts/STKAITI.TTF",    # 华文楷体
    "C:/Windows/Fonts/Deng.ttf",       # 等线
    "C:/Windows/Fonts/Dengb.ttf",      # 等线 Bold
    "C:/Windows/Fonts/STFANGSO.TTF",   # 华文仿宋
]

_MAC_FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]


def _find_chinese_font() -> str | None:
    """在系统中查找可用中文字体，返回第一个找到的路径。"""
    if sys.platform == "win32":
        candidates = _WIN_FONT_CANDIDATES
    elif sys.platform == "darwin":
        candidates = _MAC_FONT_CANDIDATES
    else:
        # Linux: 尝试常见路径
        candidates = [
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        ]

    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


# ===================================================================
# 图片读取（中文路径兼容）
# ===================================================================

def imread_cn(file_path: str) -> np.ndarray | None:
    """读取图片，兼容中文路径。

    使用 cv2.imdecode(np.fromfile()) 方式绕过 OpenCV 的编码限制。
    同时利用 Pillow 作为备选方案。
    """
    try:
        data = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass

    # 备选：Pillow 读取 → numpy 转换
    if HAS_PILLOW:
        try:
            pil_img = Image.open(file_path).convert("RGB")
            img = np.array(pil_img)
            # Pillow 读取为 RGB，OpenCV 使用 BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        except Exception:
            pass

    return None


# ===================================================================
# 元数据读取
# ===================================================================

def load_metadata(metadata_path: str) -> tuple[list[dict], str]:
    """读取 metadata.json，返回 (帧数据列表, 图片目录绝对路径)。

    兼容 extractor.py 的 "frames" 格式：
      {frame_index, timestamp, image, subtitle_text, subtitle_start, subtitle_end}

    同时也兼容 "pairs" 格式（frame_id 作为排序键）。
    """
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 兼容 "frames" 和 "pairs" 两种键名
    frames = meta.get("frames") or meta.get("pairs") or []

    if not frames:
        raise ValueError("metadata.json 中未找到 'frames' 或 'pairs' 数组")

    # 图片目录：默认为 metadata.json 所在目录
    image_dir = os.path.dirname(os.path.abspath(metadata_path))

    # 按 frame_index 或 frame_id 排序
    sort_key = "frame_index" if "frame_index" in frames[0] else "frame_id"
    frames.sort(key=lambda x: x.get(sort_key, 0))

    return frames, image_dir


# ===================================================================
# 硬字幕烧录
# ===================================================================

def burn_subtitle(frame: np.ndarray, text: str, font_path: str | None,
                  font_size: int = 28, bar_height_ratio: float = 0.12) -> np.ndarray:
    """在帧画面底部烧录字幕文本。

    绘制黑色半透明背景条 + 白色文字，居中对齐。
    若 Pillow 不可用或 text 为空，返回原帧。

    Args:
        frame: OpenCV BGR 图像 (H, W, 3)
        text: 字幕文本
        font_path: 中文字体路径，None 则使用 Pillow 默认字体
        font_size: 字号
        bar_height_ratio: 背景条高度占帧高度的比例

    Returns:
        烧录字幕后的帧
    """
    if not HAS_PILLOW or not text.strip():
        return frame

    h, w = frame.shape[:2]

    # BGR → RGB → Pillow
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)

    # 加载字体
    try:
        if font_path and os.path.isfile(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # 文字换行处理
    max_text_width = int(w * 0.9)
    lines = _wrap_text(text, font, draw, max_text_width)

    # 计算背景条尺寸
    line_height = font_size + 6
    bar_height = max(int(h * bar_height_ratio), line_height * len(lines) + 20)
    bar_top = h - bar_height

    # 与 extractor 版式一致：纯黑背景条 + 白色文字
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([(0, bar_top), (w, h)], fill=(0, 0, 0, 180))
    pil_img = Image.alpha_composite(pil_img.convert("RGBA"), overlay)

    # 绘制文字
    draw = ImageDraw.Draw(pil_img)
    y_start = bar_top + (bar_height - line_height * len(lines)) // 2
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        x = (w - text_w) // 2
        y = y_start + i * line_height
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))

    # RGBA → RGB → BGR → numpy
    pil_rgb = pil_img.convert("RGB")
    result = np.array(pil_rgb)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    return result


def _wrap_text(text: str, font, draw, max_width: int) -> list[str]:
    """按像素宽度折行。"""
    lines = []
    current_line = ""
    for char in text:
        test_line = current_line + char
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = char
    if current_line:
        lines.append(current_line)
    return lines if lines else [text]


# ===================================================================
# 视频编码
# ===================================================================

def _create_video_writer(output_path: str, fps: float, width: int, height: int,
                         codec_priority: list[str] | None = None) -> tuple[cv2.VideoWriter, str]:
    """创建 VideoWriter，按优先级尝试编码器。

    Returns:
        (VideoWriter, 实际使用的 fourcc 字符串)
    """
    if codec_priority is None:
        codec_priority = ["avc1", "H264", "X264", "mp4v"]

    last_error = None
    for fourcc_str in codec_priority:
        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if writer.isOpened():
                return writer, fourcc_str
            writer.release()
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(
        f"无法创建 VideoWriter，已尝试编码器: {codec_priority}。"
        f"最后错误: {last_error}"
    )


# ===================================================================
# 主流程
# ===================================================================

def pack_keyframes(
    metadata_path: str,
    output_video: str,
    duration: float = 2.0,
    fps: float = 24.0,
    codec: str = "avc1",
    burn_sub: bool = False,
    font_size: int = 28,
) -> None:
    """将关键帧图片打包为视频。"""
    start_time = time.time()

    # ---- 加载元数据 ----
    print(f"读取元数据: {metadata_path}")
    frames_data, image_dir = load_metadata(metadata_path)

    keyframe_count = len(frames_data)
    print(f"关键帧数量: {keyframe_count}")

    if keyframe_count == 0:
        print("[ERROR] metadata 中没有关键帧数据")
        sys.exit(1)

    # ---- 检查编码器支持 ----
    codec_priority = [codec] + [c for c in ["avc1", "H264", "X264", "mp4v"] if c != codec]

    # ---- 中文字体 ----
    font_path = None
    if burn_sub and HAS_PILLOW:
        font_path = _find_chinese_font()
        if font_path:
            print(f"中文字体: {font_path}")
        else:
            print("[WARN] 未找到系统中文字体，字幕将使用默认字体（可能无法正确显示中文）")

    # ---- 第一帧：确定视频尺寸 ----
    first_frame_path = os.path.join(image_dir, frames_data[0]["image"])
    first_frame = imread_cn(first_frame_path)
    if first_frame is None:
        print(f"[ERROR] 无法读取第一帧图片: {first_frame_path}")
        sys.exit(1)

    height, width = first_frame.shape[:2]
    print(f"分辨率: {width}x{height} | 输出帧率: {fps} FPS | 每帧保持: {duration}s")

    # ---- 创建 VideoWriter ----
    os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)

    writer, used_codec = _create_video_writer(
        output_path=output_video,
        fps=fps,
        width=width,
        height=height,
        codec_priority=codec_priority,
    )
    print(f"编码器: {used_codec}")

    # ---- 逐帧处理 ----
    hold_frames = int(duration * fps)
    total_output_frames = keyframe_count * hold_frames
    skipped = 0

    print(f"\n每关键帧 → {hold_frames} 帧 × {keyframe_count} = {total_output_frames} 输出帧\n")

    pbar = tqdm(total=keyframe_count, desc="打包关键帧", unit="帧")

    for item in frames_data:
        # 兼容 frame_index / frame_id
        frame_idx = item.get("frame_index", item.get("frame_id", "?"))
        image_name = item.get("image", "")
        subtitle_text = item.get("subtitle_text", "")

        image_path = os.path.join(image_dir, image_name)
        img = imread_cn(image_path)

        if img is None:
            print(f"[WARN] 跳过无法读取的图片: {image_path}")
            skipped += 1
            pbar.update(1)
            continue

        # 烧录字幕
        if burn_sub and subtitle_text:
            img = burn_subtitle(img, subtitle_text, font_path, font_size=font_size)

        # 复制 N 帧写入视频
        for _ in range(hold_frames):
            writer.write(img)

        pbar.update(1)

    pbar.close()

    # ---- 收尾 ----
    writer.release()
    elapsed = time.time() - start_time

    actual_keyframes = keyframe_count - skipped
    actual_output_frames = actual_keyframes * hold_frames
    file_size = os.path.getsize(output_video) if os.path.isfile(output_video) else 0
    file_size_mb = file_size / (1024 * 1024)
    video_duration = actual_output_frames / fps

    print(f"\n{'='*55}")
    print(f"打包完成！")
    print(f"关键帧数量:      {actual_keyframes}  (跳过 {skipped} 张)")
    print(f"输出帧数:        {actual_output_frames}")
    print(f"输出时长:        {video_duration:.1f} 秒 ({video_duration/60:.1f} 分钟)")
    print(f"文件大小:        {file_size_mb:.2f} MB ({file_size:,} 字节)")
    print(f"编码器:          {used_codec}")
    print(f"帧率:            {fps} FPS")
    print(f"耗时:            {elapsed:.1f} 秒")
    print(f"输出视频:        {os.path.abspath(output_video)}")
    print(f"{'='*55}")


# ===================================================================
# CLI 入口
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="关键帧电影生成工具 — 将关键帧图片 + 字幕打包为固定帧率视频",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础用法
  python packer.py keyframes_output/metadata.json -o output.mp4

  # 自定义参数
  python packer.py metadata.json -o output.mp4 --duration 3 --fps 30

  # 烧录字幕
  python packer.py metadata.json -o output.mp4 --burn-subtitle --font-size 32
        """,
    )
    parser.add_argument(
        "metadata", help="metadata.json 文件路径（由 extractor.py 生成）"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="输出视频路径（.mp4）"
    )
    parser.add_argument(
        "-d", "--duration", type=float, default=2.0,
        help="每张关键帧的保持时长，单位秒（默认 2.0）"
    )
    parser.add_argument(
        "--fps", type=float, default=24.0,
        help="输出视频帧率（默认 24）"
    )
    parser.add_argument(
        "--codec", type=str, default="avc1",
        choices=["avc1", "H264", "X264", "mp4v"],
        help="视频编码器 fourcc（默认 avc1）"
    )
    parser.add_argument(
        "--burn-subtitle", action="store_true", default=False,
        help="将字幕文本烧录为硬字幕（需要 Pillow）"
    )
    parser.add_argument(
        "--font-size", type=int, default=28,
        help="字幕字号（默认 28）"
    )

    args = parser.parse_args()

    # 参数校验
    if not os.path.isfile(args.metadata):
        print(f"[ERROR] metadata 文件不存在: {args.metadata}")
        sys.exit(1)

    if args.duration <= 0:
        print("[ERROR] --duration 必须 > 0")
        sys.exit(1)

    if args.fps <= 0:
        print("[ERROR] --fps 必须 > 0")
        sys.exit(1)

    if args.burn_subtitle and not HAS_PILLOW:
        print("[ERROR] --burn-subtitle 需要 Pillow 库，请执行: pip install Pillow")
        sys.exit(1)

    pack_keyframes(
        metadata_path=args.metadata,
        output_video=args.output,
        duration=args.duration,
        fps=args.fps,
        codec=args.codec,
        burn_sub=args.burn_subtitle,
        font_size=args.font_size,
    )


if __name__ == "__main__":
    main()
