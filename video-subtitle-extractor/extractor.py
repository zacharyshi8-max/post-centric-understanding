#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频关键帧 + 字幕提取工具
原理：场景变化检测（SSIM）去重 → 字幕时间轴匹配 → 输出关键帧图片 + 字幕文本配对数据集
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 可选依赖：scikit-image 提供更精确的 SSIM，降级方案为 OpenCV MSE
# ---------------------------------------------------------------------------
try:
    from skimage.metrics import structural_similarity as ssim

    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# ---------------------------------------------------------------------------
# 可选依赖：pysrt 用于 SRT 解析
# ---------------------------------------------------------------------------
try:
    import pysrt

    HAS_PYSRT = True
except ImportError:
    HAS_PYSRT = False


# ===================================================================
# 时间工具函数
# ===================================================================

def timestamp_to_seconds(ts: str) -> float:
    """将字幕时间戳 HH:MM:SS,mmm 或 H:MM:SS.cc 转换为秒（浮点数）。

    支持格式：
      - SRT:  00:01:23,456  (逗号分隔毫秒，3位)
      - ASS:   0:00:01.23   (小数点分隔百分秒，2位)
      - ASS:   0:00:01.234  (小数点分隔毫秒，3位)
    """
    ts = ts.strip()
    # 分离时分秒与毫秒/百分秒部分
    if "," in ts:
        hms, frac = ts.split(",")
    elif "." in ts:
        hms, frac = ts.split(".")
    else:
        hms = ts
        frac = "0"

    parts = hms.split(":")
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h = "0"
        m, s = parts
    else:
        raise ValueError(f"无法解析时间戳: {ts}")

    h, m, s = int(h), int(m), int(s)

    # 毫秒处理：3位就是毫秒，2位是百分秒需×10
    frac = frac.ljust(3, "0")[:3]  # 补齐到3位
    ms = int(frac)

    return h * 3600 + m * 60 + s + ms / 1000.0


def seconds_to_timestamp(seconds: float) -> str:
    """秒数转为 HH:MM:SS.mmm 格式字符串（用于图片命名）。"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}_{m:02d}_{s:02d}.{ms:03d}"


# ===================================================================
# 字幕解析
# ===================================================================

def parse_srt(file_path: str) -> list[dict]:
    """解析 SRT 字幕，返回字幕条目列表。

    每条记录: {start, end, text, raw_start, raw_end}
    start/end 为秒（float），text 为字幕文本。
    """
    if not HAS_PYSRT:
        raise ImportError("pysrt 未安装，无法解析 SRT 字幕。请执行: pip install pysrt")

    subs = pysrt.open(file_path, encoding="utf-8")
    entries = []
    for sub in subs:
        start_sec = (sub.start.hours * 3600 + sub.start.minutes * 60 +
                     sub.start.seconds + sub.start.milliseconds / 1000.0)
        end_sec = (sub.end.hours * 3600 + sub.end.minutes * 60 +
                   sub.end.seconds + sub.end.milliseconds / 1000.0)
        text = sub.text.replace("\n", " ").strip()
        entries.append({
            "start": start_sec,
            "end": end_sec,
            "text": text,
            "raw_start": str(sub.start),
            "raw_end": str(sub.end),
        })
    return entries


# ASS Dialogue 行正则：匹配 "Dialogue: Layer,Start,End,Style,..."
# 注意 ASS 中 Start/End 格式为 H:MM:SS.cc（百分秒）
_DIALOGUE_RE = re.compile(
    r"^Dialogue:\s*"                           # Dialogue: 开头
    r"(?:[^,]*,\s*)?"                          # Layer (可选，有些文件省略)
    r"([0-9]+:[0-9]{2}:[0-9]{2}[.,][0-9]+),\s*"  # Start
    r"([0-9]+:[0-9]{2}:[0-9]{2}[.,][0-9]+),\s*"  # End
    r"(?:[^,]*,\s*){6}"                        # Style, Name, MarginL, MarginR, MarginV, Effect
    r"(.*?)$"                                  # Text
)


def parse_ass(file_path: str) -> list[dict]:
    """解析 ASS 字幕，返回字幕条目列表。

    每条记录: {start, end, text, raw_start, raw_end}
    start/end 为秒（float），text 为字幕文本（去除特效标签如 {\\...}）。
    """
    # 尝试多种编码
    content = None
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb2312", "latin-1"):
        try:
            with open(file_path, "r", encoding=enc) as f:
                content = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    if content is None:
        raise ValueError(f"无法读取 ASS 字幕文件: {file_path}")

    entries = []
    for line in content.splitlines():
        line = line.strip()
        if not line.startswith("Dialogue:"):
            continue

        m = _DIALOGUE_RE.match(line)
        if not m:
            continue

        start_str, end_str, text = m.group(1), m.group(2), m.group(3).strip()

        # 去除 ASS 特效标签 {\xxx} 和 \N 换行
        text = re.sub(r"\{[^}]*\}", "", text)
        text = text.replace("\\N", " ").replace("\\n", " ").strip()

        if not text:
            continue

        try:
            start_sec = timestamp_to_seconds(start_str)
            end_sec = timestamp_to_seconds(end_str)
        except ValueError:
            continue

        entries.append({
            "start": start_sec,
            "end": end_sec,
            "text": text,
            "raw_start": start_str,
            "raw_end": end_str,
        })

    return entries


def parse_subtitle(file_path: str) -> list[dict]:
    """自动识别字幕格式并解析。"""
    ext = Path(file_path).suffix.lower()
    if ext == ".srt":
        return parse_srt(file_path)
    elif ext in (".ass", ".ssa"):
        return parse_ass(file_path)
    else:
        raise ValueError(f"不支持的字幕格式: {ext}（仅支持 .srt / .ass / .ssa）")


# ===================================================================
# 场景变化检测
# ===================================================================

def _ssim_score(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """使用 skimage 计算两帧的 SSIM。"""
    # 转为灰度
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    return ssim(gray_a, gray_b, data_range=255)


def _mse_score(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """使用 OpenCV 计算两帧的均方误差（MSE），作为 SSIM 的降级方案。"""
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    err = np.sum((gray_a.astype("float") - gray_b.astype("float")) ** 2)
    err /= float(gray_a.shape[0] * gray_a.shape[1])
    return err


def is_scene_change(prev_frame: np.ndarray, curr_frame: np.ndarray,
                    threshold: float, mse_threshold: float) -> bool:
    """判断是否发生场景变化。

    优先使用 skimage SSIM；不可用时降级为 OpenCV MSE。
    - SSIM < threshold   → 场景变化
    - SSE  > mse_threshold → 场景变化
    """
    if HAS_SKIMAGE:
        score = _ssim_score(prev_frame, curr_frame)
        return score < threshold
    else:
        score = _mse_score(prev_frame, curr_frame)
        return score > mse_threshold


# ===================================================================
# 字幕匹配
# ===================================================================

def find_subtitle(frame_timestamp: float, subtitles: list[dict]) -> dict | None:
    """根据帧时间戳查找匹配的字幕条目。

    使用二分查找定位字幕区间。若帧时间落在某条字幕的 [start, end] 内，返回该字幕。
    若多条字幕重叠覆盖同一时间点，返回第一条匹配的。
    """
    lo, hi = 0, len(subtitles) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        sub = subtitles[mid]
        if sub["start"] <= frame_timestamp <= sub["end"]:
            return sub
        elif frame_timestamp < sub["start"]:
            hi = mid - 1
        else:
            lo = mid + 1

    return None


# ===================================================================
# 主流程
# ===================================================================

def extract_keyframes(video_path: str, subtitle_path: str | None,
                      output_dir: str, threshold: float,
                      mse_threshold: float) -> None:
    """执行关键帧提取 + 字幕匹配全流程。"""
    start_time = time.time()

    # ---- 打开视频 ----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频文件: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        print("[ERROR] 无法获取视频帧率")
        cap.release()
        sys.exit(1)

    print(f"视频: {video_path}")
    print(f"帧率: {fps:.2f} FPS | 总帧数: {total_frames}")

    # ---- 解析字幕 ----
    subtitles = []
    has_subtitle = subtitle_path is not None
    if has_subtitle:
        print(f"字幕: {subtitle_path}")
        try:
            subtitles = parse_subtitle(subtitle_path)
            # 按 start 排序确保二分查找正确
            subtitles.sort(key=lambda x: x["start"])
            print(f"字幕条目: {len(subtitles)}")
        except Exception as e:
            print(f"[WARN] 字幕解析失败: {e}")
            print("       将跳过字幕匹配，保留所有关键帧。")
            has_subtitle = False
            subtitles = []

    # ---- 创建输出目录 ----
    os.makedirs(output_dir, exist_ok=True)

    # ---- 逐帧处理 ----
    method_name = "SSIM" if HAS_SKIMAGE else "MSE"
    print(f"\n场景检测方法: {method_name} | 阈值: {threshold if HAS_SKIMAGE else mse_threshold}")
    print(f"输出目录: {output_dir}\n")

    keyframe_count = 0
    subtitled_count = 0
    metadata_frames = []

    prev_keyframe = None
    frame_idx = 0

    pbar = tqdm(total=total_frames, desc="处理中", unit="帧")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps

        is_keyframe = False

        if prev_keyframe is None:
            # 第一帧始终是关键帧
            is_keyframe = True
        else:
            if is_scene_change(prev_keyframe, frame, threshold, mse_threshold):
                is_keyframe = True

        if is_keyframe:
            keyframe_count += 1
            prev_keyframe = frame.copy()

            # 字幕匹配
            sub_text = ""
            sub_start = 0.0
            sub_end = 0.0
            if has_subtitle and subtitles:
                matched = find_subtitle(timestamp, subtitles)
                if matched:
                    subtitled_count += 1
                    sub_text = matched["text"]
                    sub_start = matched["start"]
                    sub_end = matched["end"]
                else:
                    # 无字幕匹配 → 丢弃该关键帧
                    frame_idx += 1
                    pbar.update(1)
                    continue
            elif not has_subtitle:
                # 无字幕模式：保留所有关键帧
                subtitled_count += 1

            # 保存图片
            ts_str = seconds_to_timestamp(timestamp)
            image_name = f"frame_{frame_idx:05d}_{ts_str}.jpg"
            image_path = os.path.join(output_dir, image_name)
            cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(image_path)

            metadata_frames.append({
                "frame_index": frame_idx,
                "timestamp": round(timestamp, 3),
                "image": image_name,
                "subtitle_text": sub_text,
                "subtitle_start": round(sub_start, 3),
                "subtitle_end": round(sub_end, 3),
            })

        frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    # ---- 写入 metadata.json ----
    elapsed = time.time() - start_time
    metadata = {
        "video_path": os.path.abspath(video_path),
        "subtitle_path": os.path.abspath(subtitle_path) if subtitle_path else None,
        "fps": fps,
        "threshold": threshold if HAS_SKIMAGE else None,
        "mse_threshold": None if HAS_SKIMAGE else mse_threshold,
        "ssim_method": "skimage" if HAS_SKIMAGE else "opencv_mse",
        "total_frames": total_frames,
        "key_frames": keyframe_count,
        "subtitled_frames": subtitled_count,
        "elapsed_seconds": round(elapsed, 2),
        "frames": metadata_frames,
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # ---- 统计输出 ----
    print(f"\n{'='*50}")
    print(f"处理完成！")
    print(f"总帧数:        {total_frames}")
    print(f"关键帧数:      {keyframe_count}")
    print(f"有字幕帧数:    {subtitled_count}")
    print(f"丢弃帧数:      {keyframe_count - subtitled_count}")
    print(f"耗时:          {elapsed:.1f} 秒")
    print(f"输出目录:      {os.path.abspath(output_dir)}")
    print(f"Metadata:      {metadata_path}")
    print(f"{'='*50}")


# ===================================================================
# CLI 入口
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="视频关键帧 + 字幕提取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video_path", help="视频文件路径")
    parser.add_argument("-s", "--subtitle", default=None,
                        help="字幕文件路径（.srt / .ass），不提供则保留所有关键帧")
    parser.add_argument("-o", "--output", default=None,
                        help="输出目录，默认视频同目录下的 keyframes_output/")
    parser.add_argument("-t", "--threshold", type=float, default=0.85,
                        help="SSIM 阈值（0~1），默认 0.85。值越低保留的帧越多")
    parser.add_argument("--mse-threshold", type=float, default=500.0,
                        help="MSE 阈值（降级方案），默认 500。值越高保留的帧越多")

    args = parser.parse_args()

    # 校验参数
    if not os.path.isfile(args.video_path):
        print(f"[ERROR] 视频文件不存在: {args.video_path}")
        sys.exit(1)

    if args.subtitle and not os.path.isfile(args.subtitle):
        print(f"[ERROR] 字幕文件不存在: {args.subtitle}")
        sys.exit(1)

    if not (0 < args.threshold <= 1):
        print("[ERROR] SSIM 阈值必须在 (0, 1] 区间")
        sys.exit(1)

    # 默认输出目录
    if args.output is None:
        video_dir = os.path.dirname(os.path.abspath(args.video_path))
        args.output = os.path.join(video_dir, "keyframes_output")

    extract_keyframes(
        video_path=args.video_path,
        subtitle_path=args.subtitle,
        output_dir=args.output,
        threshold=args.threshold,
        mse_threshold=args.mse_threshold,
    )


if __name__ == "__main__":
    main()
