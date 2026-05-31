# video-subtitle-extractor

视频关键帧 + 字幕提取工具。通过场景变化检测去重，结合字幕时间轴过滤，输出「关键帧图片 + 对应字幕文本」配对数据集。

## 原理

1. **场景检测**：相邻视频帧大量重复，通过 SSIM（结构相似度）逐帧比较，低于阈值则判定为场景变化，保留该帧作为关键帧。
2. **字幕匹配**：解析 SRT/ASS 字幕文件，根据关键帧的时间戳匹配对应字幕文本。
3. **输出**：无字幕匹配的关键帧被丢弃，最终输出帧图片 + `metadata.json`。

## 安装

```bash
pip install -r requirements.txt
```

- `scikit-image` 用于高精度 SSIM 计算；若无法安装，程序自动降级为基于 OpenCV 的 MSE（均方误差）方案。
- `pysrt` 用于 SRT 字幕解析；ASS 字幕使用内置正则解析。

## 命令行用法

```bash
python extractor.py <video_path> [options]
```

### 参数说明

| 参数 | 必需 | 说明 |
|------|------|------|
| `video_path` | 是 | 视频文件路径（支持中文路径） |
| `-s, --subtitle` | 否 | 字幕文件路径（.srt 或 .ass）。不提供则跳过字幕匹配，保留所有关键帧 |
| `-o, --output` | 否 | 输出目录，默认为视频同目录下的 `keyframes_output/` |
| `-t, --threshold` | 否 | SSIM 阈值（0~1），默认 0.85。值越低保留的帧越多 |
| `--mse-threshold` | 否 | MSE 降级方案的阈值，默认 500。值越高保留的帧越多（仅 skimage 不可用时生效） |

### 示例

```bash
# 基本用法：提取关键帧并匹配字幕
python extractor.py D:\video\demo.mp4 -s D:\video\demo.srt

# 指定输出目录和阈值
python extractor.py D:\video\demo.mp4 -s D:\video\demo.ass -o D:\output\result -t 0.80

# 无字幕模式：仅提取所有关键帧
python extractor.py D:\video\demo.mp4 -o D:\output\frames_only
```

## 输出结构

```
output_dir/
├── frame_00150_00.000.jpg
├── frame_00320_12.800.jpg
├── frame_00680_27.200.jpg
├── ...
└── metadata.json
```

### metadata.json 格式

```json
{
  "video_path": "...",
  "subtitle_path": "...",
  "fps": 25.0,
  "threshold": 0.85,
  "total_frames": 1250,
  "key_frames": 45,
  "subtitled_frames": 32,
  "elapsed_seconds": 12.5,
  "frames": [
    {
      "frame_index": 150,
      "timestamp": 6.0,
      "image": "frame_00150_06.000.jpg",
      "subtitle_text": "今天天气真好",
      "subtitle_start": 5.5,
      "subtitle_end": 7.2
    }
  ]
}
```

## 依赖

- Python 3.7+
- OpenCV（`opencv-python`）
- pysrt
- Pillow
- numpy
- tqdm
- scikit-image（可选，推荐）
