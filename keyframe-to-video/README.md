# Keyframe-to-Video Packer

将关键帧图片 + 字幕文本重新打包为固定帧率的"关键帧电影"，与 [video-subtitle-extractor](../video-subtitle-extractor) 形成完整闭环管线：

```
原始视频 → video-subtitle-extractor → 关键帧图片 + metadata.json
                                              ↓
                                  keyframe-to-video → 关键帧电影 (.mp4)
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基础用法

```bash
# 将 extractor 产出的关键帧打包为视频（每帧保持 2 秒，24 FPS）
python packer.py keyframes_output/metadata.json -o output.mp4

# 自定义保持时长和帧率
python packer.py metadata.json -o output.mp4 --duration 3 --fps 30

# 烧录硬字幕到画面底部
python packer.py metadata.json -o output.mp4 --burn-subtitle --font-size 32
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `metadata` | metadata.json 路径（位置参数） | 必填 |
| `-o, --output` | 输出视频路径（.mp4） | 必填 |
| `-d, --duration` | 每张关键帧保持时长（秒） | 2.0 |
| `--fps` | 输出视频帧率 | 24 |
| `--codec` | 视频编码器 fourcc | avc1 |
| `--burn-subtitle` | 启用硬字幕烧录 | 关闭 |
| `--font-size` | 字幕字号 | 28 |

## 工作原理

1. 读取 `metadata.json` 中的关键帧列表（按 `frame_index` 排序）
2. 对每张关键帧图片，复制 `duration × fps` 帧写入视频
3. 可选：使用 Pillow 将字幕文本烧录为硬字幕（黑色半透明背景条 + 白色文字）
4. 使用 OpenCV VideoWriter 输出 H.264 MP4 视频
5. 输出统计信息：关键帧数量、输出帧数、时长、文件大小

## 输入格式

`metadata.json` 由 `video-subtitle-extractor` 生成，关键结构：

```json
{
  "frames": [
    {
      "frame_index": 0,
      "timestamp": 0.0,
      "image": "frame_00000_00_00_00.000.jpg",
      "subtitle_text": "大家好，欢迎观看本视频",
      "subtitle_start": 0.0,
      "subtitle_end": 2.5
    }
  ]
}
```

同时兼容 `pairs` 键名和 `frame_id` 排序键。

## 编码器兼容性

按优先级自动选择编码器：

1. `avc1` — Windows 上推荐，兼容性最佳
2. `H264` — 通用 H.264
3. `X264` — x264 编码器
4. `mp4v` — MPEG-4（兜底）

可通过 `--codec` 手动指定。

## 中文路径支持

图片读取使用 `cv2.imdecode(np.fromfile())` 方式，完全兼容中文路径。备选方案为 Pillow → numpy 转换。

## 输出示例

```
=======================================================
打包完成！
关键帧数量:      156  (跳过 0 张)
输出帧数:        7488
输出时长:        312.0 秒 (5.2 分钟)
文件大小:        45.32 MB (47,520,128 字节)
编码器:          avc1
帧率:            24.0 FPS
耗时:            12.3 秒
输出视频:        D:\output\keyframe_movie.mp4
=======================================================
```
