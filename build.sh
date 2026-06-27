#!/bin/bash
# Qwen3-ASR CLI 构建脚本
# 使用 PyInstaller 打包为单文件可执行文件

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 检查 Python 环境
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
    echo "❌ 找不到 Python: $PYTHON"
    exit 1
fi

# 检查 PyInstaller
if ! "$PYTHON" -c "import PyInstaller" 2>/dev/null; then
    echo "🔧 安装 PyInstaller..."
    pip install pyinstaller
fi

# 清理旧的构建产物
echo "🧹 清理旧的构建..."
rm -rf build/ dist/ __pycache__/
find . -name "*.spec" -delete 2>/dev/null || true

# 构建
echo "🚀 构建 Qwen3-ASR CLI..."
"$PYTHON" -m PyInstaller --noconfirm \
    --onefile \
    --name "qwen3-asr" \
    --distpath dist \
    --add-data "$("$PYTHON" -c "import silero_vad; import os; print(os.path.dirname(silero_vad.__file__) + '/data')"):silero_vad/data" \
    --hidden-import qwen3_asr_toolkit \
    --hidden-import qwen3_asr_toolkit.llamacpp_asr \
    --hidden-import qwen3_asr_toolkit.audio_tools \
    --hidden-import qwen3_asr_toolkit.qwen3asr \
    --hidden-import onnxruntime \
    --hidden-import silero_vad \
    --hidden-import librosa \
    --hidden-import soundfile \
    --hidden-import srt \
    --hidden-import tqdm \
    --hidden-import openai \
    --hidden-import requests \
    --hidden-import numpy \
    --hidden-import audioread \
    --collect-data silero_vad \
    --collect-data qwen3_asr_toolkit \
    --collect-submodules onnxruntime \
    qwen3_asr_cli.py

# 检查产物
EXE="dist/qwen3-asr"
if [ -f "$EXE" ]; then
    SIZE_MB=$(du -sm "$EXE" | cut -f1)
    echo ""
    echo "✅ 构建成功！"
    echo "   可执行文件: $(pwd)/$EXE"
    echo "   大小: ${SIZE_MB} MB"
    echo ""
    echo "使用示例:"
    echo "  $EXE -i speech.wav -m qwen3-asr"
    echo "  $EXE -i video.mp4 -m qwen3-asr --llamacpp-url http://localhost:8080 -srt"
else
    echo "❌ 构建失败！"
    exit 1
fi
