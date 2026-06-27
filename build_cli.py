#!/usr/bin/env python3
"""
构建 Qwen3-ASR CLI 单文件可执行文件
"""
import os
import sys
import shutil
import subprocess

# 激活 venv
VENV_PYTHON = os.path.expanduser("~/qwen3-env/bin/python3")
VENV_SITE = os.path.expanduser("~/qwen3-env/lib/python3.14/site-packages")

# 要打包的入口脚本
ENTRY_SCRIPT = os.path.join(os.path.dirname(__file__), "qwen3_asr_cli.py")
OUTPUT_NAME = "qwen3-asr"

# silero_vad 数据目录（包含 .onnx 模型文件）
SILERO_DATA = os.path.join(VENV_SITE, "silero_vad", "data")

# 输出目录
DIST_DIR = os.path.join(os.path.dirname(__file__), "dist")

def build():
    # 清理旧的构建产物
    for d in ["build", "dist", "__pycache__"]:
        p = os.path.join(os.path.dirname(__file__), d)
        if os.path.exists(p):
            shutil.rmtree(p)

    os.makedirs(DIST_DIR, exist_ok=True)

    cmd = [
        VENV_PYTHON, "-m", "PyInstaller",
        "--onefile",
        "--name", OUTPUT_NAME,
        "--distpath", DIST_DIR,
        "--add-data", f"{SILERO_DATA}{os.pathsep}silero_vad/data",
        "--hidden-import", "qwen3_asr_toolkit",
        "--hidden-import", "qwen3_asr_toolkit.llamacpp_asr",
        "--hidden-import", "qwen3_asr_toolkit.audio_tools",
        "--hidden-import", "qwen3_asr_toolkit.qwen3asr",
        "--hidden-import", "onnxruntime",
        "--hidden-import", "silero_vad",
        "--hidden-import", "librosa",
        "--hidden-import", "soundfile",
        "--hidden-import", "srt",
        "--hidden-import", "tqdm",
        "--hidden-import", "openai",
        "--hidden-import", "requests",
        "--hidden-import", "numpy",
        "--hidden-import", "audioread",
        "--collect-data", "silero_vad",
        "--collect-data", "qwen3_asr_toolkit",
        "--collect-submodules", "onnxruntime",
        ENTRY_SCRIPT,
    ]

    print("🚀 开始构建 Qwen3-ASR CLI...")
    print(f"   入口: {ENTRY_SCRIPT}")
    print(f"   输出: {os.path.join(DIST_DIR, OUTPUT_NAME)}")
    print()

    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    if result.returncode != 0:
        print(f"\n❌ 构建失败 (exit code {result.returncode})")
        sys.exit(1)

    # 检查产物
    exe_path = os.path.join(DIST_DIR, OUTPUT_NAME)
    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / 1024 / 1024
        print(f"\n✅ 构建成功！")
        print(f"   可执行文件: {exe_path}")
        print(f"   大小: {size_mb:.1f} MB")
    else:
        print(f"\n⚠️  产物未找到: {exe_path}")

if __name__ == "__main__":
    build()
